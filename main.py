from __future__ import annotations

import argparse
import csv
import html
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wpilog-current-analysis-mpl"))

import matplotlib
import numpy as np
import pandas as pd
from wpiutil.log import DataLogReader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ENABLED_KEY = "NT:/AdvantageKit/DriverStation/Enabled"
AUTONOMOUS_KEY = "NT:/AdvantageKit/DriverStation/Autonomous"
REAL_OUTPUTS_PREFIX = "NT:/AdvantageKit/RealOutputs/"
CURRENT_SUFFIXES = ("Supply Current", "Stator Current")
PHASES = ("enabled", "auto", "teleop")

MOTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "Flywheel Left": ("Flywheel Left Lead", "Flywheel Left Follower"),
    "Flywheel Right": ("Flywheel Right Lead", "Flywheel Right Follower"),
    "Hood Left": ("Hood Left",),
    "Hood Right": ("Hood Right",),
    "Turret Left": ("Turret Left",),
    "Turret Right": ("Turret Right",),
    "Ground Pivot": ("Ground Pivot",),
    "Ground Rollers": ("Ground Roller Lead", "Ground Roller Follower"),
    "Roller Floor": ("Roller Floor",),
    "B2": ("B2",),
    "Kicker": ("Kicker",),
}

DERIVED_GROUPS: dict[str, tuple[str, ...]] = {
    "Flywheels": ("Flywheel Left", "Flywheel Right"),
    "Hoods": ("Hood Left", "Hood Right"),
    "Turrets": ("Turret Left", "Turret Right"),
    "Intake": ("Ground Pivot", "Ground Rollers"),
    "Indexer": ("Roller Floor", "B2", "Kicker"),
    "Superstructure": tuple(MOTOR_GROUPS.keys()),
}

GLOBAL_CHANNELS: dict[str, str] = {
    "Battery Current": "NT:/AdvantageKit/SystemStats/BatteryCurrent",
    "Battery Voltage": "NT:/AdvantageKit/SystemStats/BatteryVoltage",
    "PDH Total Current": "NT:/AdvantageKit/PowerDistribution/TotalCurrent",
}

DIRECT_AGGREGATES: dict[str, dict[str, str]] = {
    "Drive": {
        "Supply Current": "NT:/AdvantageKit/RealOutputs/Drive/TotalDriveSupplyCurrent",
        "Stator Current": "NT:/AdvantageKit/RealOutputs/Drive/TotalDriveStatorCurrent",
        "Supply Current Abs": "NT:/AdvantageKit/RealOutputs/Drive/TotalDriveSupplyCurrentAbs",
        "Power": "NT:/AdvantageKit/RealOutputs/Drive/TotalDrivePower",
    }
}

CORE_MECHANISMS = list(MOTOR_GROUPS.keys()) + ["Drive"]
DERIVED_SUMMARIES = list(DERIVED_GROUPS.keys()) + ["Drive"]


def current_topic(motor_name: str, current_type: str) -> str:
    return f"{REAL_OUTPUTS_PREFIX}{motor_name}/{current_type}"


def bucket_label(index: int, bucket_count: int) -> str:
    start = int(round(index * 100 / bucket_count))
    end = int(round((index + 1) * 100 / bucket_count))
    return f"{start}_{end}"


def bucket_column_names(bucket_count: int) -> list[str]:
    return [f"bucket_{bucket_label(index, bucket_count)}_avg" for index in range(bucket_count)]


def duty_cycle_from_buckets(bucket_values: list[float], threshold: float) -> float:
    finite = [value for value in bucket_values if not math.isnan(value)]
    if not finite:
        return math.nan
    return float(sum(1 for value in finite if value >= threshold) / len(finite))


@dataclass
class WeightedIntervalStats:
    intervals: list[tuple[float, float]] = field(default_factory=list)
    total_duration: float = 0.0
    weighted_sum: float = 0.0
    max_value: float = 0.0
    update_count: int = 0

    def add_interval(self, value: float, duration_seconds: float) -> None:
        if duration_seconds <= 0.0:
            return
        self.intervals.append((value, duration_seconds))
        self.total_duration += duration_seconds
        self.weighted_sum += value * duration_seconds
        self.max_value = max(self.max_value, value)

    def note_update(self) -> None:
        self.update_count += 1

    def average(self) -> float:
        if self.total_duration <= 0.0:
            return math.nan
        return self.weighted_sum / self.total_duration

    def weighted_percentile(self, percentile: float) -> float:
        if self.total_duration <= 0.0:
            return math.nan
        target = self.total_duration * percentile
        elapsed = 0.0
        sorted_intervals = sorted(self.intervals, key=lambda item: item[0])
        for value, duration in sorted_intervals:
            elapsed += duration
            if elapsed >= target:
                return value
        return sorted_intervals[-1][0]

    def bucket_averages(self, bucket_count: int) -> list[float]:
        if self.total_duration <= 0.0:
            return [math.nan] * bucket_count

        sorted_intervals = sorted(self.intervals, key=lambda item: item[0])
        bucket_width = self.total_duration / bucket_count
        bucket_weighted_sum = [0.0] * bucket_count
        bucket_duration = [0.0] * bucket_count
        bucket_index = 0
        bucket_remaining = bucket_width

        for value, duration in sorted_intervals:
            remaining = duration
            while remaining > 1e-12 and bucket_index < bucket_count:
                slice_duration = remaining if bucket_index == bucket_count - 1 else min(remaining, bucket_remaining)
                bucket_weighted_sum[bucket_index] += value * slice_duration
                bucket_duration[bucket_index] += slice_duration
                remaining -= slice_duration

                if bucket_index < bucket_count - 1:
                    bucket_remaining -= slice_duration
                    if bucket_remaining <= 1e-12:
                        bucket_index += 1
                        bucket_remaining = bucket_width

        return [
            bucket_weighted_sum[index] / bucket_duration[index] if bucket_duration[index] > 0.0 else math.nan
            for index in range(bucket_count)
        ]


@dataclass
class ChannelState:
    value: float = 0.0
    seen: bool = False


@dataclass
class PhaseWindow:
    stats: WeightedIntervalStats = field(default_factory=WeightedIntervalStats)
    active_since: int | None = None
    segments: list[tuple[int, int, float]] = field(default_factory=list)


@dataclass
class AggregateChannel:
    member_topics: tuple[str, ...]
    present_topics: set[str] = field(default_factory=set)
    current_total: float = 0.0
    phases: dict[str, PhaseWindow] = field(
        default_factory=lambda: {phase: PhaseWindow() for phase in PHASES}
    )


@dataclass
class AnalysisResult:
    rows: list[dict[str, str]]
    segments: dict[tuple[str, str, str], list[tuple[int, int, float]]]
    log_start_timestamp: int


def build_aggregate_specs() -> dict[str, tuple[str, ...]]:
    specs: dict[str, tuple[str, ...]] = {}

    for subsystem, motors in MOTOR_GROUPS.items():
        for current_type in CURRENT_SUFFIXES:
            specs[f"{subsystem}|{current_type}"] = tuple(current_topic(motor, current_type) for motor in motors)

    for subsystem, members in DERIVED_GROUPS.items():
        for current_type in CURRENT_SUFFIXES:
            topics: list[str] = []
            for member in members:
                topics.extend(current_topic(motor, current_type) for motor in MOTOR_GROUPS[member])
            specs[f"{subsystem}|{current_type}"] = tuple(topics)

    for subsystem, metrics in DIRECT_AGGREGATES.items():
        for metric, topic in metrics.items():
            specs[f"{subsystem}|{metric}"] = (topic,)

    return specs


def format_number(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


def discover_topics(log_path: Path) -> dict[int, tuple[str, str]]:
    entry_map: dict[int, tuple[str, str]] = {}
    direct_topics = {topic for metrics in DIRECT_AGGREGATES.values() for topic in metrics.values()}
    reader = DataLogReader(str(log_path))
    if not reader.isValid():
        raise ValueError(f"{log_path} is not a valid WPILOG file")

    for record in reader:
        if not record.isStart():
            continue
        start_data = record.getStartData()
        name = start_data.name
        data_type = start_data.type
        if name in {ENABLED_KEY, AUTONOMOUS_KEY} or name in GLOBAL_CHANNELS.values() or name in direct_topics:
            entry_map[start_data.entry] = (name, data_type)
            continue
        if name.startswith(REAL_OUTPUTS_PREFIX) and name.endswith(CURRENT_SUFFIXES):
            entry_map[start_data.entry] = (name, data_type)
    return entry_map


def phase_active(phase: str, enabled: bool, autonomous: bool) -> bool:
    if phase == "enabled":
        return enabled
    if phase == "auto":
        return enabled and autonomous
    if phase == "teleop":
        return enabled and not autonomous
    raise ValueError(f"Unknown phase {phase}")


def analyze_log(log_path: Path, bucket_count: int) -> AnalysisResult:
    entry_map = discover_topics(log_path)
    aggregate_specs = build_aggregate_specs()
    channel_states: dict[str, ChannelState] = {}
    channels_to_aggregates: dict[str, list[AggregateChannel]] = {}
    aggregates: dict[str, AggregateChannel] = {}

    for aggregate_name, topics in aggregate_specs.items():
        aggregate = AggregateChannel(member_topics=topics)
        aggregates[aggregate_name] = aggregate
        for topic in topics:
            channel_states.setdefault(topic, ChannelState())
            channels_to_aggregates.setdefault(topic, []).append(aggregate)

    for global_name, topic in GLOBAL_CHANNELS.items():
        aggregate = AggregateChannel(member_topics=(topic,))
        aggregates[f"{global_name}|Value"] = aggregate
        channel_states.setdefault(topic, ChannelState())
        channels_to_aggregates.setdefault(topic, []).append(aggregate)

    enabled = False
    autonomous = False
    enabled_seen = False
    autonomous_seen = False
    last_log_timestamp = 0
    first_relevant_timestamp: int | None = None

    def close_phase_window(window: PhaseWindow, timestamp: int, current_total: float) -> None:
        if window.active_since is None:
            return
        window.segments.append((window.active_since, timestamp, current_total))
        window.stats.add_interval(current_total, (timestamp - window.active_since) / 1_000_000.0)
        window.active_since = timestamp

    def apply_phase_transition(timestamp: int, old_enabled: bool, old_autonomous: bool) -> None:
        for aggregate in aggregates.values():
            for phase in PHASES:
                was_active = phase_active(phase, old_enabled, old_autonomous)
                is_active = phase_active(phase, enabled, autonomous)
                window = aggregate.phases[phase]
                if was_active and not is_active:
                    close_phase_window(window, timestamp, aggregate.current_total)
                    window.active_since = None
                elif not was_active and is_active:
                    window.active_since = timestamp

    for record in DataLogReader(str(log_path)):
        last_log_timestamp = record.getTimestamp()
        if record.isStart() or record.isFinish() or record.isSetMetadata():
            continue

        entry = record.getEntry()
        if entry not in entry_map:
            continue

        topic_name, data_type = entry_map[entry]
        timestamp = record.getTimestamp()
        if first_relevant_timestamp is None:
            first_relevant_timestamp = timestamp

        if topic_name == ENABLED_KEY:
            if data_type != "boolean":
                raise TypeError(f"Enabled topic {topic_name} had unexpected type {data_type}")
            old_enabled, old_autonomous = enabled, autonomous
            enabled = record.getBoolean()
            enabled_seen = True
            apply_phase_transition(timestamp, old_enabled, old_autonomous)
            continue

        if topic_name == AUTONOMOUS_KEY:
            if data_type != "boolean":
                raise TypeError(f"Autonomous topic {topic_name} had unexpected type {data_type}")
            old_enabled, old_autonomous = enabled, autonomous
            autonomous = record.getBoolean()
            autonomous_seen = True
            apply_phase_transition(timestamp, old_enabled, old_autonomous)
            continue

        if data_type == "double":
            value = record.getDouble()
        elif data_type == "float":
            value = record.getFloat()
        elif data_type in {"int", "int64"}:
            value = float(record.getInteger())
        else:
            continue

        if topic_name not in channel_states:
            continue
        state = channel_states[topic_name]
        previous_value = state.value
        state.value = value
        state.seen = True

        for aggregate in channels_to_aggregates.get(topic_name, []):
            aggregate.present_topics.add(topic_name)
            for phase in PHASES:
                if phase_active(phase, enabled, autonomous):
                    close_phase_window(aggregate.phases[phase], timestamp, aggregate.current_total)
            aggregate.current_total += value - previous_value
            for phase in PHASES:
                if phase_active(phase, enabled, autonomous):
                    aggregate.phases[phase].active_since = timestamp
                    aggregate.phases[phase].stats.note_update()

    for aggregate in aggregates.values():
        for phase in PHASES:
            if phase_active(phase, enabled, autonomous):
                close_phase_window(aggregate.phases[phase], last_log_timestamp, aggregate.current_total)

    bucket_columns = bucket_column_names(bucket_count)
    rows: list[dict[str, str]] = []
    for aggregate_name, aggregate in sorted(aggregates.items()):
        subsystem, metric = aggregate_name.split("|", 1)
        has_data = bool(aggregate.present_topics)
        for phase in PHASES:
            stats = aggregate.phases[phase].stats
            buckets = stats.bucket_averages(bucket_count) if has_data else [math.nan] * bucket_count
            row = {
                "phase": phase,
                "subsystem": subsystem,
                "metric": metric,
                "enabled_state_seen": str(enabled_seen).lower(),
                "autonomous_state_seen": str(autonomous_seen).lower(),
                "phase_seconds": format_number(stats.total_duration),
                "member_topics_expected": str(len(aggregate.member_topics)),
                "member_topics_present": str(len(aggregate.present_topics)),
                "update_count_in_phase": str(stats.update_count),
                "average": format_number(stats.average() if has_data else math.nan),
                "p95": format_number(stats.weighted_percentile(0.95) if has_data else math.nan),
                "max": format_number(stats.max_value if has_data and stats.total_duration > 0.0 else math.nan),
                "top_bucket_avg": format_number(buckets[-1]),
                "frac_above_25pct": format_number(
                    duty_cycle_from_buckets(buckets, 0.25 * stats.max_value) if has_data else math.nan
                ),
                "frac_above_50pct": format_number(
                    duty_cycle_from_buckets(buckets, 0.50 * stats.max_value) if has_data else math.nan
                ),
                "frac_above_75pct": format_number(
                    duty_cycle_from_buckets(buckets, 0.75 * stats.max_value) if has_data else math.nan
                ),
                "frac_above_100pct": format_number(
                    duty_cycle_from_buckets(buckets, 1.00 * stats.max_value) if has_data else math.nan
                ),
                "stator_supply_ratio": "",
                "mechanically_loaded": "",
                "bucket_count": str(bucket_count),
                "member_topics": "; ".join(aggregate.member_topics),
            }
            for column, value in zip(bucket_columns, buckets):
                row[column] = format_number(value)
            rows.append(row)
    pair_rows: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["metric"] not in CURRENT_SUFFIXES:
            continue
        pair_rows.setdefault((row["phase"], row["subsystem"]), {})[row["metric"]] = row

    for pair in pair_rows.values():
        supply_row = pair.get("Supply Current")
        stator_row = pair.get("Stator Current")
        if supply_row is None or stator_row is None:
            continue
        supply_avg = float(supply_row.get("average") or "nan")
        stator_avg = float(stator_row.get("average") or "nan")
        ratio = math.nan if (math.isnan(supply_avg) or supply_avg <= 1e-9) else (stator_avg / supply_avg)
        ratio_text = format_number(ratio)
        loaded_text = "true" if (not math.isnan(ratio) and ratio > 1.5) else "false"
        supply_row["stator_supply_ratio"] = ratio_text
        stator_row["stator_supply_ratio"] = ratio_text
        supply_row["mechanically_loaded"] = loaded_text
        stator_row["mechanically_loaded"] = loaded_text
    segments: dict[tuple[str, str, str], list[tuple[int, int, float]]] = {}
    for aggregate_name, aggregate in aggregates.items():
        subsystem, metric = aggregate_name.split("|", 1)
        for phase in PHASES:
            segments[(phase, subsystem, metric)] = aggregate.phases[phase].segments

    return AnalysisResult(
        rows=rows,
        segments=segments,
        log_start_timestamp=first_relevant_timestamp or 0,
    )


def write_csv(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_to_dataframe(rows: list[dict[str, str]], bucket_count: int) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    numeric_cols = [
        "phase_seconds",
        "member_topics_expected",
        "member_topics_present",
        "update_count_in_phase",
        "average",
        "p95",
        "max",
        "top_bucket_avg",
        "frac_above_25pct",
        "frac_above_50pct",
        "frac_above_75pct",
        "frac_above_100pct",
        "stator_supply_ratio",
        "bucket_count",
    ] + bucket_column_names(bucket_count)
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["coverage_ratio"] = df["member_topics_present"] / df["member_topics_expected"]
    return df


def barh_plot(df: pd.DataFrame, value_col: str, title: str, output_path: Path, limit: int = 12) -> None:
    plot_df = df.dropna(subset=[value_col]).sort_values(value_col, ascending=True).tail(limit)
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.barh(plot_df["subsystem"], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel("Amps")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bucket_heatmap(df: pd.DataFrame, bucket_count: int, title: str, output_path: Path) -> None:
    if df.empty:
        return
    bucket_cols = bucket_column_names(bucket_count)
    values = df[bucket_cols].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig_height = max(3.3, len(df) * 0.32)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), constrained_layout=True)
    image = ax.imshow(masked, aspect="auto")
    labels = [bucket_label(index, bucket_count).replace("_", "-") + "%" for index in range(bucket_count)]
    ax.set_xticks(range(bucket_count), labels)
    ax.set_yticks(range(len(df)), df["subsystem"].tolist())
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Amps")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def coverage_plot(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = (
        df[["subsystem", "metric", "member_topics_expected", "member_topics_present", "coverage_ratio"]]
        .drop_duplicates()
        .sort_values("coverage_ratio", ascending=True)
    )
    labels = (plot_df["subsystem"] + " | " + plot_df["metric"]).tolist()
    fig_height = max(4.0, len(plot_df) * 0.18)
    fig, ax = plt.subplots(figsize=(7.8, fig_height), constrained_layout=True)
    ax.barh(labels, plot_df["coverage_ratio"] * 100.0)
    ax.set_title("Topic Coverage")
    ax.set_xlabel("Coverage (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_step_series(
    segments: list[tuple[int, int, float]], origin_timestamp: int
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for start, end, value in segments:
        xs.extend([(start - origin_timestamp) / 1_000_000.0, (end - origin_timestamp) / 1_000_000.0, math.nan])
        ys.extend([value, value, math.nan])
    return xs, ys


def time_series_plot(
    df: pd.DataFrame,
    segments: dict[tuple[str, str, str], list[tuple[int, int, float]]],
    origin_timestamp: int,
    phase: str,
    metric: str,
    output_path: Path,
    limit: int = 5,
) -> None:
    plot_df = df[
        (df["phase"] == phase) & (df["metric"] == metric) & (df["subsystem"].isin(CORE_MECHANISMS))
    ].dropna(subset=["top_bucket_avg"]).sort_values("top_bucket_avg", ascending=False).head(limit)

    fig, ax = plt.subplots(figsize=(8.3, 4.6), constrained_layout=True)
    for _, row in plot_df.iterrows():
        key = (phase, row["subsystem"], metric)
        xs, ys = build_step_series(segments.get(key, []), origin_timestamp)
        if xs:
            ax.plot(xs, ys, linewidth=1.0, label=row["subsystem"])
    ax.set_title(f"{phase.title()} {metric} Top 5 Time Series")
    ax.set_xlabel("Log Time (s)")
    ax.set_ylabel("Amps")
    ax.grid(True, linestyle="--", alpha=0.35)
    if not plot_df.empty:
        ax.legend(fontsize=8, ncols=2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report_assets(
    result: AnalysisResult, report_dir: Path, bucket_count: int
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = rows_to_dataframe(result.rows, bucket_count)
    currents = df[df["metric"].isin(CURRENT_SUFFIXES)].copy()
    sections: list[dict[str, str]] = []

    for phase in PHASES:
        for metric in CURRENT_SUFFIXES:
            phase_metric = currents[
                (currents["phase"] == phase)
                & (currents["metric"] == metric)
                & (currents["subsystem"].isin(CORE_MECHANISMS))
            ].copy()

            average_filename = f"{phase}_{metric.lower().replace(' ', '_')}_average.png"
            top_filename = f"{phase}_{metric.lower().replace(' ', '_')}_top_bucket.png"
            heatmap_filename = f"{phase}_{metric.lower().replace(' ', '_')}_heatmap.png"
            series_filename = f"{phase}_{metric.lower().replace(' ', '_')}_timeseries.png"

            barh_plot(
                phase_metric,
                "average",
                f"{phase.title()} {metric} Average by Mechanism",
                report_dir / average_filename,
            )
            barh_plot(
                phase_metric,
                "top_bucket_avg",
                f"{phase.title()} {metric} Highest Bucket by Mechanism",
                report_dir / top_filename,
            )
            bucket_heatmap(
                phase_metric.sort_values("top_bucket_avg", ascending=False),
                bucket_count,
                f"{phase.title()} {metric} Buckets",
                report_dir / heatmap_filename,
            )
            time_series_plot(
                currents,
                result.segments,
                result.log_start_timestamp,
                phase,
                metric,
                report_dir / series_filename,
            )

            sections.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "average": average_filename,
                    "top_bucket": top_filename,
                    "heatmap": heatmap_filename,
                    "timeseries": series_filename,
                }
            )

    coverage_plot(df, report_dir / "coverage.png")
    return df, sections


def dataframe_table_html(
    df: pd.DataFrame,
    columns: list[str],
    row_phase_column: str | None = None,
    row_metric_column: str | None = None,
) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                text = "" if math.isnan(value) else f"{value:.3f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        attrs: list[str] = []
        if row_phase_column is not None and row_phase_column in df.columns:
            attrs.append(f"data-phase='{html.escape(str(row[row_phase_column]))}'")
        if row_metric_column is not None and row_metric_column in df.columns:
            attrs.append(f"data-metric='{html.escape(str(row[row_metric_column]))}'")
        attr_text = (" " + " ".join(attrs)) if attrs else ""
        body_rows.append(f"<tr{attr_text}>" + "".join(cells) + "</tr>")
    return f"<table border='1' cellspacing='0' cellpadding='4'><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html_report(
    df: pd.DataFrame, report_dir: Path, log_path: Path, csv_path: Path, sections: list[dict[str, str]], bucket_count: int
) -> Path:
    bucket_cols = bucket_column_names(bucket_count)
    top_rows = (
        df[
            (df["metric"].isin(CURRENT_SUFFIXES))
            & (df["subsystem"].isin(CORE_MECHANISMS))
        ]
        .sort_values("top_bucket_avg", ascending=False)
        .head(12)[["phase", "subsystem", "metric", "average", "top_bucket_avg", "max"]]
    )
    missing_rows = (
        df[df["member_topics_present"] < df["member_topics_expected"]][
            ["phase", "subsystem", "metric", "member_topics_present", "member_topics_expected"]
        ]
        .drop_duplicates()
    )
    supply_rows = df[df["metric"] == "Supply Current"].copy().sort_values(
        ["average", "phase", "subsystem"], ascending=[False, True, True]
    )
    stator_rows = df[df["metric"] == "Stator Current"].copy().sort_values(
        ["average", "phase", "subsystem"], ascending=[False, True, True]
    )
    other_rows = df[~df["metric"].isin(CURRENT_SUFFIXES)].copy().sort_values(
        ["average", "phase", "subsystem"], ascending=[False, True, True]
    )
    recommendations_path = report_dir / "limit_recommendations_supply.csv"
    recommendations_df = pd.read_csv(recommendations_path) if recommendations_path.exists() else pd.DataFrame()
    battery_figures = [
        ("Subsystem Currents vs Voltage", "battery_current_stacked_vs_voltage.png"),
        ("Brownout Deficit Stack", "brownout_deficit_stack.png"),
        ("Per-Subsystem Current Histograms", "subsystem_current_histograms.png"),
        ("Brownout Event Isolation", "brownout_event_isolation.png"),
        ("Cumulative Charge by Subsystem", "cumulative_charge_stacked.png"),
        ("Total Charge per Subsystem", "total_charge_per_subsystem.png"),
        ("Current-Voltage Correlation", "current_voltage_correlation.png"),
        ("Current Slack vs. I_max", "slack_analysis.png"),
        ("Coincidence Heatmap", "coincidence_heatmap.png"),
        ("Brownout Contribution Clusters", "brownout_clusters.png"),
        ("Sensitivity and Recommended Limits", "sensitivity_and_limits.png"),
    ]
    available_battery_figures = [(title, filename) for title, filename in battery_figures if (report_dir / filename).exists()]
    known_battery_files = {filename for _, filename in available_battery_figures}
    extra_pngs = sorted(
        [
            image_path.name
            for image_path in report_dir.glob("*.png")
            if image_path.name not in known_battery_files and image_path.name != "coverage.png"
        ]
    )

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>WPILOG Current Analysis Report</title>",
        """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; line-height: 1.3; }
.tab-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 20px; }
.tab-button { border: 1px solid #9ca3af; background: #f8fafc; border-radius: 8px; padding: 7px 12px; cursor: pointer; font-weight: 600; }
.tab-button.active { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
table { border-collapse: collapse; width: 100%; max-width: 100%; font-size: 13px; }
th, td { border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; }
th { background: #f3f4f6; }
.image-card { margin: 16px 0 22px; }
.image-card img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
</style>
""",
        "</head>",
        "<body>",
        "<h1>WPILOG Current Analysis Report</h1>",
        f"<p><b>Log:</b> {html.escape(log_path.name)}</p>",
        f"<p><b>CSV:</b> {html.escape(csv_path.name)}</p>",
        f"<p><b>Configured buckets:</b> {bucket_count}</p>",
        "<p>This report is intentionally plain. Use the selectors below to switch between enabled/auto/teleop and supply/stator views.</p>",
        "<p>",
        "<label for='phaseSelect'><b>Phase:</b></label>",
        "<select id='phaseSelect'>"
        + "".join(f"<option value='{phase}'>{phase}</option>" for phase in PHASES)
        + "</select>",
        "&nbsp;&nbsp;",
        "<label for='metricSelect'><b>Metric:</b></label>",
        "<select id='metricSelect'>"
        + "".join(f"<option value='{metric}'>{metric}</option>" for metric in CURRENT_SUFFIXES)
        + "</select>",
        "</p>",
        "<div class='tab-bar'>",
        "<button class='tab-button active' data-tab='phase-plots'>Phase Plots</button>",
        "<button class='tab-button' data-tab='limits'>Limit Recommendations</button>",
        "<button class='tab-button' data-tab='battery-plots'>Battery Plots</button>",
        "<button class='tab-button' data-tab='coverage'>Coverage</button>",
        "<button class='tab-button' data-tab='tables'>Tables</button>",
        "</div>",
        "<div id='phase-plots' class='tab-panel active'>",
    ]

    for section in sections:
        parts.extend(
            [
                f"<div data-phase='{html.escape(section['phase'])}' data-metric='{html.escape(section['metric'])}' class='toggle-section'>",
                f"<h2>{html.escape(section['phase'].title())} {html.escape(section['metric'])}</h2>",
                "<p>Average ranking.</p>",
                f"<p><img src='{html.escape(section['average'])}' alt='Average ranking' width='820'></p>",
                "<p>Highest bucket ranking.</p>",
                f"<p><img src='{html.escape(section['top_bucket'])}' alt='Top bucket ranking' width='820'></p>",
                "<p>Bucket heatmap.</p>",
                f"<p><img src='{html.escape(section['heatmap'])}' alt='Bucket heatmap' width='820'></p>",
                "<p>Top 5 mechanism currents over time.</p>",
                f"<p><img src='{html.escape(section['timeseries'])}' alt='Time series' width='820'></p>",
                "</div>",
            ]
        )

    parts.append("</div>")
    parts.append("<div id='limits' class='tab-panel'>")
    parts.append("<h2>Limit Recommendations</h2>")
    if recommendations_df.empty:
        parts.append("<p>No supply-limit recommendations generated yet.</p>")
    else:
        parts.append(
            dataframe_table_html(
                recommendations_df,
                ["subsystem", "current_configured", "p95_observed", "recommended", "delta", "utilization_pct"],
            )
        )
    parts.append("</div>")
    parts.append("<div id='battery-plots' class='tab-panel'>")
    parts.append("<h2>Battery Analysis Visualizations</h2>")
    if available_battery_figures:
        parts.append("<p>Additional battery-focused plots generated by <code>generate_battery_report(...)</code>.</p>")
        for title, filename in available_battery_figures:
            parts.extend(
                [
                    "<div class='image-card'>",
                    f"<h3>{html.escape(title)}</h3>",
                    f"<p><img src='{html.escape(filename)}' alt='{html.escape(title)}' width='980'></p>",
                    "</div>",
                ]
            )
    else:
        parts.append(
            "<p>No battery-specific visualizations were found in this report directory yet. "
            "Generate them with <code>generate_battery_report(...)</code> and place them in this same folder.</p>"
        )
    if extra_pngs:
        parts.append("<h3>Additional Generated PNGs</h3>")
        parts.append("<p>Auto-discovered images in the report folder not listed in the curated sections.</p>")
        for filename in extra_pngs:
            parts.extend(
                [
                    "<div class='image-card'>",
                    f"<h4>{html.escape(filename)}</h4>",
                    f"<p><img src='{html.escape(filename)}' alt='{html.escape(filename)}' width='980'></p>",
                    "</div>",
                ]
            )
    parts.append("</div>")

    parts.extend(
        [
            "<div id='coverage' class='tab-panel'>",
            "<h2>Coverage</h2>",
            "<p>Expected signals that were absent from the log.</p>",
            "<p><img src='coverage.png' alt='Topic coverage' width='820'></p>",
            "</div>",
            "<div id='tables' class='tab-panel'>",
            "<div class='table-section' data-table-kind='top'>",
            "<h2>Top Rows</h2>",
            "<p>Highest top-bucket rows among direct mechanisms. Filtered by the selectors above.</p>",
            dataframe_table_html(
                top_rows,
                ["phase", "subsystem", "metric", "average", "top_bucket_avg", "max"],
                row_phase_column="phase",
                row_metric_column="metric",
            ),
            "</div>",
            "<div class='table-section' data-table-kind='supply' data-metric='Supply Current'>",
            "<h2>Supply Rows</h2>",
            "<p>All supply-current CSV rows. Filtered by phase.</p>",
            dataframe_table_html(
                supply_rows,
                ["phase", "subsystem", "metric", "phase_seconds", "average", "p95", "max", "top_bucket_avg"]
                + bucket_cols,
                row_phase_column="phase",
                row_metric_column="metric",
            ),
            "</div>",
            "<div class='table-section' data-table-kind='stator' data-metric='Stator Current'>",
            "<h2>Stator Rows</h2>",
            "<p>All stator-current CSV rows. Filtered by phase.</p>",
            dataframe_table_html(
                stator_rows,
                ["phase", "subsystem", "metric", "phase_seconds", "average", "p95", "max", "top_bucket_avg"]
                + bucket_cols,
                row_phase_column="phase",
                row_metric_column="metric",
            ),
            "</div>",
            "<div class='table-section' data-table-kind='other'>",
            "<h2>Other Rows</h2>",
            "<p>Non-supply/stator CSV rows. Filtered by phase.</p>",
            dataframe_table_html(
                other_rows,
                ["phase", "subsystem", "metric", "phase_seconds", "average", "p95", "max", "top_bucket_avg"]
                + bucket_cols,
                row_phase_column="phase",
                row_metric_column="metric",
            ),
            "</div>",
            "<div class='table-section' data-table-kind='coverage'>",
            "<h2>Missing Coverage Rows</h2>",
            dataframe_table_html(
                missing_rows,
                ["phase", "subsystem", "metric", "member_topics_present", "member_topics_expected"],
                row_phase_column="phase",
                row_metric_column="metric",
            ),
            "</div>",
            "</div>",
            """
<script>
function switchTab(tabId) {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === tabId);
  });
}

function updateSections() {
  const phase = document.getElementById('phaseSelect').value;
  const metric = document.getElementById('metricSelect').value;
  document.querySelectorAll('.toggle-section').forEach((section) => {
    const matches = section.dataset.phase === phase && section.dataset.metric === metric;
    section.hidden = !matches;
  });
  document.querySelectorAll('.table-section').forEach((section) => {
    const sectionMetric = section.dataset.metric;
    if (sectionMetric) {
      section.hidden = sectionMetric !== metric;
    } else {
      section.hidden = false;
    }
  });
  document.querySelectorAll('tr[data-phase]').forEach((row) => {
    const rowPhase = row.dataset.phase;
    const rowMetric = row.dataset.metric;
    const phaseMatches = rowPhase === phase;
    const metricMatches = !rowMetric || rowMetric === metric;
    row.hidden = !(phaseMatches && metricMatches);
  });
}

function parseHashState() {
  const raw = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : '';
  const params = new URLSearchParams(raw);
  return {
    phase: params.get('phase'),
    metric: params.get('metric'),
    tab: params.get('tab'),
  };
}

function writeHashState() {
  const params = new URLSearchParams();
  params.set('phase', document.getElementById('phaseSelect').value);
  params.set('metric', document.getElementById('metricSelect').value);
  const activeButton = document.querySelector('.tab-button.active');
  if (activeButton) {
    params.set('tab', activeButton.dataset.tab);
  }
  window.location.hash = params.toString();
}

document.querySelectorAll('.tab-button').forEach((button) => {
  button.addEventListener('click', () => {
    switchTab(button.dataset.tab);
    writeHashState();
  });
});
document.getElementById('phaseSelect').addEventListener('change', () => {
  updateSections();
  writeHashState();
});
document.getElementById('metricSelect').addEventListener('change', () => {
  updateSections();
  writeHashState();
});
const hashState = parseHashState();
if (hashState.phase) document.getElementById('phaseSelect').value = hashState.phase;
if (hashState.metric) document.getElementById('metricSelect').value = hashState.metric;
switchTab(hashState.tab || 'phase-plots');
updateSections();
writeHashState();
</script>
""",
            "</body></html>",
        ]
    )

    html_path = report_dir / "index.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def generate_report(
    result: AnalysisResult,
    report_dir: Path,
    log_path: Path,
    csv_path: Path,
    bucket_count: int,
    configured_limits: Mapping[str, float] | None = None,
    threshold: float = 6.8,
) -> Path:
    df, sections = build_report_assets(result, report_dir, bucket_count)
    battery_inputs = _reconstruct_enabled_battery_inputs(result)
    if battery_inputs is not None:
        time, voltage, subsystem_currents = battery_inputs
        generate_battery_report(
            time,
            voltage,
            subsystem_currents,
            report_dir,
            configured_limits=configured_limits,
            threshold=threshold,
        )
    return write_html_report(df, report_dir, log_path, csv_path, sections, bucket_count)


def _normalize_battery_report_inputs(
    time: np.ndarray | list[float],
    voltage: np.ndarray | list[float],
    subsystem_currents: Mapping[str, np.ndarray | list[float]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if not subsystem_currents:
        raise ValueError("subsystem_currents must contain at least one subsystem")

    t = np.asarray(time, dtype=float).reshape(-1)
    v = np.asarray(voltage, dtype=float).reshape(-1)
    if t.ndim != 1 or v.ndim != 1:
        raise ValueError("time and voltage must be 1D arrays")
    if len(t) != len(v):
        raise ValueError("time and voltage must have the same length")
    if len(t) < 2:
        raise ValueError("time and voltage must contain at least two samples")

    subsystem_names = list(subsystem_currents.keys())
    current_rows: list[np.ndarray] = []
    for name in subsystem_names:
        current = np.asarray(subsystem_currents[name], dtype=float).reshape(-1)
        if len(current) != len(t):
            raise ValueError(f"subsystem {name!r} has length {len(current)} but expected {len(t)}")
        current_rows.append(current)

    return t, v, subsystem_names, np.vstack(current_rows)


def _battery_palette(subsystems: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {name: cmap(index % cmap.N) for index, name in enumerate(subsystems)}


def _battery_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _detect_brownout_events(
    time: np.ndarray,
    voltage: np.ndarray,
    threshold: float = 6.8,
    merge_gap_s: float = 0.5,
    currents: np.ndarray | None = None,
    subsystem_names: list[str] | None = None,
    i_total: np.ndarray | None = None,
    i_max: float | None = None,
) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    below = voltage < threshold
    if not np.any(below):
        return [], pd.DataFrame()

    transitions = np.diff(below.astype(int))
    starts = list(np.where(transitions == 1)[0] + 1)
    ends = list(np.where(transitions == -1)[0] + 1)
    if below[0]:
        starts = [0] + starts
    if below[-1]:
        ends = ends + [len(below)]

    merged: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if time[start] - time[last_end - 1] < merge_gap_s:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    if currents is None or subsystem_names is None or i_total is None or i_max is None:
        return merged, pd.DataFrame()

    contribution_rows: list[dict[str, float | int | str]] = []
    for event_index, (start_idx, end_idx) in enumerate(merged, start=1):
        if end_idx <= start_idx:
            continue
        peak_local = int(np.argmax(i_total[start_idx:end_idx]))
        peak_idx = start_idx + peak_local
        total_at_peak = float(i_total[peak_idx])
        deficit = max(0.0, total_at_peak - i_max)
        ranked = sorted(
            [
                (subsystem, float(currents[sub_idx, peak_idx]))
                for sub_idx, subsystem in enumerate(subsystem_names)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (subsystem, current_val) in enumerate(ranked, start=1):
            contribution_rows.append(
                {
                    "event_index": event_index,
                    "event_start_s": float(time[start_idx]),
                    "event_end_s": float(time[end_idx - 1]),
                    "peak_time_s": float(time[peak_idx]),
                    "min_voltage_v": float(np.min(voltage[start_idx:end_idx])),
                    "subsystem": subsystem,
                    "peak_current_a": current_val,
                    "fraction_of_total": (current_val / total_at_peak) if total_at_peak > 1e-9 else 0.0,
                    "fraction_of_deficit": (current_val / deficit) if deficit > 1e-9 else math.nan,
                    "rank": rank,
                }
            )
    return merged, pd.DataFrame(contribution_rows)


def _segment_value_at_t(segments: list[tuple[int, int, float]], timestamp: int, cursor: int) -> tuple[float, int]:
    while cursor < len(segments) and timestamp >= segments[cursor][1]:
        cursor += 1
    if cursor < len(segments):
        start, end, value = segments[cursor]
        if start <= timestamp < end:
            return value, cursor
    return 0.0, cursor


def _reconstruct_enabled_battery_inputs(result: AnalysisResult) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    phase = "enabled"
    voltage_segments = result.segments.get((phase, "Battery Voltage", "Value"), [])
    if not voltage_segments:
        return None

    current_segments_by_subsystem = {
        subsystem: result.segments.get((phase, subsystem, "Supply Current"), [])
        for subsystem in CORE_MECHANISMS
    }
    current_segments_by_subsystem = {
        subsystem: segments for subsystem, segments in current_segments_by_subsystem.items() if segments
    }
    if not current_segments_by_subsystem:
        return None

    boundaries: set[int] = set()
    for start, end, _ in voltage_segments:
        boundaries.add(start)
        boundaries.add(end)
    for segments in current_segments_by_subsystem.values():
        for start, end, _ in segments:
            boundaries.add(start)
            boundaries.add(end)

    timestamps = sorted(boundaries)
    if len(timestamps) < 2:
        return None

    time_seconds = np.array([(ts - result.log_start_timestamp) / 1_000_000.0 for ts in timestamps], dtype=float)
    if np.any(np.diff(time_seconds) <= 0.0):
        return None

    voltage_values = np.zeros(len(timestamps), dtype=float)
    voltage_cursor = 0
    for idx, ts in enumerate(timestamps):
        voltage_values[idx], voltage_cursor = _segment_value_at_t(voltage_segments, ts, voltage_cursor)

    subsystem_currents: dict[str, np.ndarray] = {}
    for subsystem, segments in current_segments_by_subsystem.items():
        values = np.zeros(len(timestamps), dtype=float)
        cursor = 0
        for idx, ts in enumerate(timestamps):
            values[idx], cursor = _segment_value_at_t(segments, ts, cursor)
        subsystem_currents[subsystem] = values

    return time_seconds, voltage_values, subsystem_currents


def generate_battery_report(
    time: np.ndarray | list[float],
    voltage: np.ndarray | list[float],
    subsystem_currents: Mapping[str, np.ndarray | list[float]],
    output_dir: Path | str,
    configured_limits: Mapping[str, float] | None = None,
    threshold: float = 6.8,
    safety_margin: float = 0.05,
) -> dict[str, Path]:
    """
    Generate publication-quality battery/current analysis visualizations.

    Returns a mapping from analysis key to output PNG path.
    """
    _battery_plot_style()
    t, v, subsystems, currents = _normalize_battery_report_inputs(time, voltage, subsystem_currents)
    colors = _battery_palette(subsystems)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # 1) Stacked area of subsystem currents with voltage overlay.
    fig, ax = plt.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    stack_colors = [colors[name] for name in subsystems]
    ax.stackplot(t, currents, labels=subsystems, colors=stack_colors, alpha=0.9)
    ax.set_title("Subsystem Currents and Battery Voltage vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(t, v, color="black", linewidth=1.5, label="Battery Voltage")
    ax2.axhline(
        threshold, color="red", linestyle="--", linewidth=1.2, label=f"Brownout Threshold ({threshold:.2f} V)"
    )
    ax2.set_ylabel("Voltage (V)")

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", ncol=2, framealpha=0.9)
    file_path = output_path / "battery_current_stacked_vs_voltage.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["stacked_current_vs_voltage"] = file_path

    # 2) Per-subsystem current histograms sorted by p95 descending.
    subsystem_stats = [
        (name, np.median(currents[idx]), np.percentile(currents[idx], 95), currents[idx])
        for idx, name in enumerate(subsystems)
    ]
    subsystem_stats.sort(key=lambda item: item[2], reverse=True)
    panel_count = len(subsystem_stats)
    cols = max(1, int(math.ceil(math.sqrt(panel_count))))
    rows = int(math.ceil(panel_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for axis_index, ax in enumerate(axes_arr.flat):
        if axis_index >= panel_count:
            ax.set_visible(False)
            continue
        name, median_val, p95_val, series = subsystem_stats[axis_index]
        ax.hist(series, bins=40, color=colors[name], alpha=0.75, edgecolor="white")
        ax.axvline(median_val, color="black", linestyle="-", linewidth=1.2, label=f"Median: {median_val:.1f} A")
        ax.axvline(p95_val, color="darkred", linestyle="--", linewidth=1.2, label=f"P95: {p95_val:.1f} A")
        ax.set_title(name)
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Samples")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")
    fig.suptitle("Per-Subsystem Current Distributions", fontsize=13)
    file_path = output_path / "subsystem_current_histograms.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["subsystem_histograms"] = file_path

    # 3) Brownout event isolation windows with merged events.
    events, _ = _detect_brownout_events(t, v, threshold=threshold, merge_gap_s=0.5)
    if events:
        event_scored = [(start, end, float(np.min(v[start:end]))) for start, end in events]
        event_scored.sort(key=lambda item: item[2])
        selected = event_scored[:6]
        selected.sort(key=lambda item: item[0])

        fig, axes = plt.subplots(len(selected), 1, figsize=(11.0, 3.0 * len(selected)), constrained_layout=True)
        axes_arr = np.atleast_1d(axes)
        for ax, (start, end, min_v) in zip(axes_arr, selected):
            window_start_t = max(t[0], t[start] - 5.0)
            window_end_t = min(t[-1], t[end - 1] + 1.0)
            mask = (t >= window_start_t) & (t <= window_end_t)
            local_t = t[mask]
            local_currents = currents[:, mask]
            local_voltage = v[mask]

            ax.stackplot(local_t, local_currents, colors=stack_colors, alpha=0.88)
            ax.set_ylabel("Current (A)")
            ax.grid(True, linestyle="--", alpha=0.25)

            ax2 = ax.twinx()
            ax2.plot(local_t, local_voltage, color="black", linewidth=1.3)
            ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
            ax2.set_ylabel("Voltage (V)")
            ax.set_title(f"Brownout @ t={t[start]:.2f}s (min V={min_v:.2f} V)")

        axes_arr[-1].set_xlabel("Time (s)")
    else:
        fig, ax = plt.subplots(figsize=(10.0, 3.5), constrained_layout=True)
        ax.plot(t, v, color="black", linewidth=1.5)
        ax.axhline(
            threshold, color="red", linestyle="--", linewidth=1.2, label=f"Brownout Threshold ({threshold:.2f} V)"
        )
        ax.set_title("Brownout Event Isolation (No Brownout Events Detected)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    file_path = output_path / "brownout_event_isolation.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["brownout_events"] = file_path

    # 4a) Cumulative charge (integral of current over time) stacked area.
    dt = np.diff(t)
    cumulative_charges = np.zeros_like(currents)
    for idx in range(currents.shape[0]):
        increments = 0.5 * (currents[idx, 1:] + currents[idx, :-1]) * dt
        cumulative_charges[idx, 1:] = np.cumsum(increments)

    fig, ax = plt.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    ax.stackplot(t, cumulative_charges, labels=subsystems, colors=stack_colors, alpha=0.9)
    ax.set_title("Cumulative Charge Drawn by Subsystem")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Charge (C)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    file_path = output_path / "cumulative_charge_stacked.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["cumulative_charge"] = file_path

    # 4b) Total Coulombs per subsystem horizontal bars.
    if hasattr(np, "trapezoid"):
        total_charges = np.array([float(np.trapezoid(currents[idx], t)) for idx in range(currents.shape[0])], dtype=float)
    else:
        total_charges = np.array([float(np.sum(0.5 * (currents[idx, 1:] + currents[idx, :-1]) * dt)) for idx in range(currents.shape[0])], dtype=float)
    order = np.argsort(total_charges)[::-1]
    ordered_names = [subsystems[idx] for idx in order]
    ordered_charges = total_charges[order]
    ordered_colors = [colors[name] for name in ordered_names]
    fig_height = max(3.8, 0.45 * len(ordered_names) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height), constrained_layout=True)
    bars = ax.barh(ordered_names, ordered_charges, color=ordered_colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title("Total Charge Drawn by Subsystem")
    ax.set_xlabel("Charge (C)")
    ax.set_ylabel("Subsystem")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    max_charge = float(np.max(ordered_charges)) if len(ordered_charges) else 0.0
    for bar, value in zip(bars, ordered_charges):
        ax.text(
            value + max_charge * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f} C",
            va="center",
            ha="left",
            fontsize=9,
        )
    file_path = output_path / "total_charge_per_subsystem.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["total_charge_by_subsystem"] = file_path

    # 5) Correlation between subsystem current and battery voltage.
    correlations: list[tuple[str, float]] = []
    for idx, name in enumerate(subsystems):
        current_series = currents[idx]
        if np.std(current_series) == 0.0 or np.std(v) == 0.0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(current_series, v)[0, 1])
        correlations.append((name, corr))
    correlations.sort(key=lambda item: item[1])

    corr_names = [item[0] for item in correlations]
    corr_values = np.array([item[1] for item in correlations], dtype=float)
    cmap = plt.get_cmap("coolwarm")
    color_weights = np.clip(np.abs(corr_values), 0.0, 1.0)
    corr_colors = [cmap(weight) for weight in color_weights]
    fig_height = max(3.8, 0.45 * len(corr_names) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height), constrained_layout=True)
    bars = ax.barh(corr_names, corr_values, color=corr_colors, alpha=0.92)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_title("Correlation of Subsystem Current with Battery Voltage")
    ax.set_xlabel("Pearson Correlation (r)")
    ax.set_ylabel("Subsystem")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, corr_values):
        offset = 0.02 if value >= 0 else -0.02
        align = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"r={value:.3f}",
            va="center",
            ha=align,
            fontsize=9,
        )
    file_path = output_path / "current_voltage_correlation.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["current_voltage_correlation"] = file_path

    # 6) Battery parameter estimation reused by the analyses below.
    i_total = np.sum(currents, axis=0)
    v_oc = float(np.percentile(v, 99))
    v_drop = np.clip(v_oc - v, 0.0, None)
    i_energy = float(np.dot(i_total, i_total))
    if i_energy <= 1e-9:
        r_int = 0.02
    else:
        r_int = float(np.dot(i_total, v_drop) / i_energy)
    r_int = float(np.clip(r_int, 0.005, 0.1))
    i_max = float((v_oc - threshold) / r_int) if r_int > 0.0 else 0.0

    # 7) Slack analysis timeseries and distribution.
    slack = i_max - i_total
    brownout_rate = float(np.mean(slack < 0.0)) if slack.size else 0.0
    mean_slack = float(np.mean(slack)) if slack.size else 0.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.0),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    ax_top.plot(t, slack, color="#1f2937", linewidth=1.4)
    ax_top.fill_between(t, slack, 0.0, where=slack >= 0.0, color="#22c55e", alpha=0.35, interpolate=True)
    ax_top.fill_between(t, slack, 0.0, where=slack < 0.0, color="#ef4444", alpha=0.35, interpolate=True)
    ax_top.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax_top.set_ylabel("Current Headroom (A)")
    ax_top.grid(True, linestyle="--", alpha=0.3)

    ax_bottom.hist(slack, bins=50, color="#64748b", alpha=0.82, edgecolor="white")
    ax_bottom.set_ylabel("Density / Count")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.grid(True, linestyle="--", alpha=0.25)
    ax_bottom.text(
        0.98,
        0.95,
        f"Mean slack: {mean_slack:.1f} A\nBrownout rate: {brownout_rate * 100.0:.1f}%",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
    )
    fig.suptitle("Current Slack vs. I_max")
    file_path = output_path / "slack_analysis.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["slack_analysis"] = file_path

    # 8) Brownout peak-frame deficit attribution.
    events, deficit_contributions = _detect_brownout_events(
        t,
        v,
        threshold=threshold,
        merge_gap_s=0.5,
        currents=currents,
        subsystem_names=subsystems,
        i_total=i_total,
        i_max=i_max,
    )
    if not deficit_contributions.empty:
        pivot = (
            deficit_contributions.pivot(index="event_index", columns="subsystem", values="fraction_of_total")
            .fillna(0.0)
            .sort_index()
        )
        ordered_cols = sorted(pivot.columns.tolist(), key=lambda name: float(pivot[name].mean()), reverse=True)
        pivot = pivot[ordered_cols]
        y_labels = [f"Event {int(event_id)}" for event_id in pivot.index.to_numpy(dtype=int)]
        fig_height = max(3.8, 0.45 * len(y_labels) + 1.8)
        fig, ax = plt.subplots(figsize=(11.5, fig_height), constrained_layout=True)
        left = np.zeros(len(pivot), dtype=float)
        for subsystem in ordered_cols:
            vals = pivot[subsystem].to_numpy(dtype=float)
            ax.barh(y_labels, vals, left=left, color=colors[subsystem], alpha=0.92, label=subsystem)
            left += vals
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Fractional Contribution at Peak Frame")
        ax.set_title("Brownout Deficit Stack by Event")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.9)
        file_path = output_path / "brownout_deficit_stack.png"
        fig.savefig(file_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved["brownout_deficit_stack"] = file_path

        deficit_csv = output_path / "brownout_deficit_contributions.csv"
        deficit_contributions.sort_values(["event_index", "rank"]).to_csv(deficit_csv, index=False)
        saved["brownout_deficit_contributions_csv"] = deficit_csv

    # 9) Brownout contribution vectors and clustering.
    contribution_vectors: list[np.ndarray] = []
    for start_idx, end_idx in events:
        if end_idx <= start_idx:
            continue
        peak_local = int(np.argmax(i_total[start_idx:end_idx]))
        peak_idx = start_idx + peak_local
        contribution_vectors.append(currents[:, peak_idx].astype(float, copy=True))

    event_count = len(contribution_vectors)
    if event_count < 2:
        fig, ax = plt.subplots(figsize=(10.0, 3.8), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Insufficient brownout events for clustering (found {event_count}).",
            ha="center",
            va="center",
            fontsize=13,
        )
    else:
        vectors = np.vstack(contribution_vectors)
        fallback_mode = False
        k = min(3, event_count)
        try:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(vectors)
            centroids = model.cluster_centers_
        except ImportError:
            fallback_mode = True
            k = 1
            labels = np.zeros(event_count, dtype=int)
            centroids = np.mean(vectors, axis=0, keepdims=True)

        cluster_sizes = np.array([int(np.sum(labels == idx)) for idx in range(k)], dtype=int)
        centroid_sums = np.sum(centroids, axis=1)
        order = np.argsort(centroid_sums)[::-1]
        ordered_centroids = centroids[order]
        ordered_sizes = cluster_sizes[order]
        ordered_sums = centroid_sums[order]

        fig_height = max(4.2, 1.1 * k + 2.6)
        fig, ax = plt.subplots(figsize=(12.0, fig_height), constrained_layout=True)
        y_base = np.arange(k, dtype=float)
        bar_group_height = 0.78
        bar_h = bar_group_height / max(len(subsystems), 1)
        offsets = (np.arange(len(subsystems), dtype=float) - (len(subsystems) - 1) / 2.0) * bar_h

        for subsystem_idx, subsystem in enumerate(subsystems):
            y = y_base + offsets[subsystem_idx]
            ax.barh(
                y,
                ordered_centroids[:, subsystem_idx],
                height=bar_h * 0.95,
                color=colors[subsystem],
                alpha=0.92,
                label=subsystem if subsystem_idx == 0 else None,
            )

        ax.set_yticks(y_base)
        ax.set_yticklabels([f"Cluster {idx + 1}" for idx in range(k)])
        ax.set_xlabel("Current at Peak Frame (A)")
        if fallback_mode:
            ax.set_title("Brownout Contribution Clusters (KMeans unavailable, fallback mean cluster)")
        else:
            ax.set_title(f"Brownout Contribution Clusters (KMeans, k={k})")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        x_max = float(np.max(ordered_centroids)) if ordered_centroids.size else 0.0
        x_offset = max(0.8, x_max * 0.02)
        for idx in range(k):
            row_max = float(np.max(ordered_centroids[idx]))
            ax.text(
                row_max + x_offset,
                y_base[idx],
                f"n={ordered_sizes[idx]}, total={ordered_sums[idx]:.1f} A",
                va="center",
                ha="left",
                fontsize=9,
            )

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[subsystem], alpha=0.92) for subsystem in subsystems
        ]
        ax.legend(legend_handles, subsystems, loc="best", fontsize=8, framealpha=0.9, ncol=2)

    file_path = output_path / "brownout_clusters.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["brownout_clusters"] = file_path

    # 10) Coincidence matrix (fraction of time both subsystems exceed own P75).
    total_duration = float(np.sum(dt))
    p75 = np.array([float(np.percentile(currents[idx], 75)) for idx in range(len(subsystems))], dtype=float)
    coincidence = np.zeros((len(subsystems), len(subsystems)), dtype=float)
    if total_duration > 0.0:
        above = currents[:, :-1] >= p75[:, None]
        for i in range(len(subsystems)):
            for j in range(len(subsystems)):
                mask = above[i] & above[j]
                coincidence[i, j] = float(np.sum(dt[mask]) / total_duration)
    coincidence_df = pd.DataFrame(coincidence, index=subsystems, columns=subsystems)
    coincidence_csv = output_path / "coincidence_matrix.csv"
    coincidence_df.to_csv(coincidence_csv)
    saved["coincidence_matrix_csv"] = coincidence_csv

    fig_size = max(5.8, 0.42 * len(subsystems) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    image = ax.imshow(coincidence, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(subsystems)), subsystems, rotation=45, ha="right")
    ax.set_yticks(range(len(subsystems)), subsystems)
    ax.set_title("Coincidence Matrix: Time Both > P75")
    fig.colorbar(image, ax=ax, label="Coincidence Fraction")
    file_path = output_path / "coincidence_heatmap.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["coincidence_heatmap"] = file_path

    # 11) Sensitivity analysis and recommended limit reductions.
    p95 = np.array([float(np.percentile(currents[idx], 95)) for idx in range(len(subsystems))], dtype=float)
    low_slack_mask = slack < (0.1 * i_max)
    low_slack_count = int(np.sum(low_slack_mask))
    sensitivities = np.zeros(len(subsystems), dtype=float)
    if low_slack_count > 0:
        for idx in range(len(subsystems)):
            near_p95 = currents[idx] > (0.9 * p95[idx])
            sensitivities[idx] = float(np.mean(near_p95[low_slack_mask]))

    configured = {name: float(value) for name, value in (configured_limits or {}).items()}
    base_limits = np.array([configured.get(subsystem, p95[idx]) for idx, subsystem in enumerate(subsystems)], dtype=float)
    recommended_limits = base_limits.copy()
    reduction_order = list(np.argsort(sensitivities)[::-1])
    target_total = i_max * max(0.0, 1.0 - safety_margin)
    if float(np.sum(recommended_limits)) > target_total:
        for idx in reduction_order:
            other_sum = float(np.sum(recommended_limits)) - float(recommended_limits[idx])
            if other_sum + float(recommended_limits[idx]) <= target_total:
                continue
            if other_sum + 5.0 > target_total:
                recommended_limits[idx] = 5.0
                continue
            low, high = 5.0, float(recommended_limits[idx])
            while high - low > 0.01:
                mid = 0.5 * (low + high)
                if other_sum + mid <= target_total:
                    low = mid
                else:
                    high = mid
            recommended_limits[idx] = max(5.0, low)
            if float(np.sum(recommended_limits)) <= target_total:
                break

    sensitivity_order = np.argsort(sensitivities)[::-1]
    ordered_subsystems = [subsystems[idx] for idx in sensitivity_order]
    ordered_sensitivities = sensitivities[sensitivity_order]
    ordered_p95 = p95[sensitivity_order]
    ordered_configured = base_limits[sensitivity_order]
    ordered_limits = recommended_limits[sensitivity_order]

    fig_height = max(4.6, 0.42 * len(subsystems) + 2.2)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, fig_height), constrained_layout=True)

    y_pos = np.arange(len(ordered_subsystems))
    left_colors = [colors[name] for name in ordered_subsystems]
    ax_left.barh(y_pos, ordered_sensitivities, color=left_colors, alpha=0.9)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(ordered_subsystems)
    ax_left.invert_yaxis()
    ax_left.set_xlim(0.0, 1.0)
    ax_left.set_xlabel("Sensitivity Score")
    ax_left.set_title("Low-Slack Sensitivity Rank")
    ax_left.grid(axis="x", linestyle="--", alpha=0.3)

    bar_h = 0.38
    ax_right.barh(y_pos - bar_h / 2.0, ordered_configured, height=bar_h, color="#94a3b8", alpha=0.92, label="Configured/P95")
    ax_right.barh(
        y_pos + bar_h / 2.0,
        ordered_limits,
        height=bar_h,
        color="#0ea5e9",
        alpha=0.92,
        label="Recommended",
    )
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(ordered_subsystems)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Current Limit (A)")
    ax_right.set_title("Configured/P95 vs. Recommended Supply Limit")
    ax_right.grid(axis="x", linestyle="--", alpha=0.3)
    ax_right.legend(loc="lower right")
    limit_max = max(
        float(np.max(ordered_configured)) if ordered_configured.size else 0.0,
        float(np.max(ordered_limits)) if ordered_limits.size else 0.0,
    )
    text_dx = max(0.35, limit_max * 0.01)
    for pos, cfg_val, limit_val in zip(y_pos, ordered_configured, ordered_limits):
        if limit_val < cfg_val:
            delta = cfg_val - limit_val
            ax_right.text(limit_val + text_dx, pos + bar_h / 2.0, f"-{delta:.1f} A", va="center", ha="left", fontsize=9)

    fig.suptitle("Subsystem Sensitivity and Recommended Current Limits")
    file_path = output_path / "sensitivity_and_limits.png"
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["sensitivity_and_limits"] = file_path

    recommendation_rows: list[dict[str, float | str]] = []
    for idx, subsystem in enumerate(subsystems):
        configured_limit = configured.get(subsystem, p95[idx])
        utilization_pct = (100.0 * p95[idx] / configured_limit) if configured_limit > 1e-9 else math.nan
        recommendation_rows.append(
            {
                "subsystem": subsystem,
                "limit_type": "supply",
                "current_configured": configured_limit,
                "p95_observed": p95[idx],
                "recommended": recommended_limits[idx],
                "delta": recommended_limits[idx] - configured_limit,
                "utilization_pct": utilization_pct,
                "headroom_A": configured_limit - p95[idx],
                "sensitivity": sensitivities[idx],
            }
        )
    recommendation_df = pd.DataFrame(recommendation_rows).sort_values("sensitivity", ascending=False)
    recommendation_csv = output_path / "limit_recommendations_supply.csv"
    recommendation_df.to_csv(recommendation_csv, index=False)
    saved["limit_recommendations_supply_csv"] = recommendation_csv

    # Plain-text report for quick CLI inspection.
    print(f"Battery fit: V_oc={v_oc:.3f} V, R_int={r_int:.4f} ohm, I_max={i_max:.2f} A")
    print(f"Brownout rate: {brownout_rate:.4f}")
    name_w = max(len("Subsystem"), max((len(name) for name in subsystems), default=0))
    p95_w = len("P95 (A)")
    sens_w = len("Sensitivity")
    rec_w = len("Recommended (A)")
    delta_w = len("Delta (A)")
    header = (
        f"{'Subsystem':<{name_w}} | {'P95 (A)':>{p95_w}} | {'Sensitivity':>{sens_w}} | "
        f"{'Recommended (A)':>{rec_w}} | {'Delta (A)':>{delta_w}}"
    )
    print(header)
    print("-" * len(header))
    for idx in sensitivity_order:
        delta_val = p95[idx] - recommended_limits[idx]
        print(
            f"{subsystems[idx]:<{name_w}} | {p95[idx]:>{p95_w}.2f} | {sensitivities[idx]:>{sens_w}.3f} | "
            f"{recommended_limits[idx]:>{rec_w}.2f} | {delta_val:>{delta_w}.2f}"
        )

    return saved


def parse_configured_limits(raw_limits: str | None) -> dict[str, float]:
    if not raw_limits:
        return {}
    raw_limits = raw_limits.strip()
    path_candidate = Path(raw_limits)
    if path_candidate.exists():
        df = pd.read_csv(path_candidate)
        if "subsystem" in df.columns and "limit" in df.columns:
            return {str(row["subsystem"]): float(row["limit"]) for _, row in df.iterrows()}
        if len(df.columns) >= 2:
            subsystem_col, limit_col = df.columns[0], df.columns[1]
            return {str(row[subsystem_col]): float(row[limit_col]) for _, row in df.iterrows()}
        raise ValueError("limits CSV must contain at least two columns (subsystem, limit)")

    limits: dict[str, float] = {}
    for part in raw_limits.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid --limits token {piece!r}; expected KEY=VALUE")
        key, value_text = piece.split("=", 1)
        limits[key.strip()] = float(value_text.strip())
    return limits


def print_limit_recommendations(report_dir: Path) -> None:
    path = report_dir / "limit_recommendations_supply.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("sensitivity", ascending=False)
    if df.empty:
        return
    print("\nSupply limit recommendations:")
    print(
        df[["subsystem", "current_configured", "p95_observed", "recommended", "delta", "utilization_pct"]].to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize enabled-period subsystem current draw from an AdvantageKit/WPILOG file, "
            "split by enabled/auto/teleop, and optionally generate chart images plus a simple HTML report."
        )
    )
    parser.add_argument("log_path", type=Path, help="Path to the input .wpilog file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output CSV. Defaults to <log_name>_current_summary.csv",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory for PNG charts and the HTML report. Defaults to <csv_stem>_report",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating PNG/HTML visualizations and write only the CSV.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=4,
        help="How many current buckets to split enabled/auto/teleop time into. Default: 4.",
    )
    parser.add_argument(
        "--limits",
        type=str,
        help="Configured subsystem limits as CSV path or comma-separated KEY=VALUE pairs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.8,
        help="Brownout voltage threshold in volts. Default: 6.8.",
    )
    args = parser.parse_args()
    if args.quantiles < 2:
        parser.error("--quantiles must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    log_path = args.log_path.resolve()
    default_output_root = Path(__file__).resolve().parent / "generated"
    default_base_dir = default_output_root / log_path.stem
    output_path = (
        args.output.resolve()
        if args.output
        else default_base_dir / f"{log_path.stem}_current_summary.csv"
    )
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir
        else default_base_dir / "report"
    )
    configured_limits = parse_configured_limits(args.limits)

    result = analyze_log(log_path, args.quantiles)
    write_csv(result.rows, output_path)
    print(output_path)

    if not args.no_report:
        html_path = generate_report(
            result,
            report_dir,
            log_path,
            output_path,
            args.quantiles,
            configured_limits=configured_limits,
            threshold=args.threshold,
        )
        print(html_path)
        print_limit_recommendations(report_dir)


if __name__ == "__main__":
    main()
