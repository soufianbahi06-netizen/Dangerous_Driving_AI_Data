# -*- coding: utf-8 -*-
"""
DANGEROUS DRIVING AI - PHASE 10.1
Trip-level and vehicle-level feature engineering.

Inputs (must be in the same folder as this script):
    Phase9_2_Event_Audit.csv
    Phase9_2_Behavior_Episodes.csv

Outputs:
    Phase10_1_Trip_Features.csv
    Phase10_1_Vehicle_Features.csv
    Phase10_1_Report.txt

This script does not modify any Phase 9 or Phase 9.2 source files.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

EVENT_FILE = BASE_DIR / "Phase9_2_Event_Audit.csv"
EPISODE_FILE = BASE_DIR / "Phase9_2_Behavior_Episodes.csv"

TRIP_OUTPUT = BASE_DIR / "Phase10_1_Trip_Features.csv"
VEHICLE_OUTPUT = BASE_DIR / "Phase10_1_Vehicle_Features.csv"
REPORT_OUTPUT = BASE_DIR / "Phase10_1_Report.txt"
BACKUP_DIR = BASE_DIR / "Phase10_1_Backups"

EVENT_TYPES = [
    "HARD_ACCELERATION",
    "MODERATE_ACCELERATION",
    "HARD_BRAKING",
    "MODERATE_BRAKING",
]

ACCELERATION_TYPES = {"HARD_ACCELERATION", "MODERATE_ACCELERATION"}
BRAKING_TYPES = {"HARD_BRAKING", "MODERATE_BRAKING"}
HARD_TYPES = {"HARD_ACCELERATION", "HARD_BRAKING"}


def stop(message: str) -> None:
    raise RuntimeError(message)


def backup_existing_outputs() -> list[str]:
    existing = [
        path for path in (TRIP_OUTPUT, VEHICLE_OUTPUT, REPORT_OUTPUT)
        if path.exists()
    ]
    if not existing:
        return []

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed_up = []

    for source in existing:
        destination = BACKUP_DIR / f"{source.stem}_{stamp}{source.suffix}"
        counter = 1
        while destination.exists():
            destination = BACKUP_DIR / f"{source.stem}_{stamp}_{counter}{source.suffix}"
            counter += 1
        source.replace(destination)
        backed_up.append(destination.name)

    return backed_up


def require_columns(frame: pd.DataFrame, columns: list[str], file_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        stop(
            f"{file_name} is missing required columns: {', '.join(missing)}\n"
            "Please check that you are using the Phase 9.2 output files."
        )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EVENT_FILE.exists():
        stop(f"Input file not found: {EVENT_FILE.name}")
    if not EPISODE_FILE.exists():
        stop(f"Input file not found: {EPISODE_FILE.name}")

    events = pd.read_csv(
        EVENT_FILE,
        dtype={"Vehicle_ID": "string", "Trip_ID": "string", "Segment_ID": "string"},
        low_memory=False,
        keep_default_na=False,
    )
    episodes = pd.read_csv(
        EPISODE_FILE,
        dtype={"Vehicle_ID": "string", "Trip_ID": "string", "Segment_ID": "string"},
        low_memory=False,
        keep_default_na=False,
    )

    require_columns(
        events,
        [
            "Event_ID", "Vehicle_ID", "Trip_ID", "Segment_ID",
            "Start_Time_ms", "End_Time_ms", "Duration_seconds",
            "Canonical_Type", "Audit_Status",
        ],
        EVENT_FILE.name,
    )
    require_columns(
        episodes,
        [
            "Vehicle_ID", "Trip_ID", "Segment_ID",
            "Episode_Start_ms", "Episode_End_ms", "Episode_Duration_seconds",
            "Events_Count", "Hard_Acceleration_Count",
            "Moderate_Acceleration_Count", "Hard_Braking_Count",
            "Moderate_Braking_Count", "Behavior_Pattern",
        ],
        EPISODE_FILE.name,
    )

    for column in ("Vehicle_ID", "Trip_ID", "Segment_ID"):
        if (events[column].astype("string").str.strip() == "").any():
            stop(f"Empty {column} values found in {EVENT_FILE.name}.")
        if (episodes[column].astype("string").str.strip() == "").any():
            stop(f"Empty {column} values found in {EPISODE_FILE.name}.")

    for column in ("Start_Time_ms", "End_Time_ms", "Duration_seconds"):
        events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in (
        "Episode_Start_ms", "Episode_End_ms", "Episode_Duration_seconds",
        "Events_Count", "Hard_Acceleration_Count",
        "Moderate_Acceleration_Count", "Hard_Braking_Count",
        "Moderate_Braking_Count",
    ):
        episodes[column] = pd.to_numeric(episodes[column], errors="coerce")

    if events[["Start_Time_ms", "End_Time_ms", "Duration_seconds"]].isna().any().any():
        stop("Invalid or missing event time/duration values were found.")
    if episodes[
        ["Episode_Start_ms", "Episode_End_ms", "Episode_Duration_seconds", "Events_Count"]
    ].isna().any().any():
        stop("Invalid or missing episode time/count values were found.")

    if (events["End_Time_ms"] < events["Start_Time_ms"]).any():
        stop("At least one event has End_Time_ms earlier than Start_Time_ms.")
    if (events["Duration_seconds"] < 0).any():
        stop("Negative event durations were found.")
    if (episodes["Events_Count"] < 1).any():
        stop("An episode has Events_Count below 1.")

    events["Canonical_Type"] = events["Canonical_Type"].astype(str).str.strip().str.upper()
    episodes["Behavior_Pattern"] = episodes["Behavior_Pattern"].astype(str).str.strip().str.upper()

    return events, episodes


def calculate_segment_transitions(events: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive event transitions within each vehicle/trip/segment.

    Only non-negative gaps up to 60 seconds are included, matching the
    Phase 9.2 transition-report window. Events are never linked across segments.
    """
    rows = []

    ordered = events.sort_values(
        ["Vehicle_ID", "Trip_ID", "Segment_ID", "Start_Time_ms", "End_Time_ms"],
        kind="mergesort",
    )

    group_columns = ["Vehicle_ID", "Trip_ID", "Segment_ID"]
    for keys, group in ordered.groupby(group_columns, sort=False, dropna=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for index in range(1, len(group)):
            previous = group.iloc[index - 1]
            current = group.iloc[index]
            gap_seconds = (float(current["Start_Time_ms"]) - float(previous["End_Time_ms"])) / 1000.0

            if 0 <= gap_seconds <= 60:
                previous_type = str(previous["Canonical_Type"])
                next_type = str(current["Canonical_Type"])
                previous_is_accel = previous_type in ACCELERATION_TYPES
                previous_is_brake = previous_type in BRAKING_TYPES
                next_is_accel = next_type in ACCELERATION_TYPES
                next_is_brake = next_type in BRAKING_TYPES

                rows.append(
                    {
                        "Vehicle_ID": keys[0],
                        "Trip_ID": keys[1],
                        "Segment_ID": keys[2],
                        "Gap_seconds": gap_seconds,
                        "Previous_Type": previous_type,
                        "Next_Type": next_type,
                        "Accel_To_Brake": int(previous_is_accel and next_is_brake),
                        "Brake_To_Accel": int(previous_is_brake and next_is_accel),
                        "Alternation": int(
                            (previous_is_accel and next_is_brake)
                            or (previous_is_brake and next_is_accel)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_trip_features(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    trip_keys = ["Vehicle_ID", "Trip_ID"]
    rows = []

    episode_groups = {
        key: group for key, group in episodes.groupby(trip_keys, sort=False, dropna=False)
    }

    if transitions.empty:
        transition_groups = {}
    else:
        transition_groups = {
            key: group for key, group in transitions.groupby(trip_keys, sort=False, dropna=False)
        }

    for key, group in events.groupby(trip_keys, sort=False, dropna=False):
        group = group.copy()
        types = group["Canonical_Type"].astype(str)
        counts = types.value_counts()

        start_ms = float(group["Start_Time_ms"].min())
        end_ms = float(group["End_Time_ms"].max())
        observed_span_seconds = max(0.0, (end_ms - start_ms) / 1000.0)

        hard_accel = int(counts.get("HARD_ACCELERATION", 0))
        moderate_accel = int(counts.get("MODERATE_ACCELERATION", 0))
        hard_brake = int(counts.get("HARD_BRAKING", 0))
        moderate_brake = int(counts.get("MODERATE_BRAKING", 0))

        acceleration_count = hard_accel + moderate_accel
        braking_count = hard_brake + moderate_brake
        hard_event_count = hard_accel + hard_brake
        total_events = int(len(group))

        episode_group = episode_groups.get(key)
        if episode_group is None:
            episode_count = 0
            mixed_episode_count = 0
            braking_only_count = 0
            acceleration_only_count = 0
            mean_episode_duration = 0.0
            max_episode_duration = 0.0
            episode_event_count = 0
        else:
            episode_count = int(len(episode_group))
            patterns = episode_group["Behavior_Pattern"]
            mixed_episode_count = int((patterns == "MIXED_ACCELERATION_AND_BRAKING").sum())
            braking_only_count = int((patterns == "BRAKING_ONLY").sum())
            acceleration_only_count = int((patterns == "ACCELERATION_ONLY").sum())
            mean_episode_duration = float(episode_group["Episode_Duration_seconds"].mean())
            max_episode_duration = float(episode_group["Episode_Duration_seconds"].max())
            episode_event_count = int(episode_group["Events_Count"].sum())

        transition_group = transition_groups.get(key)
        if transition_group is None:
            transition_count = 0
            accel_to_brake = 0
            brake_to_accel = 0
            alternation_count = 0
            mean_transition_gap = 0.0
            median_transition_gap = 0.0
        else:
            transition_count = int(len(transition_group))
            accel_to_brake = int(transition_group["Accel_To_Brake"].sum())
            brake_to_accel = int(transition_group["Brake_To_Accel"].sum())
            alternation_count = int(transition_group["Alternation"].sum())
            mean_transition_gap = float(transition_group["Gap_seconds"].mean())
            median_transition_gap = float(transition_group["Gap_seconds"].median())

        row = {
            "Vehicle_ID": key[0],
            "Trip_ID": key[1],
            "Segment_Count": int(group["Segment_ID"].nunique()),
            "Total_Events": total_events,
            "Hard_Acceleration_Count": hard_accel,
            "Moderate_Acceleration_Count": moderate_accel,
            "Hard_Braking_Count": hard_brake,
            "Moderate_Braking_Count": moderate_brake,
            "Total_Acceleration_Count": acceleration_count,
            "Total_Braking_Count": braking_count,
            "Hard_Event_Count": hard_event_count,
            "Acceleration_To_Braking_Ratio": safe_divide(acceleration_count, braking_count),
            "Hard_Event_Share": safe_divide(hard_event_count, total_events),
            "Acceleration_Event_Share": safe_divide(acceleration_count, total_events),
            "Braking_Event_Share": safe_divide(braking_count, total_events),
            "Observed_Event_Span_seconds": round(observed_span_seconds, 3),
            "Events_Per_Observed_Minute": round(
                safe_divide(total_events * 60.0, observed_span_seconds), 6
            ),
            "Hard_Events_Per_Observed_Minute": round(
                safe_divide(hard_event_count * 60.0, observed_span_seconds), 6
            ),
            "Mean_Event_Duration_seconds": round(float(group["Duration_seconds"].mean()), 3),
            "Median_Event_Duration_seconds": round(float(group["Duration_seconds"].median()), 3),
            "Max_Event_Duration_seconds": round(float(group["Duration_seconds"].max()), 3),
            "Episode_Count": episode_count,
            "Mixed_Episode_Count": mixed_episode_count,
            "Acceleration_Only_Episode_Count": acceleration_only_count,
            "Braking_Only_Episode_Count": braking_only_count,
            "Mixed_Episode_Share": round(safe_divide(mixed_episode_count, episode_count), 6),
            "Mean_Episode_Duration_seconds": round(mean_episode_duration, 3),
            "Max_Episode_Duration_seconds": round(max_episode_duration, 3),
            "Episode_Event_Count_Check": episode_event_count,
            "Transition_Count_Within_60s": transition_count,
            "Acceleration_To_Braking_Transitions": accel_to_brake,
            "Braking_To_Acceleration_Transitions": brake_to_accel,
            "Alternation_Transition_Count": alternation_count,
            "Mean_Transition_Gap_seconds": round(mean_transition_gap, 3),
            "Median_Transition_Gap_seconds": round(median_transition_gap, 3),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        stop("No trip-level feature rows were generated.")

    result["Episode_Event_Count_Matches_Audit"] = (
        result["Episode_Event_Count_Check"] == result["Total_Events"]
    )
    return result.sort_values(["Vehicle_ID", "Trip_ID"], kind="mergesort").reset_index(drop=True)


def build_vehicle_features(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for vehicle_id, group in trips.groupby("Vehicle_ID", sort=False, dropna=False):
        total_events = int(group["Total_Events"].sum())
        total_hard_events = int(group["Hard_Event_Count"].sum())
        total_acceleration = int(group["Total_Acceleration_Count"].sum())
        total_braking = int(group["Total_Braking_Count"].sum())
        total_episodes = int(group["Episode_Count"].sum())
        total_mixed_episodes = int(group["Mixed_Episode_Count"].sum())

        rows.append(
            {
                "Vehicle_ID": vehicle_id,
                "Trip_Count_With_Events": int(len(group)),
                "Total_Events": total_events,
                "Mean_Events_Per_Trip": round(float(group["Total_Events"].mean()), 3),
                "Median_Events_Per_Trip": round(float(group["Total_Events"].median()), 3),
                "Maximum_Events_In_One_Trip": int(group["Total_Events"].max()),
                "Total_Hard_Acceleration_Count": int(group["Hard_Acceleration_Count"].sum()),
                "Total_Moderate_Acceleration_Count": int(group["Moderate_Acceleration_Count"].sum()),
                "Total_Hard_Braking_Count": int(group["Hard_Braking_Count"].sum()),
                "Total_Moderate_Braking_Count": int(group["Moderate_Braking_Count"].sum()),
                "Total_Acceleration_Count": total_acceleration,
                "Total_Braking_Count": total_braking,
                "Total_Hard_Event_Count": total_hard_events,
                "Hard_Event_Share": round(safe_divide(total_hard_events, total_events), 6),
                "Acceleration_To_Braking_Ratio": round(
                    safe_divide(total_acceleration, total_braking), 6
                ),
                "Mean_Events_Per_Observed_Minute": round(
                    float(group["Events_Per_Observed_Minute"].mean()), 6
                ),
                "Mean_Hard_Events_Per_Observed_Minute": round(
                    float(group["Hard_Events_Per_Observed_Minute"].mean()), 6
                ),
                "Trips_With_At_Least_One_Hard_Event": int(
                    (group["Hard_Event_Count"] > 0).sum()
                ),
                "Share_Of_Trips_With_Hard_Event": round(
                    safe_divide((group["Hard_Event_Count"] > 0).sum(), len(group)), 6
                ),
                "Total_Episodes": total_episodes,
                "Total_Mixed_Episodes": total_mixed_episodes,
                "Mixed_Episode_Share": round(
                    safe_divide(total_mixed_episodes, total_episodes), 6
                ),
                "Total_Acceleration_To_Braking_Transitions": int(
                    group["Acceleration_To_Braking_Transitions"].sum()
                ),
                "Total_Braking_To_Acceleration_Transitions": int(
                    group["Braking_To_Acceleration_Transitions"].sum()
                ),
                "Total_Alternation_Transitions": int(
                    group["Alternation_Transition_Count"].sum()
                ),
                "Trip_Hard_Event_Count_Std": round(
                    float(group["Hard_Event_Count"].std(ddof=0)), 3
                ),
                "Trip_Event_Count_Std": round(
                    float(group["Total_Events"].std(ddof=0)), 3
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        stop("No vehicle-level feature rows were generated.")
    return result.sort_values("Vehicle_ID", kind="mergesort").reset_index(drop=True)


def make_report(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    transitions: pd.DataFrame,
    trips: pd.DataFrame,
    vehicles: pd.DataFrame,
    backups: list[str],
) -> str:
    event_counts = events["Canonical_Type"].value_counts()
    audit_counts = events["Audit_Status"].astype(str).value_counts()
    pattern_counts = episodes["Behavior_Pattern"].value_counts()

    unknown_types = sorted(set(events["Canonical_Type"]) - set(EVENT_TYPES))
    duplicate_event_ids = int(events["Event_ID"].duplicated().sum())
    episode_event_sum = int(episodes["Events_Count"].sum())
    audit_event_count = int(len(events))
    trip_episode_check_failures = int((~trips["Episode_Event_Count_Matches_Audit"]).sum())

    lines = [
        "=" * 72,
        "DANGEROUS DRIVING AI - PHASE 10.1 REPORT",
        "TRIP AND VEHICLE FEATURE ENGINEERING",
        "=" * 72,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "INPUT FILES",
        "-" * 72,
        EVENT_FILE.name,
        EPISODE_FILE.name,
        "",
        "SOURCE DATA COUNTS",
        "-" * 72,
        f"Audit events: {audit_event_count:,}",
        f"Behavior episodes: {len(episodes):,}",
        f"Vehicles with events: {events['Vehicle_ID'].nunique():,}",
        f"Vehicle-trip pairs: {events[['Vehicle_ID', 'Trip_ID']].drop_duplicates().shape[0]:,}",
        f"Vehicle-trip-segment groups: {events[['Vehicle_ID', 'Trip_ID', 'Segment_ID']].drop_duplicates().shape[0]:,}",
        "",
        "EVENT TYPE COUNTS",
        "-" * 72,
    ]

    for event_type in EVENT_TYPES:
        lines.append(f"{event_type}: {int(event_counts.get(event_type, 0)):,}")
    if unknown_types:
        lines.append(f"Other/unknown canonical types: {', '.join(unknown_types)}")
    else:
        lines.append("Other/unknown canonical types: 0")

    lines.extend(
        [
            "",
            "AUDIT STATUS COUNTS",
            "-" * 72,
        ]
    )
    for status, count in audit_counts.items():
        lines.append(f"{status}: {int(count):,}")

    lines.extend(
        [
            "",
            "BEHAVIOR EPISODE PATTERNS",
            "-" * 72,
        ]
    )
    for pattern, count in pattern_counts.items():
        lines.append(f"{pattern}: {int(count):,}")

    lines.extend(
        [
            "",
            "FEATURE OUTPUTS",
            "-" * 72,
            f"Trip feature rows: {len(trips):,}",
            f"Vehicle feature rows: {len(vehicles):,}",
            f"Transitions counted within 0-60 seconds and within the same segment: {len(transitions):,}",
            f"Acceleration-to-braking transitions: {int(trips['Acceleration_To_Braking_Transitions'].sum()):,}",
            f"Braking-to-acceleration transitions: {int(trips['Braking_To_Acceleration_Transitions'].sum()):,}",
            f"Alternation transitions: {int(trips['Alternation_Transition_Count'].sum()):,}",
            "",
            "DATA INTEGRITY CHECKS",
            "-" * 72,
            f"Duplicate Event_ID rows in audit: {duplicate_event_ids:,}",
            f"Episode Events_Count sum: {episode_event_sum:,}",
            f"Audit event rows: {audit_event_count:,}",
            f"Trip groups failing episode-event count reconciliation: {trip_episode_check_failures:,}",
            f"Unknown canonical event types: {len(unknown_types):,}",
            "",
            "INTERPRETATION NOTES",
            "-" * 72,
            "- Observed_Event_Span_seconds is the span from the first event start to the last event end, not the full trip duration.",
            "- Event rates use that observed event span; they must not be interpreted as rates over the complete trip.",
            "- Transition counts include consecutive events only when the gap is 0 to 60 seconds and the events belong to the same segment.",
            "- Mixed episodes are analytical groups from Phase 9.2, not proof of one dangerous maneuver.",
            "- Extracted event labels originate from threshold rules and are not independently human-verified ground truth.",
            "- These features are descriptive candidates, not a validated driver-risk score.",
            "",
            "OUTPUT FILES",
            "-" * 72,
            TRIP_OUTPUT.name,
            VEHICLE_OUTPUT.name,
            REPORT_OUTPUT.name,
            "",
            "BACKUPS",
            "-" * 72,
        ]
    )

    if backups:
        lines.extend(backups)
    else:
        lines.append("No previous Phase 10.1 output files needed backup.")

    lines.extend(["", "STATUS: COMPLETED", "=" * 72])
    return "\n".join(lines)


def main() -> int:
    print("=" * 72)
    print("DANGEROUS DRIVING AI - PHASE 10.1")
    print("TRIP AND VEHICLE FEATURE ENGINEERING")
    print("=" * 72)
    print(f"Project folder: {BASE_DIR}")
    print("Loading Phase 9.2 files...")

    backups = backup_existing_outputs()
    events, episodes = load_data()

    print(f"Loaded events: {len(events):,}")
    print(f"Loaded episodes: {len(episodes):,}")

    duplicate_event_ids = int(events["Event_ID"].duplicated().sum())
    if duplicate_event_ids:
        stop(
            f"Found {duplicate_event_ids} duplicate Event_ID rows in the audit file. "
            "Resolve this before feature engineering."
        )

    print("Calculating event transitions...")
    transitions = calculate_segment_transitions(events)

    print("Building trip-level features...")
    trips = build_trip_features(events, episodes, transitions)

    print("Building vehicle-level features...")
    vehicles = build_vehicle_features(trips)

    failed_reconciliations = int((~trips["Episode_Event_Count_Matches_Audit"]).sum())
    if failed_reconciliations:
        stop(
            f"{failed_reconciliations} trip groups do not reconcile between event "
            "and episode counts. Outputs were not written."
        )

    print("Saving output CSV files...")
    trips.to_csv(TRIP_OUTPUT, index=False, encoding="utf-8-sig")
    vehicles.to_csv(VEHICLE_OUTPUT, index=False, encoding="utf-8-sig")

    report = make_report(events, episodes, transitions, trips, vehicles, backups)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    print()
    print(f"Trip features: {len(trips):,} rows")
    print(f"Vehicle features: {len(vehicles):,} rows")
    print(f"Transitions: {len(transitions):,}")
    print(f"Report: {REPORT_OUTPUT.name}")
    print()
    print("PHASE 10.1 COMPLETED SUCCESSFULLY")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print()
        print("PHASE 10.1 FAILED")
        print(str(exc))
        print()
        print("Technical details:")
        traceback.print_exc()
        input("Press Enter to close...")
        sys.exit(1)
