"""Sanitized preprocessing utilities for DuoGuard.

The functions in this module operate on de-identified event tables. They do
not depend on proprietary identifiers, operational action records, or label
construction rules.

Expected event columns for the generic path:

```
match_id, player_id, target_id, event_time_ms, event_type, attack_id,
weapon_id, weapon_category, scope, damage_region, distance_m,
attacker_x, attacker_y, target_x, target_y
```

Only `player_id` and `target_id` need to be stable within a dataset; they
should already be anonymized before use with this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .features import (
    DAMAGE_REGION_TO_CODE,
    RELATIVE_DIRECTION_TO_CODE,
    SCOPE_TO_CODE,
    SEQUENCE_FEATURES,
    STATISTICAL_FEATURES,
    WEAPON_CATEGORY_TO_CODE,
)


@dataclass(frozen=True)
class SequenceConfig:
    max_sequence_length: int = 32
    first_k: int = 7
    default_gap_ms: int = 1000
    weapon_gap_cutoffs_ms: dict[str, int] | None = None

    def cutoff_for(self, weapon_category: str) -> int:
        mapping = self.weapon_gap_cutoffs_ms or {}
        return int(mapping.get(str(weapon_category).upper(), self.default_gap_ms))


DEFAULT_WEAPON_GAP_CUTOFFS_MS = {
    "AR": 400,
    "SMG": 250,
    "LMG": 300,
    "BURST_AR": 1000,
    "DMR": 1500,
    "PISTOL": 1000,
    "SHOTGUN": 1500,
    "SR": 1500,
    "OTHER": 1000,
}


def integrate_combat_logs(
    attack_events: pd.DataFrame,
    damage_events: pd.DataFrame | None = None,
    groggy_events: pd.DataFrame | None = None,
    elimination_events: pd.DataFrame | None = None,
    flush_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge generic combat-log tables into a unified event table.

    This is a disclosure-safe version of the event integration step described
    in the paper. The function assumes that any proprietary source tables have
    already been converted to de-identified tables with generic column names.

    Parameters
    ----------
    attack_events:
        Firing events. These rows become `event_type == "shot"`.
    damage_events:
        Hit/damage events. These rows become `event_type == "hit"`.
    groggy_events:
        Incapacitation/down-but-not-out events. These rows become
        `event_type == "groggy"`.
    elimination_events:
        Final elimination events. These rows become
        `event_type == "final_elimination"`.
    flush_events:
        Final eliminations of already incapacitated targets. These rows become
        `event_type == "flush"`.

    Returns
    -------
    pandas.DataFrame
        A normalized event table sorted by match, player, and time.
    """

    pieces = []

    def add(frame: pd.DataFrame | None, event_type: str) -> None:
        if frame is None or frame.empty:
            return
        part = frame.copy()
        part["event_type"] = event_type
        pieces.append(part)

    add(attack_events, "shot")
    add(damage_events, "hit")
    add(groggy_events, "groggy")
    add(elimination_events, "final_elimination")
    add(flush_events, "flush")
    if not pieces:
        raise ValueError("At least one combat-log table must be provided.")
    return normalize_event_table(pd.concat(pieces, ignore_index=True, sort=False))


def match_outcomes(
    events: pd.DataFrame,
    outcome_window_ms: int = 1500,
) -> pd.DataFrame:
    """Attach coarse downstream outcome flags to firing/hit events.

    The public implementation uses only generic event types and a short
    temporal window within the same match, player, and target. It does not
    expose or depend on proprietary enforcement information.
    """

    df = normalize_event_table(events)
    df["has_downstream_groggy"] = False
    df["has_downstream_final_elimination"] = False
    df["has_downstream_flush"] = False

    grouped = df.groupby(["match_id", "player_id", "target_id"], dropna=False, sort=False)
    updates = []
    for _, part in grouped:
        part = part.sort_values("event_time_ms")
        outcome_rows = part[part["event_type"].isin(["groggy", "final_elimination", "flush"])]
        for idx, row in part.iterrows():
            if row["event_type"] not in {"shot", "hit"}:
                continue
            t0 = int(row["event_time_ms"])
            nearby = outcome_rows[
                (outcome_rows["event_time_ms"] >= t0)
                & (outcome_rows["event_time_ms"] <= t0 + int(outcome_window_ms))
            ]
            updates.append(
                {
                    "index": idx,
                    "has_downstream_groggy": bool((nearby["event_type"] == "groggy").any()),
                    "has_downstream_final_elimination": bool((nearby["event_type"] == "final_elimination").any()),
                    "has_downstream_flush": bool((nearby["event_type"] == "flush").any()),
                }
            )

    for item in updates:
        idx = item.pop("index")
        for key, value in item.items():
            df.at[idx, key] = value
    return df


def normalize_event_table(events: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized event table with safe defaults for optional fields."""

    required = ["match_id", "player_id", "event_time_ms", "event_type", "weapon_category"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = events.copy()
    defaults = {
        "target_id": "",
        "attack_id": "",
        "weapon_id": "unknown_weapon",
        "scope": "none",
        "damage_region": "none",
        "distance_m": np.nan,
        "attacker_x": np.nan,
        "attacker_y": np.nan,
        "target_x": np.nan,
        "target_y": np.nan,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    out["event_time_ms"] = pd.to_numeric(out["event_time_ms"], errors="coerce")
    out = out.dropna(subset=["event_time_ms"])
    out["event_time_ms"] = out["event_time_ms"].astype("int64")
    return out.sort_values(["match_id", "player_id", "event_time_ms"]).reset_index(drop=True)


def segment_shooting_sequences(
    events: pd.DataFrame,
    config: SequenceConfig | None = None,
) -> pd.DataFrame:
    """Assign a sequence id to temporally adjacent firing events.

    Consecutive events from the same player and match are grouped while their
    inter-event gap remains below the weapon-category-specific cutoff. The
    function keeps hit-bearing sequences by default because the public method
    focuses on localized shooting evidence with observed outcome context.
    """

    config = config or SequenceConfig(weapon_gap_cutoffs_ms=DEFAULT_WEAPON_GAP_CUTOFFS_MS)
    df = normalize_event_table(events)
    df = df[df["event_type"].isin(["shot", "hit", "groggy", "final_elimination", "flush"])].copy()
    if df.empty:
        return df.assign(sequence_id=pd.Series(dtype="str"))

    sequence_ids: list[str] = []
    current = -1
    previous_key = None
    previous_time = None
    previous_weapon = None

    for row in df.itertuples(index=False):
        key = (row.match_id, row.player_id)
        weapon_category = str(row.weapon_category).upper()
        gap = None if previous_time is None else int(row.event_time_ms) - int(previous_time)
        cutoff = config.cutoff_for(previous_weapon or weapon_category)
        starts_new = key != previous_key or gap is None or gap > cutoff
        if starts_new:
            current += 1
        sequence_ids.append(f"s{current:08d}")
        previous_key = key
        previous_time = int(row.event_time_ms)
        previous_weapon = weapon_category

    df["sequence_id"] = sequence_ids

    hit_like = df["event_type"].isin(["hit", "groggy", "final_elimination", "flush"])
    evidence_sequences = set(df.loc[hit_like, "sequence_id"].tolist())
    return df[df["sequence_id"].isin(evidence_sequences)].reset_index(drop=True)


def interpolate_missing_distances(seq: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing distances within a sequence and target."""

    seq = seq.sort_values("event_time_ms").copy()
    seq["distance_valid"] = seq["distance_m"].notna().astype("float32")
    pieces = []
    for _, part in seq.groupby("target_id", dropna=False, sort=False):
        part = part.copy()
        if part["distance_m"].notna().sum() >= 2:
            part["distance_m"] = part["distance_m"].interpolate(method="linear", limit_area="inside")
        pieces.append(part)
    out = pd.concat(pieces, axis=0).sort_values("event_time_ms").reset_index(drop=True)
    return out


def _relative_direction(row: pd.Series) -> str:
    if pd.isna(row.get("attacker_x")) or pd.isna(row.get("target_x")):
        return "unknown"
    dx = float(row["target_x"]) - float(row["attacker_x"])
    dy = float(row["target_y"]) - float(row["attacker_y"])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "stationary"
    horizontal = "right" if dx > 0 else "left"
    vertical = "front" if dy > 0 else "back"
    if abs(dx) > 2 * abs(dy):
        return horizontal
    if abs(dy) > 2 * abs(dx):
        return vertical
    return f"{vertical}_{horizontal}"


def _distance_bin(distance: float) -> int:
    if not np.isfinite(distance):
        return 0
    bins = [0, 25, 50, 100, 200, 400, 800]
    return int(np.digitize([distance], bins, right=False)[0])


def build_sequence_tensor(
    seq: pd.DataFrame,
    config: SequenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(x_seq, x_stat, mask)` for one shooting sequence."""

    config = config or SequenceConfig()
    seq = interpolate_missing_distances(seq)
    seq = seq.sort_values("event_time_ms").reset_index(drop=True)
    max_len = int(config.max_sequence_length)
    rows = seq.iloc[:max_len].copy()

    rows["delta_time_ms"] = rows["event_time_ms"].diff().fillna(0).clip(lower=0)
    rows["is_hit"] = rows["event_type"].isin(["hit", "groggy", "final_elimination", "flush"]).astype("float32")
    rows["is_groggy"] = (rows["event_type"] == "groggy").astype("float32")
    rows["is_final_elimination"] = (rows["event_type"] == "final_elimination").astype("float32")
    rows["is_flush"] = (rows["event_type"] == "flush").astype("float32")
    rows["damage_region_code"] = rows["damage_region"].map(DAMAGE_REGION_TO_CODE).fillna(6)
    rows["scope_code"] = rows["scope"].map(SCOPE_TO_CODE).fillna(7)
    rows["distance_m"] = pd.to_numeric(rows["distance_m"], errors="coerce").fillna(0.0)
    rows["distance_bin_code"] = rows["distance_m"].map(_distance_bin)
    rows["weapon_category_code"] = rows["weapon_category"].str.upper().map(WEAPON_CATEGORY_TO_CODE).fillna(8)
    rows["attacker_speed_mps"] = rows.get("attacker_speed_mps", pd.Series(0.0, index=rows.index)).fillna(0.0)
    rows["attacker_accel_mps2"] = rows.get("attacker_accel_mps2", pd.Series(0.0, index=rows.index)).fillna(0.0)
    rows["relative_direction_code"] = rows.apply(_relative_direction, axis=1).map(RELATIVE_DIRECTION_TO_CODE).fillna(0)
    rows["relative_direction_valid"] = (rows["relative_direction_code"] != 0).astype("float32")
    rows["target_switch"] = (rows["target_id"].fillna("").ne(rows["target_id"].fillna("").shift())).astype("float32")
    rows.loc[rows.index[:1], "target_switch"] = 0.0

    x_seq = np.zeros((max_len, len(SEQUENCE_FEATURES)), dtype=np.float32)
    values = rows[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
    x_seq[: len(values), :] = values
    mask = np.zeros((max_len,), dtype=bool)
    mask[: len(values)] = True
    x_stat = build_statistical_descriptor(rows, config=config)
    return x_seq, x_stat, mask


def build_statistical_descriptor(seq: pd.DataFrame, config: SequenceConfig | None = None) -> np.ndarray:
    """Build a domain-informed statistical descriptor for one sequence."""

    config = config or SequenceConfig()
    first_k = int(config.first_k)
    hits = seq[seq["is_hit"] > 0].copy()
    n_shots = float(len(seq))
    n_hits = float(len(hits))
    first = seq.iloc[:first_k]
    hit_regions = hits["damage_region"].fillna("none").tolist()

    def rate(region: str) -> float:
        return float((hits["damage_region"] == region).mean()) if len(hits) else 0.0

    first_hit_positions = np.flatnonzero(seq["is_hit"].to_numpy() > 0)
    shots_to_first_hit = float(first_hit_positions[0] + 1) if len(first_hit_positions) else n_shots + 1.0

    same_region_pairs = 0
    for a, b in zip(hit_regions, hit_regions[1:]):
        same_region_pairs += int(a == b and a != "none")
    same_region_transition_rate = same_region_pairs / max(1, len(hit_regions) - 1)

    returns = 0
    opportunities = 0
    previous_hit_region = None
    saw_miss_after_hit = False
    for row in seq.itertuples(index=False):
        is_hit = bool(getattr(row, "is_hit"))
        region = getattr(row, "damage_region")
        if is_hit:
            if saw_miss_after_hit and previous_hit_region is not None:
                opportunities += 1
                returns += int(region == previous_hit_region)
            previous_hit_region = region
            saw_miss_after_hit = False
        elif previous_hit_region is not None:
            saw_miss_after_hit = True
    hit_region_return_rate = returns / max(1, opportunities)

    values = {
        "n_shots": n_shots,
        "n_hits": n_hits,
        "hit_rate": n_hits / max(1.0, n_shots),
        "first_k_hit_rate": float(first["is_hit"].mean()) if len(first) else 0.0,
        "first_k_groggy_count": float(first["is_groggy"].sum()) if "is_groggy" in first else 0.0,
        "first_k_final_elimination_count": float(first["is_final_elimination"].sum()) if "is_final_elimination" in first else 0.0,
        "head_hit_rate": rate("head"),
        "torso_hit_rate": rate("torso"),
        "pelvis_hit_rate": rate("pelvis"),
        "arm_hit_rate": rate("arm"),
        "leg_hit_rate": rate("leg"),
        "shots_to_first_hit": shots_to_first_hit,
        "mean_distance_m": float(hits["distance_m"].mean()) if len(hits) else 0.0,
        "max_distance_m": float(hits["distance_m"].max()) if len(hits) else 0.0,
        "mean_attacker_speed_mps": float(seq["attacker_speed_mps"].mean()) if "attacker_speed_mps" in seq else 0.0,
        "mean_attacker_accel_mps2": float(seq["attacker_accel_mps2"].mean()) if "attacker_accel_mps2" in seq else 0.0,
        "same_region_transition_rate": same_region_transition_rate,
        "hit_region_return_rate": hit_region_return_rate,
        "unique_hit_regions": float(len(set(r for r in hit_regions if r != "none"))),
        "target_switch_count": float(seq["target_switch"].sum()) if "target_switch" in seq else 0.0,
    }
    return np.asarray([values[name] for name in STATISTICAL_FEATURES], dtype=np.float32)


def build_sequence_dataset(
    events: pd.DataFrame,
    config: SequenceConfig | None = None,
) -> pd.DataFrame:
    """Convert event rows into one row per shooting sequence.

    Tensor columns are stored as NumPy arrays for use by the demo. Production
    pipelines can serialize the returned columns with Parquet, NumPy, or any
    internal feature store.
    """

    config = config or SequenceConfig(weapon_gap_cutoffs_ms=DEFAULT_WEAPON_GAP_CUTOFFS_MS)
    segmented = segment_shooting_sequences(events, config=config)
    rows = []
    for sequence_id, seq in segmented.groupby("sequence_id", sort=False):
        x_seq, x_stat, mask = build_sequence_tensor(seq, config=config)
        first = seq.iloc[0]
        rows.append(
            {
                "sequence_id": sequence_id,
                "match_id": first["match_id"],
                "player_id": first["player_id"],
                "weapon_id": first.get("weapon_id", "unknown_weapon"),
                "weapon_category": first.get("weapon_category", "OTHER"),
                "x_seq": x_seq,
                "x_stat": x_stat,
                "mask": mask,
            }
        )
    return pd.DataFrame(rows)
