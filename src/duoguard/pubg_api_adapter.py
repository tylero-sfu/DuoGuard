"""Adapter utilities for PUBG API-style combat telemetry.

This module converts public-telemetry-style PUBG combat events into the generic
event table consumed by `duoguard.preprocessing`. It intentionally does not
contain operational enforcement logic, player-label construction, or any
proprietary anti-cheat fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PUBG_COMBAT_EVENT_TYPES = {
    "LogPlayerAttack": "shot",
    "LogPlayerTakeDamage": "hit",
    "LogPlayerMakeGroggy": "groggy",
    "LogPlayerKill": "final_elimination",
    "LogPlayerKillV2": "final_elimination",
}


def load_pubg_json_events(path: str | Path) -> list[dict[str, Any]]:
    """Load PUBG telemetry events from a JSON or JSON-lines file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a list of event objects.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _get(obj: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = obj or {}
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _name(character: dict[str, Any] | None) -> str:
    return str(_get(character, "name", default="") or "")


def _location(character: dict[str, Any] | None) -> tuple[float, float]:
    loc = _get(character, "location", default={}) or {}
    return float(loc.get("x", np.nan)), float(loc.get("y", np.nan))


def _event_time_ms(event: dict[str, Any]) -> int:
    for key in ("elapsedTime", "event_time_ms", "time_ms"):
        value = event.get(key)
        if value is not None:
            return int(float(value) * (1000 if key == "elapsedTime" else 1))
    return 0


def _attack_id(event: dict[str, Any]) -> str:
    for key in ("attackId", "attack_id", "fireWeaponStackCount"):
        value = event.get(key)
        if value is not None:
            return str(value)
    return ""


def _damage_region(event: dict[str, Any]) -> str:
    reason = str(event.get("damageReason", "") or event.get("damage_region", "")).lower()
    if "head" in reason:
        return "head"
    if "torso" in reason or "chest" in reason:
        return "torso"
    if "pelvis" in reason:
        return "pelvis"
    if "arm" in reason:
        return "arm"
    if "leg" in reason:
        return "leg"
    return "none"


def _weapon_id(event: dict[str, Any]) -> str:
    return str(
        event.get("weapon_id")
        or event.get("weaponId")
        or event.get("damageCauserName")
        or event.get("weapon")
        or "unknown_weapon"
    )


def _weapon_category(weapon_id: str, mapping: dict[str, str] | None = None) -> str:
    if mapping and weapon_id in mapping:
        return mapping[weapon_id]
    text = weapon_id.upper()
    if any(k in text for k in ("AKM", "M416", "SCAR", "QBZ", "AUG", "BERYL", "G36")):
        return "AR"
    if any(k in text for k in ("UMP", "VECTOR", "UZI", "MP5", "TOMMY", "P90")):
        return "SMG"
    if any(k in text for k in ("M249", "DP28", "MG3")):
        return "LMG"
    if any(k in text for k in ("MINI", "SLR", "SKS", "MK12", "QBU", "VSS")):
        return "DMR"
    if any(k in text for k in ("KAR98", "M24", "AWM", "MOSIN", "LYNX")):
        return "SR"
    if any(k in text for k in ("S12K", "S1897", "S686", "DBS", "O12")):
        return "SHOTGUN"
    if any(k in text for k in ("P1911", "P92", "R1895", "DEAGLE", "P18C")):
        return "PISTOL"
    return "OTHER"


def pubg_events_to_generic_table(
    events: Iterable[dict[str, Any]],
    match_id: str,
    weapon_category_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert PUBG API-style event objects into DuoGuard's generic schema."""

    rows: list[dict[str, Any]] = []
    for event in events:
        event_type_raw = str(event.get("_T") or event.get("event_type") or "")
        if event_type_raw not in PUBG_COMBAT_EVENT_TYPES:
            continue

        generic_type = PUBG_COMBAT_EVENT_TYPES[event_type_raw]
        attacker = event.get("attacker") or event.get("character") or event.get("killer")
        target = event.get("victim") or event.get("target") or event.get("character")
        ax, ay = _location(attacker)
        tx, ty = _location(target)
        weapon = _weapon_id(event)

        rows.append(
            {
                "match_id": match_id,
                "player_id": _name(attacker),
                "target_id": _name(target),
                "event_time_ms": _event_time_ms(event),
                "event_type": generic_type,
                "attack_id": _attack_id(event),
                "weapon_id": weapon,
                "weapon_category": _weapon_category(weapon, weapon_category_mapping),
                "scope": str(event.get("scope", "none") or "none"),
                "damage_region": _damage_region(event),
                "distance_m": event.get("distance", event.get("distance_m", np.nan)),
                "attacker_x": ax,
                "attacker_y": ay,
                "target_x": tx,
                "target_y": ty,
            }
        )
    return pd.DataFrame(rows)

