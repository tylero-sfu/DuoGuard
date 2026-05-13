"""Generate toy data for the sanitized DuoGuard demo.

The generated files are fully synthetic. They are not sampled from, derived
from, or statistically fitted to the proprietary telemetry used in the paper.
Their only purpose is to demonstrate the expected schema and executable code
path.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RNG = random.Random(17)

WEAPONS = [
    ("w_ar", "AR", "2x"),
    ("w_smg", "SMG", "1x"),
    ("w_lmg", "LMG", "2x"),
    ("w_dmr", "DMR", "4x"),
    ("w_sr", "SR", "6x"),
    ("w_sg", "SHOTGUN", "none"),
    ("w_pistol", "PISTOL", "none"),
]
DAMAGE_REGIONS = ["head", "torso", "pelvis", "arm", "leg"]


def player_rows() -> list[dict[str, object]]:
    rows = []
    for i in range(1, 21):
        split = "train" if i <= 12 else "valid" if i <= 16 else "test"
        label = 1 if i in {3, 6, 9, 12, 15, 18} else 0
        rows.append({"player_id": f"p{i:03d}", "label": label, "split": split})
    return rows


def sequence_count(player_index: int) -> int:
    pattern = [1, 2, 3, 5, 8, 12, 16, 20]
    return pattern[player_index % len(pattern)]


def add_sequence(events: list[dict[str, object]], player: dict[str, object], seq_no: int, match_no: int) -> None:
    player_id = str(player["player_id"])
    label = int(player["label"])
    weapon_id, weapon_category, scope = WEAPONS[(seq_no + int(player_id[1:])) % len(WEAPONS)]
    target_id = f"t{player_id[1:]}_{seq_no:02d}"
    match_id = f"m{match_no:04d}"
    attack_id = f"a{player_id[1:]}_{seq_no:03d}"

    n_shots = RNG.randint(2, 7)
    if weapon_category in {"DMR", "SR", "SHOTGUN", "PISTOL"}:
        n_shots = RNG.randint(1, 4)
    base_time = match_no * 100000 + seq_no * 4000 + RNG.randint(0, 250)
    distance = {
        "SHOTGUN": RNG.uniform(5, 25),
        "PISTOL": RNG.uniform(10, 60),
        "SMG": RNG.uniform(10, 80),
        "AR": RNG.uniform(30, 180),
        "LMG": RNG.uniform(40, 220),
        "DMR": RNG.uniform(80, 350),
        "SR": RNG.uniform(120, 500),
    }.get(weapon_category, RNG.uniform(20, 200))

    # Positive-label toy players receive slightly earlier and more central hits.
    hit_start = 1 if label else RNG.randint(1, min(3, n_shots))
    central_regions = ["head", "torso", "torso"] if label else DAMAGE_REGIONS
    terminal_done = False

    for shot_idx in range(n_shots):
        t = base_time + shot_idx * RNG.randint(120, 450)
        is_hit = shot_idx >= hit_start and RNG.random() < (0.72 if label else 0.48)
        event_type = "shot"
        damage_region = "none"
        if is_hit:
            damage_region = RNG.choice(central_regions)
            event_type = "hit"
            if label and not terminal_done and shot_idx >= max(1, hit_start + 1) and RNG.random() < 0.25:
                event_type = "groggy"
                terminal_done = True
            elif terminal_done and RNG.random() < 0.25:
                event_type = "flush"
        if shot_idx == n_shots - 1 and label and not terminal_done and is_hit and RNG.random() < 0.25:
            event_type = "final_elimination"

        events.append(
            {
                "match_id": match_id,
                "player_id": player_id,
                "target_id": target_id,
                "event_time_ms": t,
                "event_type": event_type,
                "attack_id": attack_id,
                "weapon_id": weapon_id,
                "weapon_category": weapon_category,
                "scope": scope,
                "damage_region": damage_region,
                "distance_m": "" if event_type == "shot" and RNG.random() < 0.7 else round(distance + RNG.uniform(-5, 5), 2),
                "attacker_x": round(RNG.uniform(0, 1000), 2),
                "attacker_y": round(RNG.uniform(0, 1000), 2),
                "target_x": round(RNG.uniform(0, 1000), 2),
                "target_y": round(RNG.uniform(0, 1000), 2),
            }
        )


def main() -> None:
    labels = player_rows()
    events: list[dict[str, object]] = []
    match_no = 1
    for idx, player in enumerate(labels):
        for seq_no in range(sequence_count(idx)):
            add_sequence(events, player, seq_no, match_no)
            match_no += 1

    with (ROOT / "synthetic_player_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["player_id", "label", "split"])
        writer.writeheader()
        writer.writerows(labels)

    with (ROOT / "synthetic_events.csv").open("w", newline="") as f:
        fieldnames = [
            "match_id",
            "player_id",
            "target_id",
            "event_time_ms",
            "event_type",
            "attack_id",
            "weapon_id",
            "weapon_category",
            "scope",
            "damage_region",
            "distance_m",
            "attacker_x",
            "attacker_y",
            "target_x",
            "target_y",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    print(f"wrote {len(labels)} players and {len(events)} events")


if __name__ == "__main__":
    main()
