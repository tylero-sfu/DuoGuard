"""Ordered model inputs. No label construction is performed here."""

SEQUENCE_FEATURES = [
    "delta_time_ms", "scope_code", "is_hit", "is_groggy_shot", "is_flush_shot",
    "is_direct_kill_shot", "distance_log", "distance_bin_code", "distance_valid",
    "attacker_speed_mps_filled", "attacker_accel_mps2_filled", "rel_dir_code_filled",
    "rel_dir_valid", "damage_part_code", "weapon_category", "victim_switch_event",
]
SCALE_INDICES = [0, 6, 9, 10, 15]
CATEGORY_SIZES = {1: 8, 7: 6, 11: 10, 13: 7, 14: 7}
BINARY_INDICES = [2, 3, 4, 5, 8, 12, 15]
WEAPON_CATEGORIES = {"AR": 0, "SMG": 1, "LMG": 2, "DMR": 3, "SR": 4,
                     "SHOTGUN": 5, "PISTOL": 6}
GAP_CUTOFFS_MS = {"AR": 550, "SMG": 500, "LMG": 550, "DMR": 1000,
                  "SR": 2500, "SHOTGUN": 1250, "PISTOL": 900}
