"""Feature schema used by the sanitized DuoGuard implementation.

The schema names are intentionally generic. They describe the model input
contract without exposing proprietary telemetry fields or operational label
construction logic.
"""

SEQUENCE_FEATURES = [
    "delta_time_ms",
    "is_hit",
    "is_groggy",
    "is_final_elimination",
    "is_flush",
    "damage_region_code",
    "distance_m",
    "distance_valid",
    "attacker_speed_mps",
    "attacker_accel_mps2",
    "relative_direction_code",
    "relative_direction_valid",
    "scope_code",
    "distance_bin_code",
    "weapon_category_code",
    "target_switch",
]

STATISTICAL_FEATURES = [
    "n_shots",
    "n_hits",
    "hit_rate",
    "first_k_hit_rate",
    "first_k_groggy_count",
    "first_k_final_elimination_count",
    "head_hit_rate",
    "torso_hit_rate",
    "pelvis_hit_rate",
    "arm_hit_rate",
    "leg_hit_rate",
    "shots_to_first_hit",
    "mean_distance_m",
    "max_distance_m",
    "mean_attacker_speed_mps",
    "mean_attacker_accel_mps2",
    "same_region_transition_rate",
    "hit_region_return_rate",
    "unique_hit_regions",
    "target_switch_count",
]

DAMAGE_REGION_TO_CODE = {
    "none": 0,
    "head": 1,
    "torso": 2,
    "pelvis": 3,
    "arm": 4,
    "leg": 5,
    "other": 6,
}

SCOPE_TO_CODE = {
    "none": 0,
    "1x": 1,
    "2x": 2,
    "3x": 3,
    "4x": 4,
    "6x": 5,
    "8x": 6,
    "other": 7,
}

WEAPON_CATEGORY_TO_CODE = {
    "AR": 0,
    "SMG": 1,
    "LMG": 2,
    "BURST_AR": 3,
    "DMR": 4,
    "PISTOL": 5,
    "SHOTGUN": 6,
    "SR": 7,
    "OTHER": 8,
}

RELATIVE_DIRECTION_TO_CODE = {
    "unknown": 0,
    "stationary": 1,
    "front": 2,
    "back": 3,
    "left": 4,
    "right": 5,
    "front_left": 6,
    "front_right": 7,
    "back_left": 8,
    "back_right": 9,
}

