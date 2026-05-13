# Combat-Log Preprocessing Pipeline

The public package exposes the combat-log preprocessing steps needed to
construct DuoGuard model inputs from de-identified event tables. It does not
include player-label construction from operational enforcement records.

## Input Scope

The preprocessing code expects combat telemetry that has already been
de-identified and mapped to generic columns:

```text
match_id, player_id, target_id, event_time_ms, event_type, attack_id,
weapon_id, weapon_category, scope, damage_region, distance_m,
attacker_x, attacker_y, target_x, target_y
```

The code supports the following event types:

```text
shot, hit, groggy, final_elimination, flush
```

## Steps

1. **Combat-log integration**
   - `integrate_combat_logs` combines separate attack, damage, groggy,
     final-elimination, and flush tables into a unified event stream.

2. **Outcome matching**
   - `match_outcomes` attaches coarse downstream outcome flags within a short
     temporal window for the same match, player, and target.

3. **Shooting-sequence segmentation**
   - `segment_shooting_sequences` groups temporally adjacent combat events
     from the same player and match.
   - Weapon-category-specific temporal cutoffs are provided in
     `DEFAULT_WEAPON_GAP_CUTOFFS_MS`.

4. **Distance interpolation**
   - `interpolate_missing_distances` reconstructs missing distances only
     between observed distances within the same sequence and target.

5. **Tensor and descriptor construction**
   - `build_sequence_tensor` creates a fixed-length sequence tensor and a
     padding mask.
   - `build_statistical_descriptor` creates the domain-informed statistical
     descriptor used by the dual-view encoder.

## Label Boundary

Player-level labels are loaded separately. The public release assumes an
external table with:

```text
player_id,label,split
```

The code intentionally excludes operational enforcement records and
player-label construction logic.

