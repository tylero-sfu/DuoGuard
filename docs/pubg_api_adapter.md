# PUBG API-style telemetry adapter

The public implementation does not download PUBG telemetry. Users who have
access to public PUBG telemetry exports can convert combat events into the
generic DuoGuard event schema with `duoguard.pubg_api_adapter`.

The adapter is limited to combat telemetry observable from match logs. It does
not include enforcement records, sanction histories, ban labels, or
player-label construction.

## Event types

The adapter recognizes the following PUBG API-style combat events:

```text
LogPlayerAttack       -> shot
LogPlayerTakeDamage   -> hit
LogPlayerMakeGroggy   -> groggy
LogPlayerKill         -> final_elimination
LogPlayerKillV2       -> final_elimination
```

The converted table has the generic schema used by the rest of the package:

```text
match_id, player_id, target_id, event_time_ms, event_type, attack_id,
weapon_id, weapon_category, scope, damage_region, distance_m,
attacker_x, attacker_y, target_x, target_y
```

## Example

```python
from duoguard.pubg_api_adapter import load_pubg_json_events, pubg_events_to_generic_table
from duoguard.preprocessing import build_sequence_dataset

events = load_pubg_json_events("match_telemetry.json")
generic = pubg_events_to_generic_table(events, match_id="match_001")
sequences = build_sequence_dataset(generic)
```

Weapon-category mapping can be supplied explicitly if a study needs a different
disclosure policy or a title-specific weapon taxonomy.

