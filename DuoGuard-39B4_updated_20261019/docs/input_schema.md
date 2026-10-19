# Input contract

## Portable sequence archive

Use `duoguard.data.save_sequences(path, x, lengths, metadata)` to create the NPZ.
`x` has shape `[N, 32, 16]`, dtype float32, and zero padding. `lengths` contains
valid shot counts from 1 to 32. Features are not standardized yet.
The feature names, in exact order, are:

```
delta_time_ms
scope_code
is_hit
is_groggy_shot
is_flush_shot
is_direct_kill_shot
distance_log
distance_bin_code
distance_valid
attacker_speed_mps_filled
attacker_accel_mps2_filled
rel_dir_code_filled
rel_dir_valid
damage_part_code
weapon_category
victim_switch_event
```

Categorical ranges: scope 0..7, damage part 0..6, relative direction 0..9,
distance bin 0..5, weapon category 0..6. Flags are 0/1. Preserve the encoding
conventions of the supplied preprocessed data. Weapon category order is AR,
SMG, LMG, DMR, SR, shotgun, pistol. A valid-feature table must not substitute
human-readable weapon strings for the canonical numeric `weapon_category`.

The model embeds scope, damage part, and direction into eight dimensions each,
one-hot encodes distance bins and weapon categories, and concatenates the other
11 inputs, producing 48 event dimensions. The distance-valid indicator masks
the distance-bin one-hot vector. It records original observation availability,
not whether interpolation later filled a value.

The scaler fits only valid training shots, using population standard deviations
plus 1e-6. Columns 0, 6, 9, 10, and 15 are standardized. All padding is restored
to zero after transformation. Do not pass already standardized features to this
runner, or they will be standardized twice.

Metadata has one row per sequence:

| Column | Meaning |
| --- | --- |
| player_id | De-identified account identifier |
| match_id | Match identifier used to construct match-start flags |
| sequence_id | Unique within an account and test window |
| sequence_start_ms | Numeric time comparable across matches |
| test_window | Empty for development; nonempty identifier for each test period |

Chronological order is `sequence_start_ms`, then match and sequence IDs for ties.
Do not use elapsed match time without a globally ordered match-start offset.
Each player's first sequence and each change of match receive a match-start flag.
No sequence subsampling or player-history truncation is performed.

## External labels and splits

CSV fields: `player_id,label,split`, with labels 0/1 and splits `train`, `valid`,
or `test`. `account_id`/`target` are accepted aliases. With one label per account,
it applies to each supplied test observation. For externally supplied labels
that vary by test period, add a `test_window` column and one row per account/window;
use an empty string for development. Do not duplicate keys.

The package joins these values but never creates, infers, or revises labels.
Targets are used only for loss and evaluation, not as model input features.
Both training and validation must contain positive and negative accounts.
The caller supplies the approved data partitions. The code checks account
separation and the development/test ordering, not an undisclosed label rule.

## Canonical shot-feature CSV

Provide the 16 columns above plus `player_id,match_id,event_time_ms`. Optional
`sequence_id` preserves externally prepared sequence groups; optional `test_window`
separates test periods. Without `sequence_id`, the adapter segments by time gap
and weapon changes using the reported fixed cutoffs. Without `test_window`, it
uses one test period called `test` for backward compatibility.

Provided sequences retain their first 32 shots. This is the sequence-length
limit, not truncation of a player's sequence history. Membership requires no hit.

## Legacy synthetic CSV

The previous example schema is retained for demonstration: one row per shot,
where `event_type` is `shot` (miss), `hit`, `groggy`, `flush`, or
`final_elimination` (direct kill). Its human-readable scope and damage-region
fields use toy encoding dictionaries in the adapter. Motion inputs absent from
the example remain zero; arbitrary coordinates are not interpreted as movement.
Example distance bins are defined in the toy adapter. Use canonical feature
columns, not these toy defaults, for existing research preprocessing.

Do not concatenate a raw attack row and its damage row and pass both as shots.
Source-specific attack/outcome integration is upstream of this interface.

## Prediction output

Evaluation CSV columns are `account_id,test_window,target,score,n_seq`. Each row
is one player/window. Account identifiers must remain consistent across windows
for cluster bootstrap. Inference does not need labels and instead writes
`account_id,test_window,score,n_seq,prediction`.

Metrics, including FPR, use the 0..1 scale. Multiply by 100 for percentage
presentation. AUPRC uses average precision. Weekly metrics receive equal weight
regardless of cohort size, and pooled metrics are reported separately.
