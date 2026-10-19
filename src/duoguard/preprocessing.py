"""Adapter for the existing synthetic shot table and canonical feature tables.

Input rows represent shots with outcomes already attached, not separate raw
attack and damage records. Real precomputed features should use canonical names.
"""
import numpy as np
import pandas as pd

from .data import META_COLUMNS, validate_sequences
from .features import GAP_CUTOFFS_MS, SEQUENCE_FEATURES, WEAPON_CATEGORIES


def interpolate_missing_distances(sequence):
    sequence = sequence.sort_values('event_time_ms', kind='stable').copy()
    original = pd.to_numeric(sequence['distance_m'], errors='raise')
    sequence['distance_valid'] = original.notna().astype(float)
    sequence['distance_m'] = original
    targets = sequence['target_id'].fillna('').astype(str)
    blocks = targets.ne(targets.shift()).cumsum()
    for _, part in sequence.groupby(blocks, sort=False):
        if part['target_id'].fillna('').astype(str).eq('').all():
            continue
        known = part['distance_m'].notna()
        if known.sum() < 2:
            continue
        times = part['event_time_ms'].to_numpy(dtype=float)
        good_times = times[known]
        good_dist = part.loc[known, 'distance_m'].to_numpy(dtype=float)
        # Ambiguous simultaneous observations are not interpolation anchors.
        if len(np.unique(good_times)) != len(good_times):
            continue
        fill = ~known.to_numpy() & (times > good_times[0]) & (times < good_times[-1])
        sequence.loc[part.index[fill], 'distance_m'] = np.interp(times[fill], good_times, good_dist)
    return sequence


def _legacy_features(part):
    """Convert toy shot outcomes; not a proprietary raw telemetry parser."""
    part = interpolate_missing_distances(part)
    x = np.zeros((len(part), 16), np.float32)
    scope = {'none': 0, '1x': 1, '2x': 2, '3x': 3, '4x': 4, '6x': 5, '8x': 6, 'other': 7}
    region = {'none': 0, 'head': 1, 'torso': 2, 'pelvis': 3, 'arm': 4, 'leg': 5, 'other': 6}
    events = part['event_type']
    if not events.isin(['shot', 'hit', 'groggy', 'flush', 'final_elimination']).all():
        raise ValueError('Unsupported toy shot outcome')
    x[:, 0] = part['event_time_ms'].diff().fillna(0)
    x[:, 1] = part.get('scope', pd.Series('none', index=part.index)).map(scope).fillna(7)
    x[:, 2] = events.ne('shot')
    x[:, 3] = events.eq('groggy')
    x[:, 4] = events.eq('flush')
    x[:, 5] = events.eq('final_elimination')
    distance = part['distance_m'].fillna(0).to_numpy(dtype=float)
    if (distance < 0).any():
        raise ValueError('Negative distance')
    x[:, 6] = np.log1p(distance)
    x[:, 7] = np.digitize(distance, [10, 30, 50, 100, 200])
    x[:, 8] = part['distance_valid']
    # Toy legacy input lacks measured motion features. Do not infer motion from
    # a single attacker-target displacement; canonical input supplies these.
    for index, name in [(9, 'attacker_speed_mps_filled'), (10, 'attacker_accel_mps2_filled'),
                        (11, 'rel_dir_code_filled'), (12, 'rel_dir_valid')]:
        if name in part:
            x[:, index] = part[name].to_numpy(dtype=float)
    x[:, 13] = part.get('damage_region', pd.Series('none', index=part.index)).map(region).fillna(6)
    x[:, 14] = part['weapon_category'].str.upper().map(WEAPON_CATEGORIES)
    target = part['target_id'].fillna('').astype(str)
    x[:, 15] = (target.ne(target.shift()) & target.ne('') & target.shift().fillna('').ne('')).to_numpy()
    return x


def build_sequence_dataset(events, labels):
    events = events.copy()
    labels = labels.rename(columns={'account_id': 'player_id', 'target': 'label'}).copy()
    if 'player_id' not in events and 'account_id' in events:
        events = events.rename(columns={'account_id': 'player_id'})
    required = ['player_id', 'match_id', 'event_time_ms', 'weapon_category']
    if any(c not in events for c in required):
        raise ValueError('Shot table requires ' + ', '.join(required))
    if 'split' in events or 'label' in events or 'target' in events:
        raise ValueError('Provide labels and split assignments separately')
    for col in ['player_id', 'match_id']:
        if events[col].isna().any():
            raise ValueError('Missing ' + col)
        events[col] = events[col].astype(str)
    labels['player_id'] = labels['player_id'].astype(str)
    if labels.groupby('player_id')['split'].nunique().gt(1).any():
        raise ValueError('Overlapping split accounts')
    roles = labels.drop_duplicates('player_id').set_index('player_id')['split']
    events['_role'] = events['player_id'].map(roles)
    if events['_role'].isna().any():
        raise ValueError('Unknown player in external split assignments')
    canonical = all(c in events for c in SEQUENCE_FEATURES)
    if 'test_window' not in events:
        # Backward-compatible single test set; never substitute validation.
        events['test_window'] = np.where(events['_role'].eq('test'), 'test', '')
    events['test_window'] = events['test_window'].fillna('').astype(str)
    events['event_time_ms'] = pd.to_numeric(events['event_time_ms'], errors='raise')
    if not np.isfinite(events['event_time_ms']).all():
        raise ValueError('Invalid event times')
    if canonical:
        reverse = {v: k for k, v in WEAPON_CATEGORIES.items()}
        events['_category'] = events['weapon_category'].map(reverse)
    else:
        events['_category'] = events['weapon_category'].astype(str).str.upper()
    if not events['_category'].isin(GAP_CUTOFFS_MS).all():
        raise ValueError('Unsupported weapon category')
    if not canonical:
        if 'event_type' not in events:
            raise ValueError('Provide all canonical features or legacy toy shot outcomes')
        for key, default in [('target_id', ''), ('distance_m', np.nan)]:
            if key not in events:
                events[key] = default
    events = events.sort_values(['player_id', 'test_window', 'match_id', 'event_time_ms'], kind='stable')
    tensors, lengths, rows = [], [], []
    for (player, window, match), group in events.groupby(['player_id', 'test_window', 'match_id'], sort=False):
        if 'sequence_id' in group:
            if group['sequence_id'].isna().any():
                raise ValueError('Missing supplied sequence_id')
            groups = group.groupby('sequence_id', sort=False)
        else:
            gaps = group['event_time_ms'].diff()
            changed = group['_category'].ne(group['_category'].shift())
            if 'weapon_id' in group:
                changed |= group['weapon_id'].ne(group['weapon_id'].shift())
            boundary = gaps.isna() | changed | gaps.gt(group['_category'].map(GAP_CUTOFFS_MS))
            groups = group.groupby(boundary.cumsum(), sort=False)
        for _, part in groups:
            if part['_category'].nunique() != 1:
                raise ValueError('A sequence must have a consistent weapon category')
            part = part.iloc[:32]
            values = part[SEQUENCE_FEATURES].to_numpy(dtype=np.float32) if canonical else _legacy_features(part)
            padded = np.zeros((32, 16), np.float32)
            padded[:len(values)] = values
            tensors.append(padded)
            lengths.append(len(values))
            rows.append(dict(player_id=player, test_window=window, match_id=match,
                             sequence_id=f's{len(rows):09d}', sequence_start_ms=float(part['event_time_ms'].iloc[0])))
    if not tensors:
        raise ValueError('No shooting sequences')
    x = np.stack(tensors)
    lengths = np.asarray(lengths, np.int64)
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    validate_sequences(x, lengths, meta)
    return x, lengths, meta
