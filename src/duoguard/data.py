"""Sequence input, external target joins, and training-only normalization."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BINARY_INDICES, CATEGORY_SIZES, SCALE_INDICES, SEQUENCE_FEATURES

META_COLUMNS = ['player_id', 'match_id', 'sequence_id', 'sequence_start_ms', 'test_window']


def validate_sequences(x, lengths, meta, raw=True):
    if x.ndim != 3 or x.shape[1:] != (32, 16) or len(x) == 0:
        raise ValueError('x_seq must have nonempty shape [N, 32, 16]')
    if lengths.shape != (len(x),) or not np.isfinite(lengths).all():
        raise ValueError('Invalid lengths')
    if not ((lengths >= 1) & (lengths <= 32) & (lengths == np.floor(lengths))).all():
        raise ValueError('Sequence lengths must be integers in [1,32]')
    if len(meta) != len(x) or any(c not in meta for c in META_COLUMNS):
        raise ValueError('Missing sequence metadata')
    for col in ['player_id', 'match_id', 'sequence_id']:
        if meta[col].isna().any() or meta[col].astype(str).str.strip().eq('').any():
            raise ValueError('Missing ' + col)
    if meta.duplicated(['player_id', 'test_window', 'sequence_id']).any():
        raise ValueError('Duplicate sequence IDs within a player/window')
    if not np.isfinite(meta['sequence_start_ms'].to_numpy(dtype=float)).all():
        raise ValueError('Sequence start times must be finite absolute or globally ordered times')
    valid = np.arange(32)[None, :] < lengths[:, None]
    if not np.isfinite(x).all() or np.any(x[~valid] != 0):
        raise ValueError('Features must be finite with zero padding')
    values = x[valid]
    for index, size in CATEGORY_SIZES.items():
        col = values[:, index]
        if not ((col >= 0) & (col < size) & (col == np.floor(col))).all():
            raise ValueError('Invalid categorical codes: ' + SEQUENCE_FEATURES[index])
    for index in BINARY_INDICES:
        if raw or index != 15:
            if not np.isin(values[:, index], [0, 1]).all():
                raise ValueError('Invalid binary feature: ' + SEQUENCE_FEATURES[index])


def save_sequences(path, x, lengths, meta):
    validate_sequences(x, lengths, meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_seq=np.asarray(x, np.float32), lengths=lengths.astype(np.int64),
                        feature_names=np.asarray(SEQUENCE_FEATURES, dtype=str),
                        **{c: meta[c].to_numpy(dtype=float if c == 'sequence_start_ms' else str)
                           for c in META_COLUMNS})


def load_sequences(path):
    with np.load(path, allow_pickle=False) as data:
        if data['feature_names'].tolist() != SEQUENCE_FEATURES:
            raise ValueError('Feature order differs from SEQUENCE_FEATURES')
        x = data['x_seq'].astype(np.float32)
        lengths = data['lengths']
        meta = pd.DataFrame({c: data[c] for c in META_COLUMNS})
    validate_sequences(x, lengths, meta)
    return x, lengths.astype(np.int64), meta


def attach_external_labels(meta, labels):
    labels = labels.copy()
    if 'player_id' not in labels and 'account_id' in labels:
        labels = labels.rename(columns={'account_id': 'player_id'})
    if 'label' not in labels and 'target' in labels:
        labels = labels.rename(columns={'target': 'label'})
    if any(c not in labels for c in ['player_id', 'label', 'split']):
        raise ValueError('External labels require player_id, label, split')
    labels['player_id'] = labels['player_id'].astype(str)
    labels['split'] = labels['split'].replace({'validation': 'valid'})
    labels['label'] = pd.to_numeric(labels['label'], errors='raise')
    if not labels['label'].isin([0, 1]).all() or not labels['split'].isin(['train', 'valid', 'test']).all():
        raise ValueError('Invalid external label or split')
    keys = ['player_id']
    if 'test_window' in labels:
        labels['test_window'] = labels['test_window'].fillna('').astype(str)
        keys.append('test_window')
    if labels.duplicated(keys).any():
        raise ValueError('Duplicate external labels for the join key')
    joined = meta.merge(labels[keys + ['label', 'split']], on=keys, how='left',
                        sort=False, validate='many_to_one')
    if joined['label'].isna().any():
        raise ValueError('Every sequence must have an externally supplied label')
    if joined.groupby('player_id')['split'].nunique().gt(1).any():
        raise ValueError('Accounts overlap between training, validation, and test')
    if joined.loc[joined['split'] != 'test', 'test_window'].ne('').any():
        raise ValueError('Development sequences use an empty test_window')
    if joined.loc[joined['split'] == 'test', 'test_window'].eq('').any():
        raise ValueError('Test sequences require a test_window identifier')
    if joined.groupby(['player_id', 'test_window'])['label'].nunique().gt(1).any():
        raise ValueError('Conflicting targets within a player bag')
    for role in ['train', 'valid']:
        if joined.loc[joined['split'] == role, 'label'].nunique() != 2:
            raise ValueError(role + ' must contain both classes')
    development = joined[joined['split'] != 'test']
    test = joined[joined['split'] == 'test']
    if len(test) and development['sequence_start_ms'].max() >= test['sequence_start_ms'].min():
        raise ValueError('Test sequences must start after the development observation period')
    return joined


class FeatureScaler:
    def fit(self, x, lengths):
        sums = np.zeros(len(SCALE_INDICES), np.float64)
        squares = np.zeros_like(sums)
        count = 0
        for start in range(0, len(x), 4096):
            block = x[start:start + 4096]
            mask = np.arange(32)[None, :] < lengths[start:start + 4096, None]
            values = block[mask][:, SCALE_INDICES].astype(np.float64)
            sums += values.sum(0)
            squares += np.square(values).sum(0)
            count += len(values)
        if count == 0:
            raise ValueError('Cannot fit scaler on an empty training set')
        self.mean = (sums / count).astype(np.float32)
        self.std = (np.sqrt(np.maximum(squares / count - (sums / count) ** 2, 0)) + 1e-6).astype(np.float32)
        return self

    def transform(self, x, lengths):
        result = x.copy()
        result[:, :, SCALE_INDICES] = (result[:, :, SCALE_INDICES] - self.mean) / self.std
        mask = np.arange(32)[None, :] < lengths[:, None]
        result[~mask] = 0
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({'feature_names': SEQUENCE_FEATURES,
                                        'indices': SCALE_INDICES, 'mean': self.mean.tolist(),
                                        'std': self.std.tolist()}, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text())
        if saved['feature_names'] != SEQUENCE_FEATURES or saved['indices'] != SCALE_INDICES:
            raise ValueError('Scaler schema mismatch')
        obj = cls()
        obj.mean = np.asarray(saved['mean'], np.float32)
        obj.std = np.asarray(saved['std'], np.float32)
        if obj.mean.shape != (5,) or obj.std.shape != (5,) or not np.isfinite(obj.mean).all() or not np.isfinite(obj.std).all() or (obj.std <= 0).any():
            raise ValueError('Invalid scaler')
        return obj
