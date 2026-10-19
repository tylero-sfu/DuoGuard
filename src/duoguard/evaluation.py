"""Fixed-model evaluation. Resampling never retrains or selects thresholds."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def binary_metrics(y, scores, threshold, sample_weight=None):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) == 0 or scores.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid metric inputs')
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any() or not weights.sum():
        raise ValueError('Invalid sample weights')
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1], sample_weight=weights).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {
        'auprc': float(average_precision_score(y, scores, sample_weight=weights)) if tp + fn else None,
        'auroc': float(roc_auc_score(y, scores, sample_weight=weights)) if min(tp + fn, tn + fp) > 0 else None,
        'precision': float(precision), 'recall': float(recall),
        'f1': float(2 * precision * recall / (precision + recall)) if precision + recall else 0.,
        'fpr': float(fp / (fp + tn)) if fp + tn else None,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn), 'n': int(weights.sum()),
    }


def select_threshold_by_f1(y, scores):
    candidates = np.arange(1, 100) / 100.
    values = [binary_metrics(y, scores, t)['f1'] for t in candidates]
    index = int(np.argmax(values))
    return float(candidates[index])


def validate_predictions(frame):
    required = ['account_id', 'test_window', 'target', 'score', 'n_seq']
    if any(c not in frame for c in required) or frame.empty:
        raise ValueError('Predictions require ' + ', '.join(required))
    if frame[required].isna().any().any() or frame.duplicated(['account_id', 'test_window']).any():
        raise ValueError('Missing or duplicate account/window predictions')
    if not frame['target'].isin([0, 1]).all() or not frame['score'].between(0, 1).all():
        raise ValueError('Invalid targets or scores')


def weekly_report(frame, threshold):
    validate_predictions(frame)
    weekly = {str(w): binary_metrics(p['target'], p['score'], threshold)
              for w, p in frame.groupby('test_window', sort=True)}
    names = ['auprc', 'auroc', 'precision', 'recall', 'f1', 'fpr']
    mean = {k: float(np.mean([v[k] for v in weekly.values()]))
            if all(v[k] is not None for v in weekly.values()) else None for k in names}
    return {'threshold': float(threshold), 'weekly': weekly, 'equal_weight_mean': mean,
            'pooled': binary_metrics(frame['target'], frame['score'], threshold)}


def account_cluster_bootstrap(frame, threshold, repeats=1000, seed=42):
    validate_predictions(frame)
    if repeats < 2:
        raise ValueError('At least two bootstrap resamples required')
    account_codes, accounts = pd.factorize(frame['account_id'], sort=True)
    grouped = list(frame.groupby('test_window', sort=True).indices.values())
    rng = np.random.default_rng(seed)
    values = []
    skipped = 0
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(accounts), size=len(accounts)), minlength=len(accounts))
        row_weights = counts[account_codes]
        metrics = []
        for idx in grouped:
            part = frame.iloc[idx]
            weight = row_weights[idx]
            if not weight.sum() or not weight[part['target'].to_numpy() == 1].sum():
                break
            score = binary_metrics(part['target'], part['score'], threshold, weight)
            metrics.append([score['auprc'], score['f1']])
        if len(metrics) == len(grouped):
            values.append(np.mean(metrics, axis=0))
        else:
            skipped += 1
    if len(values) < 2:
        raise ValueError('Too few valid bootstrap resamples; cohort is too small')
    interval = np.percentile(values, [2.5, 97.5], axis=0)
    return {'estimand': 'equal-weight mean across test windows', 'resamples_requested': repeats,
            'resamples_used': len(values), 'resamples_skipped': skipped,
            'auprc_95_ci': interval[:, 0].tolist(), 'f1_95_ci': interval[:, 1].tolist()}
