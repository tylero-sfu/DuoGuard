import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from duoguard.data import FeatureScaler, attach_external_labels, load_sequences, save_sequences
from duoguard.evaluation import account_cluster_bootstrap, binary_metrics, select_threshold_by_f1, weekly_report
from duoguard.models import SequenceClassifier, SequenceCore, TemporalAttention
from duoguard.preprocessing import build_sequence_dataset, interpolate_missing_distances
from duoguard.training import PlayerBagDataset

torch.set_num_threads(1)


def sample():
    x = np.zeros((4, 32, 16), np.float32)
    lengths = np.array([2, 3, 2, 1])
    meta = pd.DataFrame(dict(player_id=['a', 'b', 'c', 'd'], match_id=['m1', 'm2', 'm3', 'm4'],
                             sequence_id=['s1', 's2', 's3', 's4'], sequence_start_ms=[1., 2., 3., 4.],
                             test_window=[''] * 4))
    labels = pd.DataFrame(dict(player_id=['a', 'b', 'c', 'd'], label=[0, 1, 0, 1],
                               split=['train', 'train', 'valid', 'valid']))
    return x, lengths, meta, labels


class CoreTests(unittest.TestCase):
    def test_parameter_counts(self):
        encoder, player = SequenceClassifier(), TemporalAttention()
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 33609)
        self.assertEqual(sum(p.numel() for p in player.parameters()), 40002)
        self.assertEqual(sum(p.numel() for p in encoder.encoder.parameters()), 29384)

    def test_encoder_last_valid_not_padded(self):
        torch.manual_seed(5)
        model = SequenceCore().eval()
        x = torch.zeros(2, 32, 16)
        lengths = torch.tensor([3, 7])
        with torch.no_grad():
            batched = model(x, lengths)
            for i, length in enumerate(lengths):
                single = model(x[i:i+1, :length], length[None])
                torch.testing.assert_close(batched[i:i+1], single, atol=1e-6, rtol=1e-5)

    def test_chunked_attention_equals_full(self):
        torch.manual_seed(6)
        model = TemporalAttention(2).eval()
        x = torch.randn(3, 9, 64)
        mask = torch.arange(9)[None, :] < torch.tensor([1, 4, 9])[:, None]
        flags = torch.zeros_like(mask)
        flags[:, 0] = True
        with torch.no_grad():
            actual = model.forward_features(x, flags, mask)
            out, _ = model.lstm(torch.cat([x, flags.unsqueeze(-1).float()], -1))
            alpha = model.attention(out).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
            expected = (alpha.unsqueeze(-1) * out).sum(1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_chunk_backward_finite(self):
        model = TemporalAttention(2).train()
        x = torch.randn(2, 7, 64)
        mask = torch.arange(7)[None, :] < torch.tensor([1, 7])[:, None]
        flags = torch.zeros_like(mask)
        flags[:, 0] = True
        model(x, flags, mask).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_empty_bag_rejected(self):
        with self.assertRaises(ValueError):
            TemporalAttention()(torch.zeros(1, 2, 64), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

    def test_archive_roundtrip(self):
        x, lengths, meta, _ = sample()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.npz'
            save_sequences(path, x, lengths, meta)
            a, b, c = load_sequences(path)
            np.testing.assert_array_equal(a, x)
            np.testing.assert_array_equal(b, lengths)
            pd.testing.assert_frame_equal(c, meta)

    def test_bad_category_rejected(self):
        x, lengths, meta, _ = sample()
        x[0, 0, 1] = 100
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            save_sequences(Path(temp) / 'bad.npz', x, lengths, meta)

    def test_duplicate_targets_rejected(self):
        _, _, meta, labels = sample()
        labels = pd.concat([labels, labels.iloc[[0]]])
        with self.assertRaises(ValueError):
            attach_external_labels(meta, labels)

    def test_split_overlap_rejected(self):
        _, _, meta, labels = sample()
        extra = labels.iloc[[0]].copy()
        extra['split'] = 'valid'
        with self.assertRaises(ValueError):
            attach_external_labels(meta, pd.concat([labels, extra]))

    def test_missing_labels_rejected(self):
        _, _, meta, labels = sample()
        with self.assertRaises(ValueError):
            attach_external_labels(meta, labels.iloc[1:])

    def test_later_period_constraint(self):
        _, _, meta, labels = sample()
        row = meta.iloc[[0]].copy()
        row['player_id'] = 'test_player'
        row['test_window'] = 'week1'
        labels = pd.concat([labels, pd.DataFrame([dict(player_id='test_player', split='test', label=0)])])
        with self.assertRaises(ValueError):
            attach_external_labels(pd.concat([meta, row]), labels)

    def test_scaler_training_only_padding(self):
        x, lengths, _, _ = sample()
        x[0, :2, 0] = [1, 3]
        x[1, :3, 0] = [2, 4, 5]
        x[2, :2, 0] = 999
        scaler = FeatureScaler().fit(x[:2], lengths[:2])
        self.assertAlmostEqual(scaler.mean[0], 3.)
        transformed = scaler.transform(x, lengths)
        self.assertTrue(np.all(transformed[2, 2:] == 0))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'scaler.json'
            scaler.save(path)
            np.testing.assert_array_equal(FeatureScaler.load(path).transform(x, lengths), transformed)

    def test_miss_only_retained(self):
        events = pd.DataFrame(dict(player_id=['a'] * 2, match_id=['m'] * 2,
                                   event_time_ms=[0, 100], event_type=['shot'] * 2,
                                   weapon_category=['AR'] * 2))
        labels = pd.DataFrame([dict(player_id='a', label=0, split='train')])
        x, lengths, _ = build_sequence_dataset(events, labels)
        np.testing.assert_array_equal(lengths, [2])
        self.assertTrue(np.all(x[:, :, 2] == 0))

    def test_time_based_interpolation(self):
        part = pd.DataFrame(dict(target_id=['a'] * 3, event_time_ms=[0, 10, 100], distance_m=[10., np.nan, 110.]))
        result = interpolate_missing_distances(part)
        self.assertEqual(result.iloc[1]['distance_m'], 20.)
        self.assertEqual(result.iloc[1]['distance_valid'], 0.)

    def test_no_interpolation_across_switch(self):
        part = pd.DataFrame(dict(target_id=['a', 'a', 'b', 'a'], event_time_ms=[0, 1, 2, 3],
                                 distance_m=[10., np.nan, 30., 100.]))
        self.assertTrue(pd.isna(interpolate_missing_distances(part).iloc[1]['distance_m']))

    def test_bags_sort_and_do_not_merge_windows(self):
        meta = pd.DataFrame(dict(player_id=['a'] * 4, test_window=['w1', 'w1', 'w1', 'w2'],
                                 match_id=['m2', 'm1', 'm1', 'm3'], sequence_id=['s2', 's1', 's0', 's3'],
                                 sequence_start_ms=[3, 2, 1, 4], label=[1] * 4))
        z = np.repeat(np.arange(4)[:, None], 64, axis=1).astype(np.float32)
        bags = PlayerBagDataset(z, meta)
        self.assertEqual(len(bags), 2)
        np.testing.assert_array_equal(bags[0][2][:, 0], [2, 1, 0])
        np.testing.assert_array_equal(bags[0][3], [True, False, True])

    def test_threshold_grid(self):
        self.assertEqual(select_threshold_by_f1([0, 1], [.01, .99]), .02)

    def test_bootstrap_clusters_across_windows(self):
        rows = []
        for window in ['w1', 'w2']:
            for i in range(20):
                rows.append(dict(account_id=str(i), test_window=window, target=i % 2,
                                 score=(i % 2) * .5 + .2, n_seq=3))
        frame = pd.DataFrame(rows)
        report = weekly_report(frame, .5)
        self.assertEqual(report['equal_weight_mean']['f1'], 1.)
        ci = account_cluster_bootstrap(frame, .5, repeats=40, seed=1)
        self.assertEqual(ci['f1_95_ci'], [1., 1.])
        self.assertEqual(ci['auprc_95_ci'], [1., 1.])

    def test_bootstrap_matches_explicit_account_resampling(self):
        rng = np.random.default_rng(8)
        rows = [dict(account_id=f'a{i:02d}', test_window=f'w{w}', target=i % 2,
                     score=float(rng.uniform()), n_seq=i + 1)
                for w in range(3) for i in range(20)]
        frame = pd.DataFrame(rows)
        actual = account_cluster_bootstrap(frame, .5, 30, 9)
        ids = sorted(frame['account_id'].unique())
        rng = np.random.default_rng(9)
        samples = []
        for _ in range(30):
            selected = rng.integers(0, len(ids), size=len(ids))
            resampled = pd.concat([frame[frame['account_id'] == ids[i]] for i in selected])
            metrics = [binary_metrics(g['target'], g['score'], .5)
                       for _, g in resampled.groupby('test_window')]
            if any(m['auprc'] is None for m in metrics):
                continue
            samples.append([np.mean([m[k] for m in metrics]) for k in ['auprc', 'f1']])
        expected = np.percentile(samples, [2.5, 97.5], axis=0)
        np.testing.assert_allclose(actual['auprc_95_ci'], expected[:, 0])
        np.testing.assert_allclose(actual['f1_95_ci'], expected[:, 1])


if __name__ == '__main__':
    unittest.main()
