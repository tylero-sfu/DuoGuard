"""CPU smoke test: data -> two-stage training -> saved-model inference."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from duoguard.data import save_sequences
from duoguard.preprocessing import build_sequence_dataset
from duoguard.training import fit, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/smoke')
    parser.add_argument('--epochs', type=int, default=1, help='Smoke-only override; manuscript default is 5')
    args = parser.parse_args()
    torch.set_num_threads(1)
    # Run the existing generator in a temporary directory, without changing inputs.
    generator = runpy.run_path(str(ROOT / 'examples/generate_synthetic_data.py'))
    with tempfile.TemporaryDirectory() as temp:
        generator['main'].__globals__['ROOT'] = Path(temp)
        generator['main']()
        events = pd.read_csv(Path(temp) / 'synthetic_events.csv')
        labels = pd.read_csv(Path(temp) / 'synthetic_player_labels.csv')
    roles = labels.set_index('player_id')['split']
    events['test_window'] = ''
    test = events['player_id'].map(roles).eq('test')
    events.loc[test, 'test_window'] = events.loc[test, 'match_id'].str[1:].astype(int).mod(6).map(lambda i: f'week_{i + 1}')
    x, lengths, meta = build_sequence_dataset(events, labels)
    cfg = json.loads((ROOT / 'configs/duoguard_default.json').read_text())
    cfg.update(encoder_epochs=args.epochs, player_epochs=args.epochs, batch_size=32,
               player_batch_size=4, recurrent_chunk=8)
    save_sequences(args.output_dir / 'synthetic_sequences.npz', x, lengths, meta)
    report = fit(x, lengths, meta, labels, cfg, args.output_dir)
    prediction = predict(x, lengths, meta, args.output_dir, batch_size=32, player_batch_size=4)
    test_saved = pd.read_csv(args.output_dir / 'test_predictions.csv', keep_default_na=False)
    merged = test_saved.merge(prediction, on=['account_id', 'test_window'], suffixes=('_saved', '_reloaded'), validate='one_to_one')
    assert len(merged) == len(test_saved)
    np.testing.assert_allclose(merged['score_saved'], merged['score_reloaded'], rtol=1e-5, atol=1e-6)
    assert report['parameters'] == {'stage1': 33609, 'stage2': 40002, 'inference': 69386}
    assert len(report['test']['weekly']) == 6
    assert report['encoder_frozen_verified']
    for stage, key in [('sequence', 'encoder_best_epoch'), ('player', 'player_best_epoch')]:
        rows = [r for r in report['history'] if r['stage'] == stage]
        assert max(rows, key=lambda r: r['validation_auprc'])['epoch'] == report[key]
    prediction.to_csv(args.output_dir / 'reloaded_predictions.csv', index=False)
    result = {'status': 'PASS', 'synthetic_only': True, 'sequences': len(x),
              'test_windows': 6, 'parameters': report['parameters'],
              'frozen_encoder': True, 'checkpoint_reload_scores_match': True}
    (args.output_dir / 'smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
