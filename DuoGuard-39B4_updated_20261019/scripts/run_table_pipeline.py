"""Run the main two-stage model with externally provided labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from duoguard.data import load_sequences
from duoguard.preprocessing import build_sequence_dataset
from duoguard.training import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--events', type=Path, help='CSV: canonical shot features or legacy synthetic shots')
    inputs.add_argument('--sequences', type=Path, help='NPZ: raw canonical features and sequence metadata')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/duoguard_default.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--encoder-epochs', type=int)
    parser.add_argument('--player-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--player-batch-size', type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads(args.config.read_text())
    for key in ['encoder_epochs', 'player_epochs', 'batch_size', 'player_batch_size']:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    labels = pd.read_csv(args.labels, dtype={'player_id': str, 'account_id': str}, keep_default_na=False)
    if args.sequences:
        x, lengths, meta = load_sequences(args.sequences)
    else:
        events = pd.read_csv(args.events, dtype={'player_id': str, 'match_id': str})
        x, lengths, meta = build_sequence_dataset(events, labels)
    report = fit(x, lengths, meta, labels, cfg, args.output_dir, args.device)
    print(json.dumps({'status': 'complete', 'parameters': report['parameters'],
                      'threshold': report['threshold']}, indent=2))


if __name__ == '__main__':
    main()
