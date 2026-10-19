"""Reload a locally trained model and score sequences without label inputs."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from duoguard.data import load_sequences
from duoguard.training import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    x, lengths, meta = load_sequences(args.sequences)
    result = predict(x, lengths, meta, args.model_dir, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Wrote {len(result)} player-window scores')


if __name__ == '__main__':
    main()
