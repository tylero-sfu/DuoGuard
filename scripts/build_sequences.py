"""Convert supplied shot features to the portable sequence archive."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from duoguard.data import attach_external_labels, save_sequences
from duoguard.preprocessing import build_sequence_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = pd.read_csv(args.events, dtype={'player_id': str, 'match_id': str})
    labels = pd.read_csv(args.labels, dtype={'player_id': str}, keep_default_na=False)
    x, lengths, meta = build_sequence_dataset(events, labels)
    attach_external_labels(meta, labels)
    save_sequences(args.output, x, lengths, meta)
    print(f'Wrote {len(x)} sequences; labels remain external')


if __name__ == '__main__':
    main()
