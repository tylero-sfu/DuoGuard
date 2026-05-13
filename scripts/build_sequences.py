from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoguard.preprocessing import DEFAULT_WEAPON_GAP_CUTOFFS_MS, SequenceConfig, build_sequence_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DuoGuard shooting sequences from a generic combat-event CSV file.")
    parser.add_argument("--events", required=True, help="Input generic combat-event CSV.")
    parser.add_argument("--out", required=True, help="Output pickle file containing sequence rows with tensor columns.")
    parser.add_argument("--max-sequence-length", type=int, default=32)
    parser.add_argument("--first-k", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events)
    config = SequenceConfig(
        max_sequence_length=args.max_sequence_length,
        first_k=args.first_k,
        weapon_gap_cutoffs_ms=DEFAULT_WEAPON_GAP_CUTOFFS_MS,
    )
    sequences = build_sequence_dataset(events, config=config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sequences.to_pickle(out)
    print(f"wrote {len(sequences)} sequences to {out}")


if __name__ == "__main__":
    main()

