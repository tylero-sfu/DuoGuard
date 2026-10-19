"""Summarize fixed predictions and optionally compute account-cluster CIs."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from duoguard.evaluation import account_cluster_bootstrap, weekly_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, dtype={'account_id': str, 'test_window': str})
    report = weekly_report(frame, args.threshold)
    if args.bootstrap:
        report['bootstrap'] = account_cluster_bootstrap(frame, args.threshold, args.bootstrap, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
