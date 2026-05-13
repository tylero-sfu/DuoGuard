from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoguard.evaluation import binary_metrics
from duoguard.features import SEQUENCE_FEATURES, STATISTICAL_FEATURES
from duoguard.preprocessing import (
    DEFAULT_WEAPON_GAP_CUTOFFS_MS,
    SequenceConfig,
    build_sequence_dataset,
)
from duoguard.training import (
    TrainingConfig,
    extract_sequence_embeddings,
    score_player_embeddings,
    train_player_classifier,
    train_sequence_encoder,
)


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assign_weapon_indices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    vocab = {w: i for i, w in enumerate(sorted(frame["weapon_id"].astype(str).unique()))}
    out = frame.copy()
    out["weapon_idx"] = out["weapon_id"].astype(str).map(vocab).astype(int)
    return out, vocab


def json_safe_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a table-driven DuoGuard pipeline.")
    parser.add_argument("--events", type=Path, required=True, help="De-identified combat-event CSV.")
    parser.add_argument("--labels", type=Path, required=True, help="Player-level label CSV with player_id,label,split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "duoguard_default.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "table_pipeline")
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--player-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--player-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = load_config(args.config)
    seq_cfg = SequenceConfig(
        max_sequence_length=int(cfg_dict.get("max_sequence_length", 32)),
        first_k=int(cfg_dict.get("first_k", 7)),
        weapon_gap_cutoffs_ms=cfg_dict.get("weapon_gap_cutoffs_ms", DEFAULT_WEAPON_GAP_CUTOFFS_MS),
    )

    events = pd.read_csv(args.events)
    labels = pd.read_csv(args.labels)
    sequences = build_sequence_dataset(events, config=seq_cfg)
    sequences = sequences.merge(labels, on="player_id", how="inner")
    sequences, weapon_vocab = assign_weapon_indices(sequences)

    train_df = sequences[sequences["split"] == "train"].reset_index(drop=True)
    valid_df = sequences[sequences["split"] == "valid"].reset_index(drop=True)
    test_df = sequences[sequences["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        test_df = valid_df.copy()

    train_cfg = TrainingConfig(
        encoder_epochs=int(args.encoder_epochs or cfg_dict.get("encoder_epochs", 3)),
        player_epochs=int(args.player_epochs or cfg_dict.get("player_epochs", 3)),
        encoder_lr=float(cfg_dict.get("encoder_learning_rate", 5e-4)),
        player_lr=float(cfg_dict.get("player_learning_rate", 1e-3)),
        batch_size=int(args.batch_size),
        player_batch_size=int(args.player_batch_size),
        device=args.device,
    )

    encoder = train_sequence_encoder(
        train_df=train_df,
        valid_df=valid_df,
        sequence_features=SEQUENCE_FEATURES,
        d_stat=len(STATISTICAL_FEATURES),
        n_weapons=max(1, len(weapon_vocab)),
        cfg=train_cfg,
    )
    train_emb = extract_sequence_embeddings(encoder, train_df, SEQUENCE_FEATURES)
    valid_emb = extract_sequence_embeddings(encoder, valid_df, SEQUENCE_FEATURES)
    test_emb = extract_sequence_embeddings(encoder, test_df, SEQUENCE_FEATURES)

    player_model, threshold, valid_metrics = train_player_classifier(
        train_embeddings=train_emb,
        valid_embeddings=valid_emb,
        cfg=train_cfg,
        pooling=str(cfg_dict.get("player_pooling", "mean")),
    )
    y_test, s_test = score_player_embeddings(player_model, test_emb, cfg=train_cfg)
    test_metrics = binary_metrics(y_test, s_test, threshold)

    pred = pd.DataFrame({"label": y_test.astype(int), "score": s_test.astype(float)})
    pred["prediction"] = (pred["score"] >= float(threshold)).astype(int)
    pred.to_csv(args.output_dir / "test_predictions.csv", index=False)
    train_emb.to_pickle(args.output_dir / "train_embeddings.pkl")
    valid_emb.to_pickle(args.output_dir / "valid_embeddings.pkl")
    test_emb.to_pickle(args.output_dir / "test_embeddings.pkl")

    summary = {
        "n_sequences": int(len(sequences)),
        "n_players": int(sequences["player_id"].nunique()),
        "weapon_vocab": weapon_vocab,
        "threshold": float(threshold),
        "valid_metrics": json_safe_metrics(valid_metrics),
        "test_metrics": json_safe_metrics(test_metrics),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
