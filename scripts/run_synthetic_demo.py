from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoguard.features import SEQUENCE_FEATURES, STATISTICAL_FEATURES
from duoguard.preprocessing import DEFAULT_WEAPON_GAP_CUTOFFS_MS, SequenceConfig, build_sequence_dataset
from duoguard.training import (
    TrainingConfig,
    extract_sequence_embeddings,
    score_player_embeddings,
    train_player_classifier,
    train_sequence_encoder,
)
from duoguard.evaluation import binary_metrics


def assign_weapon_indices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    vocab = {w: i for i, w in enumerate(sorted(frame["weapon_id"].astype(str).unique()))}
    out = frame.copy()
    out["weapon_idx"] = out["weapon_id"].astype(str).map(vocab).astype(int)
    return out, vocab


def main() -> None:
    np.random.seed(7)
    torch.manual_seed(7)

    events = pd.read_csv(ROOT / "examples" / "synthetic_events.csv")
    labels = pd.read_csv(ROOT / "examples" / "synthetic_player_labels.csv")

    seq_cfg = SequenceConfig(
        max_sequence_length=32,
        first_k=7,
        weapon_gap_cutoffs_ms=DEFAULT_WEAPON_GAP_CUTOFFS_MS,
    )
    sequences = build_sequence_dataset(events, config=seq_cfg)
    sequences = sequences.merge(labels, on="player_id", how="inner")
    sequences, weapon_vocab = assign_weapon_indices(sequences)

    # The synthetic file is tiny. Duplicate rows only to exercise the training
    # loop; real experiments should use the original de-identified sequence rows.
    sequences = pd.concat([sequences] * 32, ignore_index=True)
    train_df = sequences[sequences["split"] == "train"].reset_index(drop=True)
    valid_df = sequences[sequences["split"] == "valid"].reset_index(drop=True)

    cfg = TrainingConfig(
        encoder_epochs=1,
        player_epochs=1,
        batch_size=16,
        player_batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    encoder = train_sequence_encoder(
        train_df=train_df,
        valid_df=valid_df,
        sequence_features=SEQUENCE_FEATURES,
        d_stat=len(STATISTICAL_FEATURES),
        n_weapons=max(1, len(weapon_vocab)),
        cfg=cfg,
    )
    train_emb = extract_sequence_embeddings(encoder, train_df, SEQUENCE_FEATURES, batch_size=32)
    valid_emb = extract_sequence_embeddings(encoder, valid_df, SEQUENCE_FEATURES, batch_size=32)
    player_model, threshold, valid_metrics = train_player_classifier(train_emb, valid_emb, cfg=cfg, pooling="mean")
    y, scores = score_player_embeddings(player_model, valid_emb, cfg=cfg)
    metrics = binary_metrics(y, scores, threshold)

    print("weapon_vocab:", weapon_vocab)
    print("n_sequences:", len(sequences))
    print("validation_threshold:", round(threshold, 3))
    print("validation_metrics:", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

