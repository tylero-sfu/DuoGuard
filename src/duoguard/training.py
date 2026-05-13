"""Compact training utilities for the sanitized DuoGuard implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import binary_metrics, select_threshold_by_f1
from .models import DuoGuardSequenceEncoder, PlayerRiskClassifier, default_categorical_indices


@dataclass
class TrainingConfig:
    encoder_epochs: int = 3
    player_epochs: int = 3
    encoder_lr: float = 5e-4
    player_lr: float = 1e-3
    batch_size: int = 256
    player_batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, sequence_features: list[str]):
        self.frame = frame.reset_index(drop=True)
        self.sequence_features = sequence_features

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        return {
            "x_seq": torch.tensor(row["x_seq"], dtype=torch.float32),
            "x_stat": torch.tensor(row["x_stat"], dtype=torch.float32),
            "mask": torch.tensor(row["mask"], dtype=torch.bool),
            "weapon_idx": torch.tensor(int(row.get("weapon_idx", 0)), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


class PlayerBagDataset(Dataset):
    def __init__(self, embeddings: pd.DataFrame):
        self.rows = []
        for player_id, part in embeddings.groupby("player_id", sort=False):
            z = np.stack(part["embedding"].to_numpy())
            label = float(part["label"].iloc[0])
            self.rows.append((player_id, z, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        player_id, z, label = self.rows[idx]
        return player_id, torch.tensor(z, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_player_bags(batch):
    player_ids, bags, labels = zip(*batch)
    max_len = max(b.shape[0] for b in bags)
    dim = bags[0].shape[1]
    x = torch.zeros((len(bags), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(bags), max_len), dtype=torch.bool)
    for i, bag in enumerate(bags):
        x[i, : bag.shape[0], :] = bag
        mask[i, : bag.shape[0]] = True
    return player_ids, x, mask, torch.stack(labels)


def train_sequence_encoder(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    sequence_features: list[str],
    d_stat: int,
    n_weapons: int,
    cfg: TrainingConfig | None = None,
) -> DuoGuardSequenceEncoder:
    cfg = cfg or TrainingConfig()
    device = torch.device(cfg.device)
    model = DuoGuardSequenceEncoder(
        d_seq=len(sequence_features),
        d_stat=d_stat,
        n_weapons=n_weapons,
        categorical_indices=default_categorical_indices(sequence_features),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.encoder_lr, weight_decay=1e-4)
    pos = max(1.0, float(train_df["label"].sum()))
    neg = max(1.0, float((1 - train_df["label"]).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    loader = DataLoader(SequenceDataset(train_df, sequence_features), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.encoder_epochs):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["x_seq"].to(device),
                batch["x_stat"].to(device),
                batch["weapon_idx"].to(device),
                batch["mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def extract_sequence_embeddings(model: DuoGuardSequenceEncoder, frame: pd.DataFrame, sequence_features: list[str], batch_size: int = 512) -> pd.DataFrame:
    device = next(model.parameters()).device
    loader = DataLoader(SequenceDataset(frame, sequence_features), batch_size=batch_size, shuffle=False)
    rows = []
    start = 0
    model.eval()
    for batch in loader:
        z = model.encode(
            batch["x_seq"].to(device),
            batch["x_stat"].to(device),
            batch["weapon_idx"].to(device),
            batch["mask"].to(device),
        ).cpu().numpy()
        for j, emb in enumerate(z):
            row = frame.iloc[start + j]
            rows.append({"player_id": row["player_id"], "sequence_id": row["sequence_id"], "embedding": emb, "label": row["label"]})
        start += len(z)
    return pd.DataFrame(rows)


def train_player_classifier(
    train_embeddings: pd.DataFrame,
    valid_embeddings: pd.DataFrame,
    cfg: TrainingConfig | None = None,
    pooling: str = "mean",
) -> tuple[PlayerRiskClassifier, float, dict[str, float]]:
    cfg = cfg or TrainingConfig()
    device = torch.device(cfg.device)
    emb_dim = len(train_embeddings["embedding"].iloc[0])
    model = PlayerRiskClassifier(embedding_dim=emb_dim, pooling=pooling).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.player_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(PlayerBagDataset(train_embeddings), batch_size=cfg.player_batch_size, shuffle=True, collate_fn=collate_player_bags)
    for _ in range(cfg.player_epochs):
        model.train()
        for _, x, mask, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            opt.step()

    y_valid, s_valid = score_player_embeddings(model, valid_embeddings, cfg=cfg)
    threshold, _ = select_threshold_by_f1(y_valid, s_valid)
    return model, threshold, binary_metrics(y_valid, s_valid, threshold)


@torch.no_grad()
def score_player_embeddings(model: PlayerRiskClassifier, embeddings: pd.DataFrame, cfg: TrainingConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or TrainingConfig()
    device = torch.device(cfg.device)
    loader = DataLoader(PlayerBagDataset(embeddings), batch_size=cfg.player_batch_size, shuffle=False, collate_fn=collate_player_bags)
    y_true, scores = [], []
    model.eval()
    for _, x, mask, y in loader:
        logits = model(x.to(device), mask.to(device))
        y_true.extend(y.numpy().astype(int).tolist())
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(scores)

