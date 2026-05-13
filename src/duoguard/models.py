"""Neural models for the sanitized DuoGuard implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_pool_time_first(x: torch.Tensor, mask: torch.Tensor, first_k: int = 7) -> torch.Tensor:
    """Concatenate global max/mean and first-k max/mean pooling."""

    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = mask.bool()
    mask_f = mask.unsqueeze(-1).float()
    neg = torch.finfo(x.dtype).min

    global_max = x.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values
    global_mean = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    first_mask = torch.zeros_like(mask)
    first_mask[:, : min(first_k, x.shape[1])] = mask[:, : min(first_k, x.shape[1])]
    first_mask_f = first_mask.unsqueeze(-1).float()
    first_max = x.masked_fill(~first_mask.unsqueeze(-1), neg).max(dim=1).values
    first_mean = (x * first_mask_f).sum(dim=1) / first_mask_f.sum(dim=1).clamp_min(1.0)

    return torch.cat([global_max, global_mean, first_max, first_mean], dim=1)


class EventFeatureEmbedder(nn.Module):
    """Embed categorical event features and pass through continuous features."""

    def __init__(
        self,
        d_seq: int,
        categorical_indices: dict[str, int],
        emb_dim: int = 4,
        n_damage_regions: int = 7,
        n_relative_directions: int = 10,
        n_scopes: int = 8,
        n_distance_bins: int = 8,
        n_weapon_categories: int = 9,
    ):
        super().__init__()
        self.categorical_indices = dict(categorical_indices)
        excluded = set(self.categorical_indices.values())
        self.continuous_indices = [i for i in range(d_seq) if i not in excluded]

        self.damage_emb = nn.Embedding(n_damage_regions, emb_dim)
        self.direction_emb = nn.Embedding(n_relative_directions, emb_dim)
        self.scope_emb = nn.Embedding(n_scopes, emb_dim)
        self.distance_bin_emb = nn.Embedding(n_distance_bins, emb_dim)
        self.weapon_category_emb = nn.Embedding(n_weapon_categories, emb_dim)

        self.out_dim = len(self.continuous_indices) + 5 * emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x[:, :, self.continuous_indices]]

        def take(name: str) -> torch.Tensor:
            return x[:, :, self.categorical_indices[name]].long().clamp_min(0)

        parts.append(self.damage_emb(take("damage_region_code").clamp_max(self.damage_emb.num_embeddings - 1)))
        parts.append(self.direction_emb(take("relative_direction_code").clamp_max(self.direction_emb.num_embeddings - 1)))
        parts.append(self.scope_emb(take("scope_code").clamp_max(self.scope_emb.num_embeddings - 1)))
        parts.append(self.distance_bin_emb(take("distance_bin_code").clamp_max(self.distance_bin_emb.num_embeddings - 1)))
        parts.append(self.weapon_category_emb(take("weapon_category_code").clamp_max(self.weapon_category_emb.num_embeddings - 1)))
        return torch.cat(parts, dim=-1)


class DuoGuardSequenceEncoder(nn.Module):
    """Dual-view sequence encoder used to produce sequence embeddings."""

    def __init__(
        self,
        d_seq: int,
        d_stat: int,
        n_weapons: int,
        categorical_indices: dict[str, int],
        emb_dim: int = 4,
        hidden_dim: int = 128,
        sequence_embedding_dim: int = 160,
        first_k: int = 7,
    ):
        super().__init__()
        self.first_k = int(first_k)
        self.event_embedder = EventFeatureEmbedder(d_seq=d_seq, categorical_indices=categorical_indices, emb_dim=emb_dim)
        self.weapon_embedding = nn.Embedding(n_weapons, emb_dim)
        event_dim = self.event_embedder.out_dim + emb_dim

        self.lstm = nn.LSTM(input_size=event_dim, hidden_size=hidden_dim, batch_first=True)
        self.stat_mlp = nn.Sequential(
            nn.Linear(d_stat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Linear(hidden_dim * 4 + 64, sequence_embedding_dim)
        self.sequence_head = nn.Sequential(
            nn.Linear(sequence_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def encode(self, x_seq: torch.Tensor, x_stat: torch.Tensor, weapon_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        event_emb = self.event_embedder(x_seq)
        w = self.weapon_embedding(weapon_idx.long()).unsqueeze(1).expand(-1, x_seq.shape[1], -1)
        event_emb = torch.cat([event_emb, w], dim=-1)
        out, _ = self.lstm(event_emb)
        temporal = masked_pool_time_first(out, mask, first_k=self.first_k)
        stat = self.stat_mlp(x_stat)
        return self.fusion(torch.cat([temporal, stat], dim=1))

    def forward(self, x_seq: torch.Tensor, x_stat: torch.Tensor, weapon_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encode(x_seq, x_stat, weapon_idx, mask)
        return self.sequence_head(z).squeeze(-1)


class PlayerRiskClassifier(nn.Module):
    """Player-level classifier over a variable-size bag of sequence embeddings."""

    def __init__(self, embedding_dim: int = 160, pooling: str = "mean", hidden_dim: int = 128, top_pct: float = 25.0):
        super().__init__()
        if pooling not in {"mean", "max", "attention", "top_percent"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        self.top_pct = float(top_pct)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.top_score = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def pool(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if self.pooling == "mean":
            denom = mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
            return (embeddings * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        if self.pooling == "max":
            neg = torch.finfo(embeddings.dtype).min
            return embeddings.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values
        if self.pooling == "attention":
            logits = self.attention(embeddings).squeeze(-1).masked_fill(~mask, -1e9)
            alpha = torch.softmax(logits, dim=1)
            return (embeddings * alpha.unsqueeze(-1)).sum(dim=1)

        scores = self.top_score(embeddings).squeeze(-1).masked_fill(~mask, -1e9)
        lengths = mask.sum(dim=1).clamp_min(1)
        ks = torch.ceil(lengths.float() * max(0.0, min(self.top_pct, 100.0)) / 100.0).long().clamp_min(1)
        k_max = int(ks.max().item())
        idx = scores.topk(k=k_max, dim=1).indices
        selected = embeddings.gather(1, idx.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]))
        selected_mask = mask.gather(1, idx)
        rank_mask = torch.arange(k_max, device=embeddings.device).unsqueeze(0) < ks.unsqueeze(1)
        weight = (selected_mask & rank_mask).unsqueeze(-1).float()
        return (selected * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(embeddings, mask)
        return self.classifier(pooled).squeeze(-1)


def default_categorical_indices(sequence_features: list[str]) -> dict[str, int]:
    names = [
        "damage_region_code",
        "relative_direction_code",
        "scope_code",
        "distance_bin_code",
        "weapon_category_code",
    ]
    return {name: sequence_features.index(name) for name in names}

