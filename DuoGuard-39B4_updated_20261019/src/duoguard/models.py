"""48-dimensional events -> frozen 64-dimensional sequences -> player score."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class InputEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cont_indices = [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15]
        self.scope_emb = nn.Embedding(8, 8)
        self.dmg_emb = nn.Embedding(7, 8)
        self.dir_emb = nn.Embedding(10, 8)
        self.out_dim = 48

    def forward(self, x):
        distance = F.one_hot(x[:, :, 7].long().clamp(0, 5), 6).to(x.dtype)
        distance = distance * x[:, :, 8:9]
        weapon = F.one_hot(x[:, :, 14].long().clamp(0, 6), 7).to(x.dtype)
        return torch.cat([x[:, :, self.cont_indices], self.scope_emb(x[:, :, 1].long()),
                          self.dmg_emb(x[:, :, 13].long()), self.dir_emb(x[:, :, 11].long()),
                          distance, weapon], dim=-1)


class SequenceCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = InputEmbedder()
        self.lstm = nn.LSTM(48, 64, batch_first=True)

    def forward(self, x, lengths):
        if x.ndim != 3 or x.shape[-1] != 16:
            raise ValueError("Expected [sequences, shots, 16]")
        if lengths.shape != (len(x),) or not bool(((lengths > 0) & (lengths <= x.shape[1])).all()):
            raise ValueError("Invalid sequence lengths")
        output, _ = self.lstm(self.embedder(x))
        return output[torch.arange(len(x), device=x.device), lengths.long() - 1]


class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SequenceCore()
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1))

    def forward(self, x, lengths):
        return self.head(self.encoder(x, lengths)).squeeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, recurrent_chunk=2048):
        super().__init__()
        if recurrent_chunk < 1:
            raise ValueError("recurrent_chunk must be positive")
        self.recurrent_chunk = recurrent_chunk
        self.lstm = nn.LSTM(65, 64, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
                                        nn.Dropout(.3), nn.Linear(64, 1))

    def _chunk(self, x, h, c):
        out, (hn, cn) = self.lstm(x, (h, c))
        return out, hn, cn

    def forward_features(self, x, match_start, mask):
        mask = mask.bool()
        if x.ndim != 3 or x.shape[-1] != 64 or mask.shape != x.shape[:2] or match_start.shape != mask.shape:
            raise ValueError("Invalid player bag shapes")
        if not bool(mask.any(1).all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("Bags must be nonempty with right padding only")
        z = torch.cat([x, match_start.unsqueeze(-1).to(x.dtype)], -1)
        h = x.new_zeros(1, len(x), 64)
        c = torch.zeros_like(h)
        running_max = denom = numer = None
        # Preserve recurrent state and global attention across computational chunks.
        for start in range(0, z.shape[1], self.recurrent_chunk):
            piece = z[:, start:start + self.recurrent_chunk]
            if self.training and torch.is_grad_enabled():
                out, h, c = checkpoint(self._chunk, piece, h, c, use_reentrant=False)
            else:
                out, h, c = self._chunk(piece, h, c)
            valid = mask[:, start:start + self.recurrent_chunk]
            energy = self.attention(out).squeeze(-1).masked_fill(~valid, -torch.inf)
            chunk_max = energy.max(1).values
            if running_max is None:
                running_max = chunk_max
                weight = torch.exp(energy - running_max[:, None]).masked_fill(~valid, 0)
                denom = weight.sum(1)
                numer = (weight.unsqueeze(-1) * out).sum(1)
            else:
                new_max = torch.maximum(running_max, chunk_max)
                old_scale = torch.exp(running_max - new_max)
                weight = torch.exp(energy - new_max[:, None]).masked_fill(~valid, 0)
                denom = denom * old_scale + weight.sum(1)
                numer = numer * old_scale[:, None] + (weight.unsqueeze(-1) * out).sum(1)
                running_max = new_max
        return numer / denom[:, None]

    def forward(self, x, match_start, mask):
        return self.classifier(self.forward_features(x, match_start, mask)).squeeze(-1)
