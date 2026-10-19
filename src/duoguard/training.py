"""Two-stage training with validation-only checkpoint and threshold selection."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .data import FeatureScaler, attach_external_labels
from .evaluation import binary_metrics, select_threshold_by_f1, weekly_report
from .models import SequenceClassifier, SequenceCore, TemporalAttention


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class PlayerBagDataset(Dataset):
    def __init__(self, embeddings, meta):
        self.embeddings = embeddings
        self.rows = []
        indexed = meta.reset_index(drop=True).copy()
        indexed['_index'] = np.arange(len(indexed))
        for (player, window), group in indexed.groupby(['player_id', 'test_window'], sort=True):
            group = group.sort_values(['sequence_start_ms', 'match_id', 'sequence_id'], kind='stable')
            match = group['match_id'].to_numpy()
            flags = np.concatenate([[True], match[1:] != match[:-1]])
            target = float(group['label'].iloc[0]) if 'label' in group else -1.
            self.rows.append((str(player), str(window), group['_index'].to_numpy(), flags, target))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        player, window, ids, flags, target = self.rows[index]
        return player, window, self.embeddings[ids], flags, target


def collate_bags(rows):
    max_length = max(len(row[2]) for row in rows)
    x = torch.zeros(len(rows), max_length, 64)
    flags = torch.zeros(len(rows), max_length, dtype=torch.bool)
    mask = torch.zeros_like(flags)
    for i, row in enumerate(rows):
        n = len(row[2])
        x[i, :n] = torch.as_tensor(row[2])
        flags[i, :n] = torch.as_tensor(row[3])
        mask[i, :n] = True
    return [r[0] for r in rows], [r[1] for r in rows], x, flags, mask, torch.tensor([r[4] for r in rows])


@torch.no_grad()
def extract_embeddings(encoder, x, lengths, batch_size, device):
    encoder.eval()
    result = []
    for start in range(0, len(x), batch_size):
        result.append(encoder(torch.from_numpy(x[start:start + batch_size]).to(device),
                              torch.from_numpy(lengths[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def score_sequences(model, loader, device):
    model.eval()
    targets, scores = [], []
    for x, lengths, y in loader:
        scores.extend(model(x.to(device), lengths.to(device)).sigmoid().cpu().tolist())
        targets.extend(y.tolist())
    return np.asarray(targets), np.asarray(scores)


@torch.no_grad()
def score_players(model, loader, device):
    model.eval()
    rows = []
    for ids, windows, x, flags, mask, targets in loader:
        scores = model(x.to(device), flags.to(device), mask.to(device)).sigmoid().cpu().numpy()
        for i, player in enumerate(ids):
            rows.append({'account_id': player, 'test_window': windows[i], 'target': int(targets[i]),
                         'score': float(scores[i]), 'n_seq': int(mask[i].sum())})
    return pd.DataFrame(rows)


def fit(x, lengths, meta, labels, cfg, output_dir, device='cpu'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('*.pt')):
        raise ValueError('Output contains checkpoints; use a new directory')
    cfg = copy.deepcopy(cfg)
    for key in ['encoder_epochs', 'player_epochs', 'batch_size', 'player_batch_size', 'recurrent_chunk']:
        if int(cfg[key]) < 1:
            raise ValueError(key + ' must be positive')
    seed_everything(int(cfg['seed']))
    meta = attach_external_labels(meta, labels)
    train = meta['split'].eq('train').to_numpy()
    valid = meta['split'].eq('valid').to_numpy()
    test = meta['split'].eq('test').to_numpy()
    scaler = FeatureScaler().fit(x[train], lengths[train])
    x = scaler.transform(x, lengths)
    scaler.save(output / 'scaler.json')
    save_json(output / 'config.json', cfg)

    def sequence_loader(select, shuffle=False):
        dataset = TensorDataset(torch.from_numpy(x[select]), torch.from_numpy(lengths[select]),
                                torch.tensor(meta.loc[select, 'label'].to_numpy(), dtype=torch.float32))
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle)

    model = SequenceClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['encoder_learning_rate'], weight_decay=cfg['weight_decay'])
    targets = meta.loc[train, 'label'].to_numpy()
    positive_weight = float((targets == 0).sum() / (targets == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    train_loader, valid_loader = sequence_loader(train, True), sequence_loader(valid)
    history = []
    best_ap = -1.
    best_encoder = None
    for epoch in range(1, cfg['encoder_epochs'] + 1):
        model.train()
        for batch_x, batch_lengths, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device), batch_lengths.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
        y, scores = score_sequences(model, valid_loader, device)
        ap = binary_metrics(y, scores, .5)['auprc']
        record = {'stage': 'sequence', 'epoch': epoch, 'validation_auprc': ap}
        history.append(record)
        print(json.dumps(record), flush=True)
        if ap > best_ap:
            best_ap, best_encoder = ap, epoch
            torch.save({'model': cpu_state(model), 'epoch': epoch, 'validation_auprc': ap,
                        'positive_weight': positive_weight}, output / 'encoder_best.pt')
    saved = torch.load(output / 'encoder_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    encoder = model.encoder.eval().requires_grad_(False)
    for parameter in encoder.parameters():
        parameter.grad = None
    initial_encoder = cpu_state(encoder)
    embeddings = extract_embeddings(encoder, x, lengths, cfg['batch_size'], device)

    def player_loader(select, shuffle=False):
        dataset = PlayerBagDataset(embeddings[select], meta.loc[select].reset_index(drop=True))
        return DataLoader(dataset, batch_size=cfg['player_batch_size'], shuffle=shuffle, collate_fn=collate_bags)

    player = TemporalAttention(cfg['recurrent_chunk']).to(device)
    optimizer = torch.optim.AdamW(player.parameters(), lr=cfg['player_learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    train_loader, valid_loader = player_loader(train, True), player_loader(valid)
    best_ap, best_player = -1., None
    for epoch in range(1, cfg['player_epochs'] + 1):
        player.train()
        for _, _, z, flags, mask, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(player(z.to(device), flags.to(device), mask.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(player.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
        prediction = score_players(player, valid_loader, device)
        ap = binary_metrics(prediction['target'], prediction['score'], .5)['auprc']
        record = {'stage': 'player', 'epoch': epoch, 'validation_auprc': ap}
        history.append(record)
        print(json.dumps(record), flush=True)
        if ap > best_ap:
            best_ap, best_player = ap, epoch
            torch.save({'model': cpu_state(player), 'epoch': epoch, 'validation_auprc': ap}, output / 'player_best.pt')
    saved = torch.load(output / 'player_best.pt', map_location='cpu', weights_only=True)
    player.load_state_dict(saved['model'])
    validation = score_players(player, valid_loader, device)
    threshold = select_threshold_by_f1(validation['target'], validation['score'])
    validation.to_csv(output / 'validation_predictions.csv', index=False)
    if any(not torch.equal(initial_encoder[k], v.cpu()) for k, v in encoder.state_dict().items()):
        raise RuntimeError('Frozen encoder changed during Stage 2')
    if any(p.grad is not None for p in encoder.parameters()):
        raise RuntimeError('Frozen encoder accumulated gradients')
    torch.save({'encoder': cpu_state(encoder), 'player': cpu_state(player), 'threshold': threshold,
                'recurrent_chunk': cfg['recurrent_chunk']}, output / 'inference.pt')
    report = {'encoder_best_epoch': best_encoder, 'player_best_epoch': best_player,
              'threshold': threshold, 'validation': binary_metrics(validation['target'], validation['score'], threshold),
              'parameters': {'stage1': sum(p.numel() for p in model.parameters()),
                             'stage2': sum(p.numel() for p in player.parameters()),
                             'inference': sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in player.parameters())},
              'encoder_frozen_verified': True, 'history': history}
    if test.any():
        prediction = score_players(player, player_loader(test), device)
        prediction.to_csv(output / 'test_predictions.csv', index=False)
        report['test'] = weekly_report(prediction, threshold)
    else:
        report['test'] = None
    save_json(output / 'metrics.json', report)
    return report


def predict(x, lengths, meta, model_dir, device='cpu', batch_size=256, player_batch_size=32):
    model_dir = Path(model_dir)
    saved = torch.load(model_dir / 'inference.pt', map_location='cpu', weights_only=True)
    encoder = SequenceCore().to(device)
    player = TemporalAttention(saved['recurrent_chunk']).to(device)
    encoder.load_state_dict(saved['encoder'])
    player.load_state_dict(saved['player'])
    x = FeatureScaler.load(model_dir / 'scaler.json').transform(x, lengths)
    embeddings = extract_embeddings(encoder, x, lengths, batch_size, device)
    loader = DataLoader(PlayerBagDataset(embeddings, meta), batch_size=player_batch_size, collate_fn=collate_bags)
    prediction = score_players(player, loader, device).drop(columns='target')
    prediction['prediction'] = (prediction['score'] >= saved['threshold']).astype(int)
    return prediction
