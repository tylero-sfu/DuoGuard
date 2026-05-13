# DuoGuard: Sanitized Implementation

This repository contains a sanitized implementation package for DuoGuard,
a weakly supervised framework for shooting-assistance cheating-risk scoring
from server-side combat telemetry.

The package is intended for anonymous review and reproducibility of the
modeling pipeline. It includes:

- shooting-sequence construction utilities,
- distance interpolation and feature construction,
- the dual-view sequence encoder,
- player-level bag aggregation models,
- evaluation metrics and threshold selection,
- plotting helpers, and
- a small synthetic demo.

The repository does **not** include proprietary telemetry, player identifiers,
operational enforcement records, or code for constructing operational labels.
Training labels are expected to be supplied externally as a de-identified table
with columns such as `player_id`, `label`, and `split`.

## Repository Layout

```text
configs/
  duoguard_default.json
examples/
  synthetic_events.csv
  synthetic_player_labels.csv
scripts/
  run_synthetic_demo.py
src/duoguard/
  evaluation.py
  features.py
  models.py
  plotting.py
  preprocessing.py
  training.py
```

## Quick Start

```bash
python examples/generate_synthetic_data.py
python scripts/run_synthetic_demo.py
```

The demo builds short shooting sequences from synthetic combat events, trains
a small DuoGuard-style sequence encoder, aggregates sequence embeddings by
player with mean pooling, and reports player-level metrics.
The synthetic files are toy examples only. They are not sampled from, derived
from, or statistically fitted to the proprietary telemetry used in the paper.

## Data Interface

The public code assumes that event data are already available in a
de-identified, tabular form. Required columns for the synthetic-compatible
preprocessing path are documented in `src/duoguard/preprocessing.py`.

Operational label construction is intentionally excluded. To train on a private
dataset, provide a separate label table:

```text
player_id,label,split
p0001,0,train
p0002,1,train
p0003,0,valid
```

where `label` is a weak player-level supervision signal.

## Scope

This code is a disclosure-safe implementation of the methods described in the
paper. It is not a production anti-cheat system and is not intended for
automatic punitive action.
