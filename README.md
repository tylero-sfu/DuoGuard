# DuoGuard

This repository contains the anonymous research implementation package for
DuoGuard, a weakly supervised framework for shooting-assistance cheating-risk
scoring from server-side combat telemetry in battle royale shooters.

The package is intended for anonymous review and inspection of the method
pipeline. It provides the public, disclosure-safe parts of the implementation:

- PUBG API-style combat telemetry conversion into the generic event schema,
- combat-log integration and outcome matching,
- shooting-sequence construction,
- distance interpolation,
- event-level tensor and statistical descriptor construction,
- the dual-view sequence encoder,
- player-level bag aggregation models,
- evaluation metrics and threshold selection,
- plotting helpers, and
- a small synthetic demo.

The repository does **not** include proprietary gameplay telemetry, player
identifiers, operational enforcement records, ban/action taxonomies, internal
label curation rules, or code for constructing player labels. Training labels
are expected to be supplied externally as a de-identified table with columns
such as `player_id`, `label`, and `split`.

## Repository Layout

```text
configs/
  duoguard_default.json
docs/
  preprocessing_pipeline.md
  pubg_api_adapter.md
examples/
  synthetic_events.csv
  synthetic_player_labels.csv
scripts/
  build_sequences.py
  run_table_pipeline.py
  run_synthetic_demo.py
src/duoguard/
  evaluation.py
  features.py
  models.py
  plotting.py
  preprocessing.py
  pubg_api_adapter.py
  training.py
```

## What Is Publicly Implemented?

The public pipeline follows the same high-level structure as the manuscript:

```text
PUBG API-style combat events or generic combat-event table
  -> combat-log integration and outcome matching
  -> weapon-aware shooting-sequence segmentation
  -> distance interpolation
  -> T=32 event tensor and statistical descriptor construction
  -> dual-view sequence encoder
  -> sequence embedding extraction
  -> player-level mean-pooling aggregation
  -> player-level risk score and evaluation metrics
```

The package therefore exposes how combat telemetry is transformed into
shooting sequences and model inputs. It intentionally does not expose how
operator-derived labels were created.

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

For a table-driven run that more closely follows the paper pipeline:

```bash
python scripts/run_table_pipeline.py \
  --events examples/synthetic_events.csv \
  --labels examples/synthetic_player_labels.csv \
  --output-dir outputs/table_pipeline_demo
```

This command builds shooting sequences, trains the sequence encoder, extracts
sequence embeddings, trains the player-level classifier, and writes metrics and
predictions. With private data, replace the two input files with de-identified
event and player-label tables that follow the documented schema.

## Data Interface

The public code supports two input paths. First, de-identified tabular combat
events can be provided directly with the generic schema documented in
`src/duoguard/preprocessing.py`. Second, PUBG API-style combat telemetry can be
converted into this generic schema with `src/duoguard/pubg_api_adapter.py`.
The adapter covers combat-log events only and excludes all operational
enforcement or label-construction logic.

The PUBG API-style adapter recognizes the following public combat event names:

```text
LogPlayerAttack
LogPlayerTakeDamage
LogPlayerMakeGroggy
LogPlayerKill
LogPlayerKillV2
```

Operational label construction is intentionally excluded. To train on a private
dataset, provide a separate label table:

```text
player_id,label,split
p0001,0,train
p0002,1,train
p0003,0,valid
```

where `label` is a weak player-level supervision signal.

## Synthetic Data

The files in `examples/` are small artificial examples for exercising the code
path. They are not sampled from, derived from, or statistically fitted to the
proprietary telemetry used in the study. They are included only so that the
preprocessing, training, and evaluation scripts can be run without private data.

## Scope

This code is a disclosure-safe research implementation of the methods described
in the paper. It is not a production anti-cheat system and is not intended for
automatic punitive action. The private study used proprietary telemetry exports
and operator-derived labels; those materials are not redistributed in this
package.
