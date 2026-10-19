# DuoGuard

Portable implementation of the revised DuoGuard main model. Labels and split
assignments are supplied by the user; this package does not construct them.

The model encodes each shooting sequence with a 64-unit LSTM, freezes the selected
encoder, and processes chronological sequence embeddings using a second 64-unit
LSTM and attention pooling. A match-start flag accompanies each sequence embedding.
Each stage selects its checkpoint by validation AUPRC. The final threshold
maximizes validation F1 over 0.01 through 0.99 in steps of 0.01.

## Install

Python 3.10 or later and PyTorch 2.1 or later are required. CPU execution is
sufficient for the included example.

```bash
python -m pip install -e .
```

## Quick smoke test

```bash
python scripts/run_synthetic_demo.py --output-dir outputs/smoke
python -m unittest discover -s tests -v
```

The smoke test regenerates the existing synthetic example, trains both stages
for one epoch, evaluates six artificial test windows, and checks that reloading
the model preserves scores. It checks the frozen encoder and parameter counts:
33,609 for Stage 1, 40,002 for Stage 2, and 69,386 for inference without the
auxiliary sequence head. The normal configuration uses five epochs per stage.
Use a new output directory for each training run.

The toy records are generated independently of the research data. Toy scores
demonstrate execution, not the paper's numerical results. Some tiny example
windows have only one class; undefined metrics are reported as null, not zero.

## Existing example CSV workflow

```bash
python examples/generate_synthetic_data.py
python scripts/run_table_pipeline.py \
  --events examples/synthetic_events.csv \
  --labels examples/synthetic_player_labels.csv \
  --output-dir outputs/example --device cpu
```

The existing filenames and label CSV interface are retained. The legacy example
contains one row per shot with its outcome already attached. It is not a set of
separate raw attack/damage records. All-miss sequences are retained. See
[input schema](docs/input_schema.md) for the canonical research input route.

## Preprocessed research input

For existing preprocessed data, provide the 16 canonical features directly,
without recomputing them through the toy adapter. The sequence NPZ contains raw
(not standardized) features, lengths, and ordering metadata. Labels remain in a
separate CSV. No pickle is needed for data input.

```bash
python scripts/run_table_pipeline.py \
  --sequences data/sequences.npz --labels data/player_labels.csv \
  --output-dir outputs/main --device cuda:0
```

Alternatively, a CSV with all 16 canonical feature columns and sequence metadata
can use `--events`. A supplied `sequence_id` preserves sequence membership.
`scripts/build_sequences.py` converts this table into the portable NPZ format.

Training-only statistics standardize the continuous input columns. Development
bags group an account's complete supplied development history. Test bags are
separate for each `test_window`; repeated test accounts are allowed across windows,
but training, validation, and test account sets must be disjoint. The input must
place test observations after the development observation period.

## Outputs and inference

Training writes `encoder_best.pt`, `player_best.pt`, `inference.pt`, `scaler.json`,
the resolved configuration, validation/test predictions, and `metrics.json`.
The latter includes the selected epochs and weekly, equal-weight, and pooled
test metrics. If test data are absent, no test score is fabricated.

```bash
python scripts/predict.py --sequences data/sequences.npz \
  --model-dir outputs/main --output outputs/inference.csv

python scripts/evaluate.py --predictions outputs/main/test_predictions.csv \
  --threshold 0.45 --bootstrap 1000 --output outputs/ci.json
```

Replace `0.45` with the threshold in your run's `metrics.json`. Evaluation never
selects a threshold from test data. The bootstrap samples account IDs and retains
all observations of a sampled account across windows. It estimates uncertainty
in the equal-weight mean of weekly AUPRC and F1, not retraining variability.
Use only trusted, locally generated model checkpoints.

## Scope

This release implements the main two-stage training and inference path and its
fixed-prediction evaluation. It does not claim to reproduce every comparison
table: the other encoder families, ML/MIL baselines, joint-training experiments,
and operational follow-up analysis are not implemented in this package.

The canonical input route preserves the reported model computations. The legacy
CSV adapter supports the synthetic example; it does not reconstruct unavailable
movement measurements or proprietary raw-event integrations. A public API parser
is not included because API-style records alone are not a verified substitute
for the canonical feature input.

This is an in-memory reference runner. Large datasets require account-complete
batching/sharding around the same model and evaluation interfaces. The recurrent
chunk option limits activation memory without truncating player history, but does
not make the input loader an out-of-core production pipeline. Different batch
sizes, hardware, and stochastic training can produce different fitted models.

Reproducing the reported numerical results requires the original dataset, which
is not publicly available. No real telemetry, account labels, enforcement records,
label-construction rules, or fitted research checkpoints are distributed.

## License

MIT; see [LICENSE](LICENSE).
