# Run-folder and evaluation-pickle artifacts

The on-disk names and pickle keys are part of the reproducibility contract.
They are not renamed between releases.

## Run folder

A training run writes under `<models-root>/<model-fullname>/run_XXX/`:

```text
run_XXX/
  model_checkpoint.pth     latest resumable checkpoint
  model_best.pth           best-validation checkpoint (or a relative symlink in sweep runs)
  model_best.pth -> best-epochs/epoch_NNN.pth
  best-selection.json      sweep manifest (sweep runs only)
  hyperparams.json         resolved run configuration
  log.pk.xz                compressed performance curves, one entry per epoch
  best-epochs/             per-epoch sweep snapshots (sweep runs only)
```

`log.pk.xz` is a compressed mapping preserving the historical
`performance_curves` keys, including `epoch`, `train_loss`, `lr`,
`auc_roc_macro`, `classification_class`, `localization`, and for sweep runs
`localization_sweep` and `localization_sweep_selection`.

## Evaluation pickle

`clarity-evaluate` writes:

```text
<checkpoint-stem>-evaluation-classify.pk
<checkpoint-stem>-evaluation-localize-all-images.pk
```

The classification pickle contains `val_metrics` and `metrics` (macro and
per-class values). The localization pickle contains `IoU` and `IoBB` families,
each with `val_metrics`, `metrics`, per-threshold/per-finding accuracy, and
the heatmap thresholds. The top-level `localization_method` block records:

```text
bbox_method, selection, selection_metric,
params (per finding), val5, fit_seconds, fit_workers,
optimize_metrics_separately (legacy runs)
```

Archived SingleCut pickles have no `localization_method` block; their scalar
heatmap thresholds identify the legacy method. New runs always write the
method block, so downstream readers do not have to infer it.

## Reading the paper tables

The reported classification and localization metrics are read from these
pickles. The paper's figures, appendix tables, bootstrap intervals and
ablation tables are produced by internal analysis code that is not part of
this repository.

## Frozen calibration files

`calibration/a1_onebox_nih_*.json` are schema-1 `localization_method` blocks
fitted on the NIH validation split (arm A1, NIH-calibrated). Besides the
per-finding `params` keyed by the NIH class index (provenance), they carry a
`target_label_map` keyed by the evaluation label index; `clarity-evaluate
--frozen-onebox-calibration` loads that mapping through
`clarity.localization.artifacts.load_frozen_calibration`. See `docs/transfer.md`.
