# Evaluation protocols

`clarity-evaluate` writes a compressed evaluation pickle next to the
checkpoint. Choose the split with `--split classify` or `--split localize`.

## Classification

`--split classify` evaluates the validation split and the official NIH test
split. The output pickle contains macro and per-class AUC, F1, sensitivity and
specificity under `val_metrics` and `metrics`.

The paper's validation split is the historical `classification_val_list_70`
list; supply it with `--splits-dir` to reproduce the reported protocol. Without
a manifest the official split is used.

## Localization

`--split localize` evaluates the localization validation and test split. It
requires `--splits-dir` (the historical localization lists) and a checkpoint
trained with localization attention.

The localization evaluation uses OneBox:

- the fused model-attention heatmap is gated between a low and high score
  threshold;
- each 4-connected component contributes its middle row/column span as a box;
- the five parameters are fitted on the validation split by the exact 0.05
  score grid and `joint_argmax` selection;
- the validation-fitted parameters are frozen for the test split.

The default protocol uses all images. `--only_correctly_classified` selects
the historical filtered protocol, but it is not the reported all-images
protocol.

Both IoU and IoBB families are reported from the same validation calibration;
IoBB is reported but never used for selection. The output pickle contains the
per-finding parameters and the method block so a reader can recover exactly
which method, selection rule and score thresholds produced the numbers.

## Split-manifest requirement

`--splits-dir` is required for the paper's localization evaluation. The
manifest loader validates the counts and patient disjointness; `--verify-splits`
prints the checks and exits non-zero on failure. The lists themselves are not
included in this repository.

## Metrics

- **IoU accuracy**: fraction of annotated findings whose predicted boxes hit
  the ground-truth box at the score threshold.
- **IoBB accuracy**: the same detection rule at the box level.
- **MSLI / UDP**: the mean localization accuracy over the score grid and the
  paper's combined classification/localization score.
- **AFP / boxes per image**: optional box-budget statistics exposed by the
  localization evaluation.

See `docs/onebox.md` for the exact grid and `docs/artifacts.md` for the pickle
keys.

## Transfer to RSNA Pneumonia and VinDr-CXR

`clarity-evaluate -ds rsna|vindr` evaluates the NIH checkpoint on the external
datasets in two arms: **A1 (NIH-calibrated)** scores a frozen NIH-fitted OneBox
calibration (`--frozen-onebox-calibration`), **A2 (Target-calibrated)** refits
OneBox on the target validation split. The label maps, split-list format,
commands and the external baselines are in `docs/transfer.md`.
