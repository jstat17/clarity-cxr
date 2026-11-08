# Transfer to RSNA Pneumonia and VinDr-CXR

The paper evaluates the ChestX-ray14 (NIH) checkpoint on two external
datasets without retraining it:

- **RSNA Pneumonia Detection Challenge**: one finding, `Pneumonia`, with
  multi-box ground truth;
- **VinDr-CXR**: 11 findings for classification and 6 box findings for
  localization, with `Nodule/Mass` fused from two NIH classes.

This repository ships the readers, the frozen label maps, the evaluation path
and the baselines. It does **not** ship data, checkpoints, evaluation pickles
or split lists.

## Arms

All CLARiTy arms use the NIH-trained 14-class checkpoint. They differ only in
where the OneBox localization calibration comes from.

| id | name (as in the paper) | calibration | `clarity-evaluate` |
|---|---|---|---|
| A1 | **NIH-calibrated** | fitted on the NIH validation split, frozen, applied unchanged to the target (strict zero-shot) | `--frozen-onebox-calibration FILE` |
| A2 | **Target-calibrated** | refitted on the target (RSNA or VinDr-CXR) validation split | omit `--frozen-onebox-calibration` |

`--arm a1|a2|nih-calibrated|target-calibrated` is an optional guard: it makes
`clarity-evaluate` refuse a command that does not match the named arm. The log
prints `Transfer arm: A1 (NIH-calibrated)` / `A2 (Target-calibrated)` for every
transfer localization run. The names live in `clarity.evaluation.arms`.

The two external baselines are read through HiResCAM and calibrated on the
target validation split, like A2:

| baseline | classifier | maps |
|---|---|---|
| TorchXRayVision DenseNet121-NIH, 224 px | the published checkpoint, unchanged (zero-shot head) | HiResCAM, last convolutional stage |
| RAD-DINO, frozen encoder, 518 px | one logistic-regression probe per finding, fitted on the target train list (C chosen on validation) | HiResCAM on the last transformer block, target = the probe's own logit |

## Datasets

Obtain the data yourself and keep it outside the repository.

- RSNA Pneumonia Detection Challenge (stage 2 train labels, `stage_2_train_labels.csv`);
- VinDr-CXR (PhysioNet, credentialed access: the annotation CSVs under
  `annotations/` and the `image_labels_*.csv` files).

Point the tools at the parent directory with `--data-root` or
`CLARITY_DATA_ROOT`, or at a dataset directory with `CXR_RSNA_PATH` /
`CXR_VINDR_PATH`. Images are read from PNG files you convert yourself, at the
evaluation resolution (224 or 512); pass the directory with `-img`. Anatomical
masks (`-seg`) come from `clarity-segment` / `clarity-prepare`.

### Split lists

The split lists are not distributed (VinDr-CXR identifiers are covered by the
PhysioNet agreement, and some of the paper's splits cannot be regenerated from
a seed). Supply your own: a directory with three text files, one image file
name per line (`<stem>.png`):

| dataset | files | option |
|---|---|---|
| RSNA | `rsna_train_list.txt`, `rsna_val_list.txt`, `rsna_test_list.txt` | `--rsna-splits-dir` |
| VinDr-CXR | `vindr_train_list.txt`, `vindr_val_list.txt`, `vindr_test_list.txt` | `--vindr-splits-dir` |

The paper's protocol, for reference: RSNA test is every RSNA image whose NIH
source image is in the NIH official test list (7,186 images, 2,589 positives;
validation 2,438, train 17,060); VinDr-CXR uses the official 3,000-image test
split, 1,500 validation images and 13,499 training images. The train list is
only used by the RAD-DINO probe. Images missing from the PNG directory are
dropped from a list, and the run prints the coverage: A1 can score a partial
validation list, A2 needs it complete.

## Label maps

Frozen, derived from `CXR_14_ABNORMALITIES` by name at import time.

- RSNA: `Pneumonia` is NIH index 12, so the 14-class checkpoint is evaluated
  with `logit_mask=[12]` and `--num_classes 14`.
- VinDr-CXR classification (11 findings): `logit_mask=[0,1,2,3,4,5,6,8,9,10,11,13]`
  (`clarity.data.vindr.VINDR_CLS_LOGIT_MASK`); localization (6 box findings):
  `logit_mask=[0,1,4,8,9,10,13]` (`VINDR_LOC_LOGIT_MASK`). Both masks keep
  `Mass` (9) and `Nodule` (10); `--vindr-fuse-outputs` fuses them with `max`
  into the single VinDr `Nodule/Mass` finding. It is required for the masked
  checkpoint and refused if the model already emits the target label space.
- VinDr-only anatomical-region override: `Pleural effusion` also uses the
  `Facies Diaphragmatica` region (`VINDR_DISEASE_REGIONS`).

## Commands

A1 on VinDr-CXR at 224 px (use the checkpoint you trained, see
`docs/training.md`). `-alc` is the attention-layer count selected for that
checkpoint; it is explicit because a copied `model_best.pth` has no
`best-selection.json` beside it:

```bash
# classification, 11 findings
clarity-evaluate clarity s-16 -ds vindr -spl classify -s 224 \
  --vindr-splits-dir SPLITS -img PNG224 --data-root DATA \
  -cls 14 -mkw 'logit_mask=[0,1,2,3,4,5,6,8,9,10,11,13]' -alc 5 \
  --vindr-fuse-outputs -wg model_best.pth

# localization, 6 findings, frozen NIH calibration (A1, NIH-calibrated)
clarity-evaluate clarity s-16 -ds vindr -spl localize -s 224 \
  --vindr-splits-dir SPLITS -img PNG224 -seg MASKS224 --data-root DATA \
  -cls 14 -mkw 'logit_mask=[0,1,4,8,9,10,13]' -alc 5 \
  --vindr-fuse-outputs -wg model_best.pth \
  --frozen-onebox-calibration calibration/a1_onebox_nih_vindr6_224.json --arm a1
```

A2 (Target-calibrated) is the same localization command without the frozen
file (`--arm a2`). RSNA is the same shape with `-ds rsna`, `--rsna-splits-dir`,
`-mkw '...;logit_mask=[12]'` and `calibration/a1_onebox_nih_224.json`:

```bash
clarity-evaluate clarity s-16 -ds rsna -spl localize -s 224 \
  --rsna-splits-dir SPLITS -img PNG224 -seg MASKS224 --data-root DATA \
  -cls 14 -mkw 'img_size=224;n_attn_layers_avg=8;segment_cam_norm_heatmap=False;logit_mask=[12]' \
  -lckw "localization_method='model_attention';inference_attention_type='mask'" \
  -wg model_best.pth \
  --frozen-onebox-calibration calibration/a1_onebox_nih_224.json --arm a1
```

For 512 px use `-s 512`, the 512 checkpoint, and the `_512.json` calibration.

### Frozen calibrations

`calibration/` holds the NIH-fitted parameters for arm A1:

| file | target | resolution |
|---|---|---|
| `a1_onebox_nih_224.json`, `a1_onebox_nih_512.json` | RSNA (`Pneumonia`) | 224 / 512 |
| `a1_onebox_nih_vindr6_224.json`, `a1_onebox_nih_vindr6_512.json` | VinDr-CXR (6 box findings) | 224 / 512 |

Each is a schema-1 `localization_method` block (see `docs/artifacts.md`) with a
`target_label_map` keyed by the evaluation label index. They are fitted on NIH
validation data only; no RSNA or VinDr image, label or box was used.

### Baselines

```bash
# zero-shot TorchXRayVision DenseNet121-NIH (224 px PNGs)
clarity-baselines classify --baseline txv --dataset vindr -img PNG224 \
  --splits-dir SPLITS --data-root DATA -o out/txv-vindr.json
clarity-baselines localize --baseline txv --dataset vindr -img PNG224 -seg MASKS224 \
  --splits-dir SPLITS --data-root DATA -o out/txv-vindr-loc.json

# RAD-DINO probe (512 px PNGs); 'classify' writes the probe weights for 'localize'
clarity-baselines classify --baseline raddino --dataset vindr -img PNG512 \
  --splits-dir SPLITS --data-root DATA -o out/raddino-vindr.json
clarity-baselines localize --baseline raddino --dataset vindr -img PNG512 -seg MASKS512 \
  --splits-dir SPLITS --data-root DATA --probes out/raddino-vindr.probes.npz \
  -o out/raddino-vindr-loc.json
```

Use `--dataset rsna` for RSNA. Install the extras with `uv sync --extra transfer`
(HiResCAM, TorchXRayVision, the RAD-DINO weights through `transformers`, and
`pydicom` for the VinDr image-size fallback). `--limit N` runs a smoke test on
the first N validation/test images; the JSON records it, and such a run is
not a result.

## Metrics

- Classification: macro AUROC over the evaluable findings (a finding with no
  positive image in the test list, e.g. VinDr `Edema`, is undefined and left out
  of the macro).
- Localization: macro IoU accuracy on the 17-point score grid 0.10 to 0.90
  (and IoBB, reported but never used for selection), all-images protocol.
