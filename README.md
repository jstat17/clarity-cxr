# CLARiTy

`clarity-cxr` is the code companion to the CLARiTy paper:

> **CLARiTy: A Vision Transformer for Multi-Label Classification and
> Weakly-Supervised Localization of Chest X-ray Pathologies**
> John M. Statheros, Hairong Wang, Richard Klein.
> arXiv:2512.16700 (v2): <https://arxiv.org/abs/2512.16700>

CLARiTy (Class Localizing and Attention Refining Image Transformer) is a
vision-transformer model for multi-label chest X-ray classification with
weakly-supervised localization.

## Contents

- The CLARiTy-S-16 model and the ConvNeXtV2-B teacher model.
- The paper's loss implementations: the combined objective built from
  weighted binary cross-entropy (WBCE), background activation suppression
  (BAS) and soft Dice, plus the MLSM classification-loss ablation.
- ChestX-ray14 readers, the classification/localization evaluation protocols,
  the OneBox localization method, and the split-manifest loader/validator.
- The transfer study to RSNA Pneumonia and VinDr-CXR: the dataset readers with
  their frozen label maps, the two evaluation arms (A1 *NIH-calibrated*, A2
  *Target-calibrated*), the NIH-fitted A1 calibrations in `calibration/`, and
  the TorchXRayVision DenseNet121-NIH and RAD-DINO baselines read through
  HiResCAM. See `docs/transfer.md`.
- The command-line entry points used by the paper's training and evaluation
  pipeline:

  ```text
  clarity-train     clarity-evaluate     clarity-predict
  clarity-segment   clarity-prepare      clarity-baselines
  ```

- The recipes in `configs/`: CLARiTy-S-16 at 224 px (the 224 px
  model) and at 512 px (warm-started from the 224 model), plus the
  ConvNeXtV2-B teacher recipes.

## Install

Python 3.11 is required. The repository uses [uv](https://docs.astral.sh/uv/)
and ships a locked environment:

```bash
git clone https://github.com/jstat17/clarity-cxr.git
cd clarity-cxr
uv sync --extra data          # --extra data adds torchxrayvision for segmentation
uv sync --extra transfer      # RSNA / VinDr-CXR transfer study and baselines
```

The lock is pinned to the Python, PyTorch/CUDA and scientific-Python versions
used for the paper's environment; those are the only tested versions. `torch`
and `torchvision` resolve from the CUDA 13 PyTorch index declared in
`pyproject.toml`. CPU-only installations are possible with Torch, but are not
the tested release configuration. The optional `data` extra is only needed for
`clarity-segment`, and the `transfer` extra (HiResCAM, TorchXRayVision, RAD-DINO
through `transformers`, `pydicom`) only for `clarity-baselines` and HiResCAM
maps; training and evaluation on ChestX-ray14 use the main dependencies.

Pretrained initialisation (`imagenet` / `dino`) and the segmentation model use
their loader's download caches on first use. See `docs/data.md` for the data
layout and `docs/training.md` for the recipes.

Check the installation without a dataset:

```bash
uv run clarity-train --help
uv run clarity-evaluate --help
uv run python -c "import clarity; print(clarity.__version__)"
```

## Data

This code reads the public NIH ChestX-ray14 release. The metadata CSVs
(`Data_Entry_2017_v2020.csv`, `BBox_List_2017.csv`) and the extracted images
are expected under a root such as:

```text
<data-root>/
  ChestX-ray14/
    Data_Entry_2017_v2020.csv
    BBox_List_2017.csv
    images/
```

Set `CLARITY_DATA_ROOT=<data-root>` or pass `--data-root`. The official NIH
split is available out of the box. To use the paper's historical
classification/localization protocol, pass a directory containing the four
list files via `--splits-dir`. See `docs/data.md` for the exact file names and
the segmentation-mask workflow.

The transfer study additionally reads RSNA Pneumonia and VinDr-CXR, which you
obtain yourself (VinDr-CXR requires PhysioNet credentialed access). No data,
checkpoints, evaluation pickles or split lists are distributed with this
repository, and none of the three image sets can be redistributed;
`docs/transfer.md` describes the expected layout and the split-list format.

### Dataset links

| Dataset | Source | Link |
|---|---|---|
| ChestX-ray14 (NIH) | National Institutes of Health | <https://nihcc.app.box.com/v/ChestXray-NIHCC> |
| RSNA Pneumonia Detection Challenge | Kaggle | <https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge> |
| VinDr-CXR 1.0.0 | PhysioNet (credentialed access) | <https://physionet.org/content/vindr-cxr/1.0.0/> |

## Quickstart

```bash
# classification training at 224 px
uv run clarity-train --config configs/clarity_s16_224.yaml

# classification evaluation on the official test split
uv run clarity-evaluate clarity s-16 \
  --split classify --weights <checkpoint.pth>

# localization evaluation (requires the supplied split manifest)
uv run clarity-evaluate clarity s-16 \
  --split localize --weights <checkpoint.pth> \
  --splits-dir <splits-dir>
```

See `docs/training.md` and `docs/evaluation.md` for the 512 and teacher
recipes, attention-layer sweep options, and the exact localization protocol.

## License, citation and clinical disclaimer

This project is released under the GNU Affero General Public License v3.0
(AGPL-3.0-only). See `LICENSE` and `NOTICE`; the latter retains the
attributions for the timm and torchxrayvision components.

If you use this code, please cite the CLARiTy paper (and this repository, via
`CITATION.cff`):

```bibtex
@misc{statheros2025clarity,
  title         = {{CLARiTy}: A Vision Transformer for Multi-Label Classification and
                   Weakly-Supervised Localization of Chest X-ray Pathologies},
  author        = {Statheros, John M. and Wang, Hairong and Klein, Richard},
  year          = {2025},
  eprint        = {2512.16700},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2512.16700},
  note          = {Version 2}
}
```

This software is research code, not a medical device. It must not be used for
clinical diagnosis, treatment, or any decision affecting patient care without
independent validation and regulatory approval.
