"""VinDr-CXR reader.

Flattens the VinDr-CXR metadata into the same ``Filename`` / ``PatientID`` /
``Labels`` dataframe shape the ChestX-ray14 and RSNA readers produce, so
:class:`clarity.data.loader.MultilabelDataset` consumes it unchanged, and emits a
bounding-box dataframe in the ``BBox_List_2017`` shape (``Filename``,
``Label``, ``row_start``, ``row_end``, ``col_start``, ``col_end``).

The label mapping is **frozen** (see ``docs/transfer.md``); every index below
is derived from ``CXR_14_ABNORMALITIES`` **by name** at import time, never
hard-coded.

Dataset facts the contract depends on:

* 18,000 images: 15,000 ``train`` + 3,000 ``test``; the official test split is
  reported and never trained on;
* ``annotations_*.csv`` carry 22 real box classes in train and 21 in test
  (``Edema`` has no test boxes) plus coordinate-less ``No finding`` rows that
  are **not** boxes and must be filtered before parsing;
* ``image_labels_train.csv`` has three rows per image and a ``rad_id`` column
  (17 distinct raters), so the train consensus is any-rater-positive;
  ``image_labels_test.csv`` has one row per image and no ``rad_id``;
* the two splits spell one label differently (train ``Other diseases``, test
  ``Other disease``); both normalise to ``Other disease``;
* the DICOM headers carry **no patient identifier** (only sex/age/size/weight
  and pixel tags are retained), so rules split on the image id and
  ``PatientID`` is a pseudo-patient equal to the image id.

Nodule/Mass is one VinDr class but two NIH findings (``Mass`` 9, ``Nodule``
10). It is **fused** with ``max`` over post-activation values, once, at the
model-output boundary: :func:`fuse_vindr_outputs` implements that step and the
two masks below are deliberately one entry longer than the finding count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from clarity.data.chestxray14 import CXR_14_ABNORMALITIES, CXR_14_ABNORMALITY_REGIONS
from clarity.data.utils import read_file_lines

#: the 28 image-level label columns of ``image_labels_*.csv``, in CSV order,
#: canonicalised to the test spelling (train writes ``Other diseases``)
VINDR_IMAGE_LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other disease",
    "No finding",
]

#: the 22 VinDr box classes (``No finding`` is a placeholder, not a class, and
#: the five image-level-only labels never appear in ``class_name``)
VINDR_BOX_CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
]

#: VinDr finding -> NIH finding name(s); order is the frozen classification
#: order (ascending NIH index, fused finding at its lowest member index)
_MAPPING: list[tuple[str, tuple[str, ...], bool]] = [
    ("Atelectasis", ("Atelectasis",), True),
    ("Cardiomegaly", ("Cardiomegaly",), True),
    ("Consolidation", ("Consolidation",), False),
    ("Edema", ("Edema",), False),
    ("Pleural effusion", ("Effusion",), True),
    ("Emphysema", ("Emphysema",), False),
    ("Pulmonary fibrosis", ("Fibrosis",), False),
    ("Infiltration", ("Infiltration",), True),
    ("Nodule/Mass", ("Mass", "Nodule"), True),
    ("Pleural thickening", ("Pleural Thickening",), False),
    ("Pneumothorax", ("Pneumothorax",), True),
]

#: the 11 mapped findings, in the frozen classification-output order
VINDR_DISEASES = [name for name, _nih, _box in _MAPPING]
#: the 6 findings that are both mapped and box-supervised in NIH
VINDR_BOX_FINDINGS = [name for name, _nih, box in _MAPPING if box]
#: mapped findings with no NIH box supervision (classification-only transfer)
VINDR_CLASSIFICATION_ONLY = [name for name, _nih, box in _MAPPING if not box]
#: VinDr box findings with no NIH counterpart (reported, never scored)
VINDR_UNTRANSFERABLE_BOX = [
    label for label in VINDR_BOX_CLASSES if label not in set(VINDR_DISEASES)
]
#: findings with image-level labels but no boxes anywhere in the release
VINDR_IMAGE_ONLY_LABELS = ["COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other disease"]

#: VinDr finding -> NIH finding name(s); the one-to-many entries are fused
VINDR_NIH_LABELS = {name: nih_names for name, nih_names, _box in _MAPPING}

#: VinDr finding -> NIH index tuple, derived by name (asserted by a test)
VINDR_NIH_INDICES = {
    name: tuple(CXR_14_ABNORMALITIES.index(nih) for nih in nih_names)
    for name, nih_names, _box in _MAPPING
}

#: **VinDr-only** anatomical-region overrides.
#:
#: ChestX-Det's ``Left Lung``/``Right Lung`` masks stop at the lung base, while
#: VinDr effusion boxes are drawn at the costophrenic angle: only 0.327 of an
#: effusion box falls inside the frozen lung-field prior, and adding the
#: model's ``Facies Diaphragmatica`` region lifts that to 0.754 (measured on the
#: VinDr-CXR test boxes).
#:
#: This override is applied **here and nowhere else**: the shared
#: :data:`clarity.data.chestxray14.CXR_14_ABNORMALITY_REGIONS` map is the baseline
#: for every NIH training/evaluation run and is left byte-identical. A test
#: asserts that ("effusion region override is VinDr-only").
VINDR_REGION_OVERRIDES: dict[str, tuple[str, ...]] = {
    "Pleural effusion": ("Left Lung", "Right Lung", "Facies Diaphragmatica"),
}

def _union_regions(nih_names: tuple[str, ...]) -> list[str]:
    """Union of the NIH region lists, preserving the NIH order.

    First-seen order (not alphabetical) keeps the fused finding readable —
    ``[Left Lung, Right Lung, Mediastinum]`` rather than
    ``[Left Lung, Mediastinum, Right Lung]`` — and is deterministic because the
    mapping above is frozen.
    """
    regions: list[str] = []
    for nih in nih_names:
        for region in CXR_14_ABNORMALITY_REGIONS[nih]:
            if region not in regions:
                regions.append(region)
    return regions


#: VinDr finding -> anatomical regions: the NIH names by default (union for the
#: fused finding), with the VinDr-only overrides above applied
VINDR_DISEASE_REGIONS = {
    name: list(VINDR_REGION_OVERRIDES[name])
    if name in VINDR_REGION_OVERRIDES
    else _union_regions(nih_names)
    for name, nih_names, _box in _MAPPING
}

#: NIH finding -> VinDr finding (the inverse of the mapping above)
_NIH_TO_VINDR = {nih: name for name, nih_names, _box in _MAPPING for nih in nih_names}

#: the one-to-many mapping that must be fused rather than renamed
VINDR_FUSION = {"Nodule/Mass": ("Mass", "Nodule")}

#: Zero-shot masks over the 14-class checkpoint head (arms A1 NIH-calibrated and
#: A2 Target-calibrated, see ``docs/transfer.md``). Each mask is one
#: entry longer than its finding count: both Mass (9) and Nodule (10) survive so
#: the fusion has two operands.
VINDR_CLS_LOGIT_MASK = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13]
VINDR_LOC_LOGIT_MASK = [0, 1, 4, 8, 9, 10, 13]

#: output layout of the two evaluation label spaces. ``fuse_pairs`` index the
#: *masked* vector (the order the model emits after ``logit_mask``).
VINDR_OUTPUT_LAYOUT = {
    "classification": {
        "logit_mask": VINDR_CLS_LOGIT_MASK,
        "fuse_pairs": ((8, 9),),
        "labels": VINDR_DISEASES,
    },
    "localization": {
        "logit_mask": VINDR_LOC_LOGIT_MASK,
        "fuse_pairs": ((4, 5),),
        "labels": VINDR_BOX_FINDINGS,
    },
}

#: train/test spellings of the "no mapped finding" label that differ
_LABEL_ALIASES = {"Other diseases": "Other disease"}

#: the DICOM headers carry no patient id; ``PatientID`` is the image id
VINDR_PATIENT_ID_IS_IMAGE_ID = True

#: names of the split lists inside the directory passed as ``splits_dir``
SPLIT_FILES = {
    "train": "vindr_train_list.txt",
    "val": "vindr_val_list.txt",
    "test": "vindr_test_list.txt",
}

REQUIRED_COLUMNS = ("Filename", "PatientID", "Labels")

#: ``No finding`` rows are placeholders (coordinate-less), never boxes
NO_FINDING = "No finding"


def _annotation_path(vindr_path: Path, filename: str) -> Path:
    path = Path(vindr_path) / "annotations" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"VinDr-CXR annotation file not found: {path} (pass --data-root or set "
            "CXR_VINDR_PATH)"
        )
    return path


def _normalise_labels(columns) -> dict[str, str]:
    """Map a split's spelling differences onto the canonical label names."""
    return {column: _LABEL_ALIASES.get(column, column) for column in columns}


def _positive_labels(
    vindr_path: Path, split: str, *, consensus: str = "any"
) -> pd.DataFrame:
    """Boolean label frame (indexed by image id) for one raw split.

    Train collapses the three rater rows with the requested consensus; test is
    single-rater and only accepts ``consensus="any"``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"unknown VinDr split {split!r}; expected 'train' or 'test'")
    path = _annotation_path(vindr_path, f"image_labels_{split}.csv")
    df = pd.read_csv(path)
    df = df.rename(columns=_normalise_labels(df.columns))
    missing = [label for label in VINDR_IMAGE_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing label column(s): {missing}")
    if not df[VINDR_IMAGE_LABELS].isin((0, 1)).all().all():
        raise ValueError(f"{path} label columns must be binary (0/1)")

    if split == "train":
        counts = df.groupby("image_id")[VINDR_IMAGE_LABELS].sum()
        n_rows = df.groupby("image_id").size().reindex(counts.index)
        match consensus:
            case "any":
                positive = counts > 0
            case "majority":
                positive = counts.ge(n_rows // 2 + 1, axis=0)
            case _:
                raise ValueError(
                    f"unsupported consensus {consensus!r}; expected 'any' or 'majority'"
                )
    else:
        if consensus != "any":
            raise ValueError(
                f"test labels are single-rater; consensus {consensus!r} is not "
                "selectable on the test split"
            )
        if df["image_id"].duplicated().any():
            raise ValueError(f"{path} has more than one label row per image")
        positive = df.set_index("image_id")[VINDR_IMAGE_LABELS].astype(bool)

    return positive.sort_index()


def _label_splits(split: str) -> tuple[str, ...]:
    if split == "all":
        return ("train", "test")
    if split in ("train", "test"):
        return (split,)
    raise ValueError(
        f"unknown VinDr split {split!r}; expected 'train', 'test' or 'all' "
        "('all' = the evaluation view: train + official test)"
    )


def get_vindr_full_labels(vindr_path: Path, split: str = "train") -> pd.DataFrame:
    """All 28 image-level labels per image, one row per image.

    Train uses any-radiologist-positive consensus; test has a single row per
    image. ``split="all"`` concatenates the two (the evaluation view, because
    the official test images are labelled in their own CSV). This is the
    28-label view used for the "not transferable" tables — the transfer arms
    use :func:`get_vindr_dataframe`, which keeps only the 11 mapped findings.
    """
    frames = []
    for name in _label_splits(split):
        frame = _positive_labels(vindr_path, name).reset_index()
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    if labels["image_id"].duplicated().any():
        raise ValueError("train and test image ids overlap; the release changed")
    return labels


def get_vindr_dataframe(
    vindr_path: Path,
    split: str = "train",
    *,
    consensus: str = "any",
) -> pd.DataFrame:
    """Build the ``Filename``/``PatientID``/``Labels`` dataframe for VinDr.

    One row per image. ``Labels`` holds the positive findings **from the 11
    mapped findings only** (``VINDR_DISEASES``), in the frozen order; images
    with no mapped positive get an empty list (all-zero target), which is the
    honest encoding: such an image may still be positive for an untransferable
    VinDr finding. ``Filename`` is ``<image_id>.png``, the PNG converted from
    the DICOM ``<image_id>.dicom``.

    ``split="all"`` is the evaluation view: train (consensus) + official test
    (single-rater). The dataset then selects rows by filename, so one dataframe
    serves both the val and the test loader.

    ``consensus`` only binds on train (test has one row per image):
    ``"any"`` = any-radiologist-positive, ``"majority"`` = at least
    ``n_rows // 2 + 1`` raters.
    """
    splits = _label_splits(split)
    frames = []
    for name in splits:
        positive = _positive_labels(vindr_path, name, consensus=consensus)
        frame = pd.DataFrame(
            {
                "PatientID": positive.index,
                "Filename": [f"{image_id}.png" for image_id in positive.index],
            }
        )
        frame["Labels"] = [
            [label for label in VINDR_DISEASES if row[label]]
            for _image_id, row in positive.iterrows()
        ]
        frames.append(frame)

    dataframe = pd.concat(frames, ignore_index=True)
    if dataframe["Filename"].duplicated().any():
        raise ValueError("duplicate image ids across VinDr splits")
    return dataframe


def get_vindr_bbox_dataframe(
    vindr_path: Path,
    split: str = "train",
    image_size: int = 512,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    image_ids: set[str] | list[str] | None = None,
    images_path: Path | None = None,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Boxes in the ``BBox_List_2017`` shape, scaled to ``image_size``.

    ``No finding`` placeholder rows are dropped **before** any coordinate is
    parsed (train writes one per rater, and 128 train images carry both
    placeholders and real boxes). Every remaining ``(image, finding)`` row is
    kept, so a finding with several boxes keeps all of them; the evaluation
    path must then run with ``multi_box_gt=True``.

    ``split="all"`` concatenates the train and test annotation tables (the
    evaluation view). ``image_ids`` restricts the result to the images a run
    actually needs (e.g. val + test), which also avoids requiring every train
    image to be converted before a test-only evaluation.

    VinDr images are non-square and vary in size, so the x/y scales differ per
    image: ``col * image_size / Columns`` and ``row * image_size / Rows``.
    Sizes come from ``image_sizes`` when given, else from the conversion
    metadata beside ``images_path`` (``<dir>_conversion_meta.json``, optional),
    else from the DICOM headers (needs ``pydicom``).

    ``labels`` defaults to the 6 localization findings; pass another subset of
    ``VINDR_BOX_CLASSES`` (e.g. all 22) for scope tables.
    """
    wanted = tuple(VINDR_BOX_FINDINGS if labels is None else labels)
    unknown = [label for label in wanted if label not in VINDR_BOX_CLASSES]
    if unknown:
        raise ValueError(f"labels are not VinDr box classes: {unknown}")

    frames = []
    for name in _label_splits(split):
        path = _annotation_path(vindr_path, f"annotations_{name}.csv")
        frame = pd.read_csv(path)
        missing = [
            column
            for column in ("image_id", "class_name", "x_min", "y_min", "x_max", "y_max")
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}")
        frame["_split"] = name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    # placeholders first: they have no coordinates and must never become boxes
    placeholder = df["class_name"].eq(NO_FINDING)
    df = df.loc[~placeholder].copy()
    if df.empty:
        raise ValueError(f"VinDr {split} annotations contain no real boxes")
    df = df.loc[df["class_name"].isin(wanted)].copy()
    if df.empty:
        raise ValueError(f"VinDr {split} annotations contain no boxes for {list(wanted)}")
    if image_ids is not None:
        keep = {str(image_id) for image_id in image_ids}
        df = df.loc[df["image_id"].astype(str).isin(keep)].copy()
        if df.empty:
            raise ValueError("no boxes remain after the image_ids filter")

    if df[["x_min", "y_min", "x_max", "y_max"]].isna().any().any():
        raise ValueError("a non-placeholder VinDr row has empty coordinates")

    sizes = dict(image_sizes) if image_sizes is not None else None
    if sizes is None:
        sizes = get_vindr_image_sizes(
            vindr_path,
            split,
            images_path=images_path,
            # only the images this run loads: a test-only evaluation must not
            # need every train DICOM to be downloaded
            image_ids=set(df["image_id"].astype(str)),
        )
    absent = sorted(set(df["image_id"].astype(str)) - set(sizes))
    if absent:
        raise ValueError(
            f"no image size available for {len(absent)} image(s), e.g. {absent[:5]}; "
            "pass image_sizes= or images_path= pointing at the converted PNGs"
        )

    rows = df["image_id"].astype(str).map(lambda image_id: sizes[image_id][0]).to_numpy(dtype=float)
    cols = df["image_id"].astype(str).map(lambda image_id: sizes[image_id][1]).to_numpy(dtype=float)
    if (rows <= 0).any() or (cols <= 0).any():
        raise ValueError("image sizes must be positive")

    df["Filename"] = df["image_id"].astype(str) + ".png"
    df["Label"] = df["class_name"]
    df["row_start"] = df["y_min"].astype(float) * (image_size / rows)
    df["row_end"] = df["y_max"].astype(float) * (image_size / rows)
    df["col_start"] = df["x_min"].astype(float) * (image_size / cols)
    df["col_end"] = df["x_max"].astype(float) * (image_size / cols)

    box_columns = ["row_start", "row_end", "col_start", "col_end"]
    values = df[box_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("VinDr annotations contain non-finite box coordinates")
    tolerance = 1e-6 * image_size
    if (values < -tolerance).any() or (values > image_size + tolerance).any():
        raise ValueError(
            f"boxes outside the {image_size}px frame after scaling; check that "
            "image sizes match the annotation coordinate space"
        )

    return df[["Filename", "Label", *box_columns]].reset_index(drop=True)


def _sizes_from_conversion_meta(images_path: Path) -> dict[str, tuple[int, int]] | None:
    """Read per-image ``(rows, cols)`` from a ``dicom_to_png`` metadata file."""
    images_path = Path(images_path)
    meta_path = images_path.parent / f"{images_path.name}_conversion_meta.json"
    if not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text())
    sizes: dict[str, tuple[int, int]] = {}
    for record in payload.get("files", []):
        image_id = Path(record["source"]).stem
        shape = record.get("original_shape")
        if shape is None:
            continue
        sizes[image_id] = (int(shape[0]), int(shape[1]))
    return sizes or None


def get_vindr_image_sizes(
    vindr_path: Path,
    split: str = "train",
    *,
    images_path: Path | None = None,
    image_ids: set[str] | list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Native ``(rows, cols)`` for every annotated image of a split.

    Prefers the conversion metadata beside ``images_path`` (free, and the same
    shapes the PNGs were written from); any image the metadata does not cover
    falls back to its DICOM header, so a partially converted directory still
    yields complete sizes for the images that are on disk.

    ``pydicom`` is imported only when that fallback is actually needed: a fully
    converted tree with its metadata must not require the DICOM reader (an
    environment with the PNGs and the metadata but no pydicom).
    """
    wanted = None if image_ids is None else {str(image_id) for image_id in image_ids}
    sizes: dict[str, tuple[int, int]] = {}
    if images_path is not None:
        sizes.update(_sizes_from_conversion_meta(Path(images_path)) or {})

    for name in _label_splits(split):
        dicom_dir = Path(vindr_path) / name
        labels = _annotation_path(vindr_path, f"annotations_{name}.csv")
        split_ids = sorted(
            pd.read_csv(labels, usecols=["image_id"])["image_id"].astype(str).unique()
        )
        if wanted is not None:
            split_ids = [image_id for image_id in split_ids if image_id in wanted]
        for image_id in split_ids:
            if image_id in sizes:
                continue
            path = dicom_dir / f"{image_id}.dicom"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing DICOM for annotated image {image_id}: {path} "
                    "(the download is incomplete and the image is not converted?)"
                )
            import pydicom

            header = pydicom.dcmread(
                str(path), stop_before_pixels=True, specific_tags=["Rows", "Columns"]
            )
            sizes[image_id] = (int(header.Rows), int(header.Columns))
    return sizes


def _split_file(splits_dir: Path, split: str) -> Path:
    try:
        filename = SPLIT_FILES[split]
    except KeyError as exc:
        raise KeyError(
            f"unknown VinDr split {split!r}; expected one of {sorted(SPLIT_FILES)}"
        ) from exc
    path = Path(splits_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"VinDr {split} split list not found: {path}; generate the lists with "
            "supply the three lists (one filename per line), see docs/transfer.md"
        )
    return path


def get_split_filenames(
    splits_dir: Path,
    available_images_path: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return the VinDr train/validation/test filename lists.

    When ``available_images_path`` is given, filenames absent from that image
    directory are dropped, exactly like the ChestX-ray14 and RSNA split
    readers.
    """
    available_filenames = None
    if available_images_path is not None:
        if not Path(available_images_path).is_dir():
            raise FileNotFoundError(f"VinDr image directory not found: {available_images_path}")
        available_filenames = set(os.listdir(available_images_path))

    splits = []
    for split in ("train", "val", "test"):
        filenames = read_file_lines(_split_file(splits_dir, split))
        if available_filenames is not None:
            filenames = [name for name in filenames if name in available_filenames]
        splits.append(filenames)

    return splits[0], splits[1], splits[2]


def count_split_filenames(splits_dir: Path, split: str) -> int:
    """Number of filenames listed in a committed split file (before filtering)."""
    return len(read_file_lines(_split_file(splits_dir, split)))


def get_vindr_localization_filenames(
    vindr_path: Path,
    split: str = "train",
    *,
    images_path: Path | None = None,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Filenames that carry at least one box of the 6 localization findings."""
    df_bbox = get_vindr_bbox_dataframe(
        vindr_path, split, images_path=images_path, image_sizes=image_sizes
    )
    return df_bbox["Filename"].drop_duplicates().tolist()


def patient_id(filename: str) -> str:
    """Return the group key used for splitting: the **image id**.

    VinDr's DICOM headers carry no patient identifier, so ``PatientID`` in
    :func:`get_vindr_dataframe` is the image id and a split on it is
    image-disjoint, not patient-disjoint. Named ``patient_id`` to keep the
    split-reader signature uniform; the caveat is the point.
    """
    return Path(filename).stem


def fuse_vindr_outputs(
    outputs,
    label_space: str = "classification",
):
    """Fuse ``Mass`` and ``Nodule`` into one ``Nodule/Mass`` output.

    ``outputs`` is the masked model output with the class dimension at axis 1
    (``[batch, classes]`` logits or ``[batch, classes, height, width]``
    heatmaps). The expected width is the length of the label space's logit mask
    (12 classification, 7 localization); the result has one fewer class, with
    the fused value ``max(a, b)`` at the first member's position.

    ``max`` is applied to the tensors as given: for logits this is equivalent
    to ``max(sigmoid(a), sigmoid(b))`` because sigmoid is monotone, and
    heatmaps are already post-activation. Apply exactly once — the caller must
    not fuse a second time.
    """
    import torch as th

    if label_space not in VINDR_OUTPUT_LAYOUT:
        raise ValueError(
            f"unknown VinDr label space {label_space!r}; expected one of "
            f"{sorted(VINDR_OUTPUT_LAYOUT)}"
        )
    layout = VINDR_OUTPUT_LAYOUT[label_space]
    expected = len(layout["logit_mask"])
    if outputs.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {tuple(outputs.shape)}")
    if outputs.shape[1] != expected:
        raise ValueError(
            f"{label_space}: expected {expected} masked outputs "
            f"(logit_mask={layout['logit_mask']}), got {outputs.shape[1]}"
        )

    fused = outputs
    # consume pairs from the highest index down, so earlier positions stay valid
    for low, high in sorted(layout["fuse_pairs"], reverse=True):
        if high != low + 1:
            raise ValueError(
                f"fuse pair {(low, high)} is not adjacent; the adapter assumes "
                "adjacent members (Mass is emitted immediately before Nodule)"
            )
        merged = th.maximum(fused[:, low], fused[:, high]).unsqueeze(1)
        fused = th.cat([fused[:, :low], merged, fused[:, high + 1 :]], dim=1)

    return fused


@dataclass(frozen=True)
class FindingOutputPlan:
    """How to map a model's class axis onto an evaluation label space.

    ``fuse_pairs`` are positions in the **emitted** vector that must be fused
    (VinDr only ever has one pair: Mass/Nodule). ``keep`` are positions in the
    **post-fusion** vector, in target order, so ``index_select`` produces the
    label space the dataset and the calibration use.
    """

    label_space: str
    emitted_width: int
    fuse_pairs: tuple[tuple[int, int], ...]
    keep: tuple[int, ...]
    targets: tuple[str, ...]

    @property
    def is_identity(self) -> bool:
        return not self.fuse_pairs and self.keep == tuple(range(len(self.targets)))


def finding_output_plan(
    label_space: str, *, logit_mask: list[int] | tuple[int, ...] | None = None
) -> FindingOutputPlan:
    """Derive the adapter plan for a VinDr evaluation label space.

    Two model shapes occur:

    * **Arms A1 (NIH-calibrated) and A2 (Target-calibrated), 14-class
      checkpoint + ``logit_mask``**: the emitted classes are the masked NIH indices, so ``Mass`` and ``Nodule`` appear twice and are
      fused; the mask must cover every target finding.
    * **An 11-class VinDr head, no mask**: the emitted classes are
      ``VINDR_DISEASES`` in order; localization selects the 6 box findings.

    The plan is derived from the frozen mapping, never hand-written, so a mask
    that forgets a finding fails here instead of silently mis-indexing later.
    """
    if label_space not in VINDR_OUTPUT_LAYOUT:
        raise ValueError(
            f"unknown VinDr label space {label_space!r}; expected one of "
            f"{sorted(VINDR_OUTPUT_LAYOUT)}"
        )
    targets = tuple(VINDR_OUTPUT_LAYOUT[label_space]["labels"])

    if logit_mask is None:
        emitted = list(VINDR_DISEASES)
    else:
        mask = list(logit_mask)
        if not mask:
            raise ValueError("logit_mask is empty")
        if any(not 0 <= index < len(CXR_14_ABNORMALITIES) for index in mask):
            raise ValueError(f"logit_mask indices out of range: {mask}")
        emitted = [_NIH_TO_VINDR[CXR_14_ABNORMALITIES[index]] for index in mask]

    fuse_pairs: list[tuple[int, int]] = []
    chosen: list[int] = []
    for target in targets:
        positions = [index for index, name in enumerate(emitted) if name == target]
        if not positions:
            raise ValueError(
                f"{label_space}: finding {target!r} is not emitted by this model "
                f"(emitted={emitted}); check the logit_mask"
            )
        if len(positions) == 1:
            if logit_mask is not None and target in VINDR_FUSION:
                raise ValueError(
                    f"{label_space}: {target!r} is declared as a fusion of "
                    f"{VINDR_FUSION[target]} but the model emits it once; the "
                    "logit_mask must keep both members"
                )
            chosen.append(positions[0])
            continue
        if len(positions) > 2:
            raise ValueError(
                f"{label_space}: finding {target!r} is emitted {len(positions)} times; "
                "only a one-to-many Nodule/Mass split is supported"
            )
        low, high = positions
        if high != low + 1:
            raise ValueError(
                f"{label_space}: {target!r} members at {positions} are not adjacent; "
                "the adapter cannot fuse them positionally"
            )
        if set(VINDR_FUSION.get(target, ())) != set(VINDR_NIH_LABELS[target]):
            raise ValueError(
                f"{label_space}: {target!r} is emitted twice but the frozen mapping "
                f"does not declare a fusion for it"
            )
        if any(high == pair[1] for pair in fuse_pairs):
            raise ValueError(f"{label_space}: position {high} fused twice")
        fuse_pairs.append((low, high))
        chosen.append(low)

    # positions shift when a fused member is removed
    def _adjust(position: int) -> int:
        for low, high in fuse_pairs:
            if position > high:
                position -= 1
            elif low < position <= high:
                raise ValueError(
                    f"{label_space}: target position {position} lies inside a fusion pair"
                )
        return position

    keep = tuple(_adjust(position) for position in chosen)
    if logit_mask is not None:
        # the masked A1/A2 path must emit the target order exactly; a
        # reordered mask would silently permute findings if this were allowed
        expected_keep = tuple(range(len(targets)))
        if keep != expected_keep:
            raise ValueError(
                f"{label_space}: derived keep positions {keep} are not the identity "
                f"{expected_keep}; the model's class order does not match the target order"
            )

    layout_pairs = tuple(tuple(pair) for pair in VINDR_OUTPUT_LAYOUT[label_space]["fuse_pairs"])
    if fuse_pairs and tuple(sorted(fuse_pairs)) != layout_pairs:
        raise ValueError(
            f"{label_space}: derived fuse pairs {fuse_pairs} disagree with the frozen "
            f"layout {layout_pairs}"
        )

    return FindingOutputPlan(
        label_space=label_space,
        emitted_width=len(emitted),
        fuse_pairs=tuple(sorted(fuse_pairs)),
        keep=keep,
        targets=targets,
    )


def apply_finding_output_plan(outputs, plan: FindingOutputPlan):
    """Apply ``plan`` to a ``[batch, classes, ...]`` output tensor."""
    if outputs.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got shape {tuple(outputs.shape)}")
    if outputs.shape[1] != plan.emitted_width:
        raise ValueError(
            f"{plan.label_space}: expected {plan.emitted_width} emitted outputs, "
            f"got {outputs.shape[1]}"
        )
    fused = outputs
    if plan.fuse_pairs:
        fused = fuse_vindr_outputs(fused, plan.label_space)
    if plan.keep != tuple(range(fused.shape[1])):
        index = fused.new_tensor(plan.keep, dtype=th.long)
        fused = fused.index_select(1, index)
    return fused


def expected_output_count(label_space: str) -> int:
    """Number of findings the label space produces after fusion."""
    return len(VINDR_OUTPUT_LAYOUT[label_space]["labels"])


def validate_splits(
    train: list[str],
    val: list[str],
    test: list[str],
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Assert the three lists are disjoint and (optionally) sized as expected.

    There is no patient key in the VinDr release, so this checks images only.
    The caller is responsible for the documented limitation that a patient
    could in principle appear in two splits (the DICOM headers carry no patient
    identifier); do not claim patient disjointness from this function.
    """
    counts = {"train": len(train), "val": len(val), "test": len(test)}
    if expected_counts is not None:
        wrong = {
            name: (counts[name], expected)
            for name, expected in expected_counts.items()
            if counts.get(name) != expected
        }
        if wrong:
            details = ", ".join(
                f"{name}: got {got}, expected {want}" for name, (got, want) in sorted(wrong.items())
            )
            raise ValueError(f"VinDr split count mismatch: {details}")

    sets = {"train": set(train), "val": set(val), "test": set(test)}
    names = list(sets)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = sets[name_a] & sets[name_b]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"VinDr image overlap between {name_a} and {name_b}: {sample}")

    return counts
