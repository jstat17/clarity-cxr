"""RSNA Pneumonia Detection Challenge reader.

The reader flattens the challenge metadata into the same ``Filename`` /
``PatientID`` / ``Labels`` dataframe shape the ChestX-ray14 reader produces, so
:class:`clarity.data.loader.MultilabelDataset` consumes it unchanged, and it emits a
bounding-box dataframe in the ``BBox_List_2017`` shape (``Filename``,
``Label``, ``row_start``, ``row_end``, ``col_start``, ``col_end``).

Dataset facts that the loader contract depends on:

* 26,684 labelled images, one DICOM/one patient each, 1024x1024 at ~0.14 mm
  pixel spacing (inside NIH's native range);
* 9,555 boxes over 6,012 images; the other 20,672 images are negatives;
* the challenge CSV stores several rows for a positive image — one per box —
  and one all-NaN row for a negative image;
* the 3,000 ``stage_2_test_images`` carry no local labels and are never read.

Only the pneumonia finding transfers to the CLARiTy label space: pneumonia is
index 12 of ``CXR_14_ABNORMALITIES``.  :data:`RSNA_PNEUMONIA_NIH_INDEX` is
derived from the name at import time, never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from clarity.data.chestxray14 import CXR_14_ABNORMALITIES, CXR_14_ABNORMALITY_REGIONS
from clarity.data.utils import read_file_lines

#: the single abnormality this dataset annotates, in the reader's own label space
RSNA_DISEASES = ["Pneumonia"]
RSNA_DISEASE_REGIONS = {"Pneumonia": list(CXR_14_ABNORMALITY_REGIONS["Pneumonia"])}
#: index of pneumonia in the ChestX-ray14 label space (asserted by a test)
RSNA_PNEUMONIA_NIH_INDEX = CXR_14_ABNORMALITIES.index("Pneumonia")

#: native image size of every RSNA DICOM (and of the converted PNGs)
RSNA_IMAGE_SIZE = 1024

#: columns of ``stage_2_train_labels.csv`` we depend on
TRAIN_LABELS_FILE = "stage_2_train_labels.csv"
DETAILED_CLASS_FILE = "stage_2_detailed_class_info.csv"
TRAIN_IMAGES_DIR = "stage_2_train_images"
TEST_IMAGES_DIR = "stage_2_test_images"

#: names of the split lists inside the directory passed as ``splits_dir``
SPLIT_FILES = {
    "train": "rsna_train_list.txt",
    "val": "rsna_val_list.txt",
    "test": "rsna_test_list.txt",
}

REQUIRED_COLUMNS = ("Filename", "PatientID", "Labels")


def get_rsna_dataframe(rsna_path: Path) -> pd.DataFrame:
    """Build the ``Filename``/``PatientID``/``Labels`` dataframe for RSNA.

    One row per image.  ``Labels`` is ``["Pneumonia"]`` when the challenge
    labels the image positive for lung opacity and ``["No Finding"]``
    otherwise; ``Filename`` is ``<patientId>.png``, the PNG converted from the
    challenge DICOM.
    """
    rsna_path = Path(rsna_path)
    labels_path = rsna_path / TRAIN_LABELS_FILE
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"RSNA labels file not found: {labels_path} (pass --data-root or set "
            "CXR_RSNA_PATH)"
        )

    df_labels = pd.read_csv(labels_path)
    missing = [column for column in ("patientId", "Target") if column not in df_labels.columns]
    if missing:
        raise ValueError(f"{labels_path} is missing required column(s): {missing}")

    # one row per image: a positive image contributes one row per box, so the
    # positive set must be de-duplicated before it is turned into labels
    positive_ids = set(df_labels.loc[df_labels["Target"] == 1, "patientId"].unique())
    all_ids = pd.unique(df_labels["patientId"])

    df = pd.DataFrame(
        {
            "PatientID": all_ids,
            "Filename": [f"{patient_id}.png" for patient_id in all_ids],
        }
    )
    df["Labels"] = [
        list(RSNA_DISEASES) if patient_id in positive_ids else ["No Finding"]
        for patient_id in all_ids
    ]
    df["Rows"] = RSNA_IMAGE_SIZE
    df["Columns"] = RSNA_IMAGE_SIZE

    # the on-disk test images carry no local labels; never include them
    df = df.reset_index(drop=True)
    return df


def get_rsna_detailed_class_dataframe(rsna_path: Path) -> pd.DataFrame:
    """The challenge's 3-class view (``Normal`` / ``No Lung Opacity…`` / ``Lung Opacity``).

    Not used by the transfer arms; exposed so a further analysis does not have
    to re-derive it.
    """
    path = Path(rsna_path) / DETAILED_CLASS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"RSNA detailed class file not found: {path}")
    df = pd.read_csv(path)
    expected = {"patientId", "class"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} is missing required column(s): {sorted(expected - set(df.columns))}")
    return df


def get_rsna_bbox_dataframe(rsna_path: Path, image_size: int = RSNA_IMAGE_SIZE) -> pd.DataFrame:
    """Boxes in the ``BBox_List_2017`` shape, scaled to ``image_size``.

    The challenge stores ``x``, ``y``, ``width``, ``height`` in pixels of the
    native 1024x1024 image with the origin at the top-left, i.e. the same
    convention as the ChestX-ray14 ``Bbox [x``/``y``/``w``/``h]`` columns.
    """
    rsna_path = Path(rsna_path)
    labels_path = rsna_path / TRAIN_LABELS_FILE
    if not labels_path.is_file():
        raise FileNotFoundError(f"RSNA labels file not found: {labels_path}")

    df_labels = pd.read_csv(labels_path)
    df_bbox = df_labels[df_labels["Target"] == 1].copy()
    if df_bbox.empty:
        raise ValueError(f"{labels_path} contains no positive boxes")

    scale = image_size / RSNA_IMAGE_SIZE
    df_bbox["Filename"] = df_bbox["patientId"].astype(str) + ".png"
    df_bbox["Label"] = RSNA_DISEASES[0]
    df_bbox["row_start"] = df_bbox["y"].astype(float) * scale
    df_bbox["col_start"] = df_bbox["x"].astype(float) * scale
    df_bbox["row_end"] = (df_bbox["y"] + df_bbox["height"]).astype(float) * scale
    df_bbox["col_end"] = (df_bbox["x"] + df_bbox["width"]).astype(float) * scale

    return df_bbox[["Filename", "Label", "row_start", "row_end", "col_start", "col_end"]].reset_index(
        drop=True
    )


def get_rsna_localization_filenames(rsna_path: Path) -> list[str]:
    """Filenames that carry at least one box."""
    return get_rsna_bbox_dataframe(rsna_path)["Filename"].drop_duplicates().tolist()


def _split_file(splits_dir: Path, split: str) -> Path:
    try:
        filename = SPLIT_FILES[split]
    except KeyError as exc:
        raise KeyError(f"unknown RSNA split {split!r}; expected one of {sorted(SPLIT_FILES)}") from exc
    path = Path(splits_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"RSNA {split} split list not found: {path}; generate the lists with "
            "supply the three lists (one filename per line), see docs/transfer.md"
        )
    return path


def get_split_filenames(
    splits_dir: Path,
    available_images_path: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return the RSNA train/validation/test filename lists.

    When ``available_images_path`` is given, filenames absent from that image
    directory are dropped, exactly like the ChestX-ray14 split
    readers.
    """
    available_filenames = None
    if available_images_path is not None:
        if not Path(available_images_path).is_dir():
            raise FileNotFoundError(f"RSNA image directory not found: {available_images_path}")
        available_filenames = set(os.listdir(available_images_path))

    splits = []
    for split in ("train", "val", "test"):
        filenames = read_file_lines(_split_file(splits_dir, split))
        if available_filenames is not None:
            filenames = [name for name in filenames if name in available_filenames]
        splits.append(filenames)

    return splits[0], splits[1], splits[2]


def patient_id(filename: str) -> str:
    """Patient identifier encoded in an RSNA filename (the stem)."""
    return Path(filename).stem


def assert_one_image_per_patient(df: pd.DataFrame) -> None:
    """Assert the RSNA one-image-per-patient property the split relies on.

    An image-level split is only patient-disjoint because RSNA stores one
    image per patient.  The property is checkable from the label table (a
    positive image contributes one row *per box*, so rows are counted per
    ``patientId``) but not from the filenames, hence this guard sits here and
    is called by the split generator before any list is written.
    """
    if df["PatientID"].duplicated().any():
        duplicated = sorted(df.loc[df["PatientID"].duplicated(), "PatientID"].unique())
        raise ValueError(
            "RSNA no longer has one image per patient "
            f"({len(duplicated)} repeated patient id(s), e.g. {duplicated[:5]}); the "
            "image-level split is then not patient-disjoint and needs a patient "
            "map before it can be used"
        )
    if df["Filename"].str[:-4].ne(df["PatientID"].astype(str)).any():
        raise ValueError("RSNA filenames are expected to be '<patientId>.png'")


def validate_splits(
    train: list[str],
    val: list[str],
    test: list[str],
    *,
    expected_counts: dict[str, int] | None = None,
    patient_ids: dict[str, str] | None = None,
) -> dict[str, int]:
    """Assert the three lists are disjoint (in images and in patients).

    RSNA has exactly one image per patient, so an image-level split is
    patient-disjoint by construction; the check still runs per patient id so a
    later re-derivation cannot silently break the property.  ``patient_ids``
    supplies an explicit filename→patient map when one exists (VinDr does, and
    it reuses this helper); without it the filename stem is the patient.
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
            raise ValueError(f"RSNA split count mismatch: {details}")

    sets = {"train": set(train), "val": set(val), "test": set(test)}
    names = list(sets)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = sets[name_a] & sets[name_b]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"RSNA image overlap between {name_a} and {name_b}: {sample}")

    lookup = patient_ids or {}
    patient_sets = {
        name: {lookup.get(value) or patient_id(value) for value in values}
        for name, values in sets.items()
    }
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = patient_sets[name_a] & patient_sets[name_b]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"RSNA patient overlap between {name_a} and {name_b}: {sample}")

    return counts
