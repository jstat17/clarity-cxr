"""Label-matrix, teacher-label and bounding-box helpers for the dataset."""
from __future__ import annotations

import pickle as pk
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from clarity.data import transform
from clarity.data.utils import get_iter_to_iter_dict


def build_label_matrix(
    df: pd.DataFrame,
    filenames: list[str],
    possible_labels: list[str],
    use_background_label: bool,
    split: str,
) -> np.ndarray:
    """Build the ``(n_images, n_labels)`` multi-hot label matrix.

    ``possible_labels`` is the label superset; with ``use_background_label`` a
    leading all-ones column is prepended.
    """
    n = len(filenames)
    n_labels = len(possible_labels) + int(use_background_label)
    labels = np.zeros(shape=(n, n_labels), dtype=np.float32)

    df_indexed = df.set_index("Filename").loc[filenames]
    filename_to_labels = get_iter_to_iter_dict(
        iter1=filenames, iter2=df_indexed["Labels"]
    )

    pbar = tqdm(
        total=n, desc=f"Loading {split} image labels into memory", unit="label"
    )
    for idx in range(n):
        vec = transform.encode_multiclass_one_hot(
            possible_labels=possible_labels,
            labels=filename_to_labels[filenames[idx]],
        )
        if use_background_label:
            vec = np.insert(vec, 0, 1)
        labels[idx] = vec
        pbar.update(1)
    pbar.close()

    return labels


def load_teacher_labels(
    teacher_labels_path: Path,
    epoch: int,
    filenames: list[str],
    shape: tuple[int, int],
) -> th.Tensor:
    """Load one epoch's precomputed teacher labels as a tensor."""
    teacher_labels_filename = f"epoch_{epoch:03}.pk"
    with open(teacher_labels_path / teacher_labels_filename, "rb") as f:
        teacher_labels_dict = pk.load(f)

    teacher_labels = np.zeros(shape=shape, dtype=np.float32)
    for idx, filename in enumerate(filenames):
        teacher_labels[idx] = teacher_labels_dict[filename]

    return th.from_numpy(teacher_labels)


def build_bbox_dict(
    df_bbox: pd.DataFrame,
    possible_labels: list[str],
    use_background_label: bool,
    multi_box: bool = False,
) -> dict[str, dict[int, dict[str, float] | list[dict[str, float]]]]:
    """Map ``filename (no extension) -> label index -> ground-truth box(es)``.

    With ``multi_box=False`` (the default, and the historical behaviour) a
    repeated ``(image, finding)`` row overwrites the previous one, so the value is
    a single box dict. That is correct for ChestX-ray14, whose
    ``BBox_List_2017.csv`` holds at most one box per ``(image, finding)`` pair,
    and every published CXR14 number depends on it.

    With ``multi_box=True`` the value is a **list** of boxes preserving the CSV
    order, which is what RSNA needs: the paper's test split has 4,229 boxes over
    2,589 positive images (1.63/image), so the overwrite would silently drop
    1,640 boxes and make any AP-style metric incomparable with the challenge.

    Consumers must accept both forms; use :func:`as_box_list` rather than
    indexing the value directly.
    """
    df_bbox = df_bbox.copy()
    df_bbox["Filename"] = df_bbox["Filename"].str[:-4]

    abnorm_to_idx = {
        abnorm: i + int(use_background_label)
        for i, abnorm in enumerate(possible_labels)
    }
    df_bbox["Label Index"] = df_bbox["Label"].map(abnorm_to_idx)

    image_abnorm_bboxes = defaultdict(dict)
    for filename in df_bbox["Filename"].unique():
        filename_data = df_bbox[df_bbox["Filename"] == filename]
        for _row_idx, row in filename_data.iterrows():
            box = {
                "row_start": row["row_start"],
                "row_end": row["row_end"],
                "col_start": row["col_start"],
                "col_end": row["col_end"],
            }
            label_index = row["Label Index"]
            if multi_box:
                image_abnorm_bboxes[filename].setdefault(label_index, []).append(box)
            else:
                image_abnorm_bboxes[filename][label_index] = box

    return image_abnorm_bboxes


def as_box_list(gt) -> list[dict[str, float]]:
    """Normalise a ground-truth box value into a list of box dicts.

    Accepts a single box dict (the historical contract), a sequence of box
    dicts (``multi_box=True``), or ``None``/empty -> ``[]``. Centralising this
    is what lets the multi-box path be additive: a consumer that calls it sees
    the same one-element list for ChestX-ray14 as before.
    """
    if gt is None:
        return []
    if isinstance(gt, dict):
        return [gt]
    return [box for box in gt if box is not None]
