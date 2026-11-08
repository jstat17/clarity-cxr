import logging
import gzip
import os
import tarfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections.abc import Iterable
from typing import Any


logger = logging.getLogger(__name__)
def intersect_iters(iter1: Iterable, iter2: Iterable) -> bool:
    """Determine if two iterables have any common entries

    Args:
        iter1 (Iterable): First iterable
        iter2 (Iterable): Second iterable

    Returns:
        bool: If the two iterables have common entries
    """
    intersection = set(iter1) & set(iter2)
    return bool(intersection)


def get_iter_to_iter_dict(iter1: Iterable[Any], iter2: Iterable[Any]) -> dict[Any, Any]:
    """Get a dictionary from iter1 values to iter2 values

    Args:
        iter1 (Iterable[Any]): First iterable (dictionary keys)
        iter2 (Iterable[Any]): Second iterable (dictionary values)

    Returns:
        dict[Any, Any]: Dictionary from iter1 values to iter2 values
    """
    mapping = dict()
    for iter1_val, iter2_val in zip(iter1, iter2):
        mapping[iter1_val] = iter2_val

    return mapping


def extract_files_from_gzips(gzips_path: Path, extract_path: Path) -> None:
    """Extract all files from gzip-compressed files

    Args:
        gzips_path (Path): Path to gzips
        extract_path (Path): Desired extract path for all files
    """
    os.makedirs(extract_path, exist_ok=True)
    existing_filenames = set(os.listdir(extract_path))
    gzips_filenames = os.listdir(gzips_path)

    # iterate through all gzips
    for filename in gzips_filenames:
        gzip_path = gzips_path / filename
        n_extracted = 0

        # read and decompress the gzip file into memory
        with gzip.open(gzip_path, "rb") as gz:
            # open the tar file from the decompressed gzip content
            with tarfile.open(fileobj=gz, mode="r:") as tar:
                for tar_member in tar.getmembers():
                    # get the base name of the file, which removes folder structure
                    tar_member_filename = Path(tar_member.name).name

                    # extract tar member if it has not been extracted before
                    if tar_member_filename not in existing_filenames:
                        file_obj = tar.extractfile(tar_member)

                        if file_obj is not None:
                            image_path = extract_path / tar_member_filename
                            with open(image_path, "wb") as out_file:
                                out_file.write(file_obj.read())

                            n_extracted += 1

        logger.info(f"Extracted {n_extracted} file/s from {gzip_path}")


def read_file_lines(file_path: Path) -> list[str]:
    """Read all lines of a file (ignoring blank lines) into an array of strings

    Args:
        file_path (Path): Path to the text file

    Returns:
        list[str]: List of file contents line-by-line
    """
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    return lines


def save_file_lines(strings: list[str], file_path: Path) -> None:
    """Save a list of strings to a text file, with each string on a new line,
    except for the last line which has no trailing newline.

    Args:
        strings (list[str]): List of strings to save
        file_path (Path): Path to the output text file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        if strings:  # check if the list is not empty
            # write all but the last string with a newline
            for s in strings[:-1]:
                f.write(f"{s}\n")
            # write the last string without a newline
            f.write(strings[-1])


def calculate_label_difference(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Calculate the total difference in proportions of each 'Label' between train_df and test_df.

    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Testing dataframe

    Returns:
        float: Sum of absolute differences in label proportions
    """
    all_labels = sorted(set(train_df["Label"].unique()) | set(test_df["Label"].unique()))
    train_props = train_df["Label"].value_counts(normalize=True).reindex(all_labels, fill_value=0)
    test_props = test_df["Label"].value_counts(normalize=True).reindex(all_labels, fill_value=0)
    total_difference = (train_props - test_props).abs().sum()
    return total_difference


def _label_combo(labels) -> str:
    """Stable key for a row's label set, treating empty/absent labels as 'Normal'."""
    if isinstance(labels, list) and labels:
        return "_".join(sorted(set(labels)))
    return "Normal"


def _assign_patients_to_partitions(
    patients,
    n_partitions: int,
    fractions,
    rng: np.random.Generator,
    carry: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Assign patients to ``n_partitions`` by target counts.

    Sizes come from :func:`_partition_sizes`, which uses **cumulative** rounding:
    ``carry`` holds each partition's running fractional shortfall so that
    rounding is not lost at every stratum boundary. Without it, splitting many
    small label strata one at a time drifts the totals (a 70/10/20 split of 600
    patients came out 417/58/125 instead of 420/60/120).

    ``carry`` is updated in place; pass the same array across calls to keep the
    global proportions. ``rng`` is consumed deterministically so a fixed seed
    stays reproducible.
    """
    patients = np.asarray(patients)
    n = len(patients)
    if n == 0:
        return [np.array([], dtype=patients.dtype) for _ in range(n_partitions)]

    targets = _partition_sizes(n, n_partitions, fractions, carry)
    order = rng.permutation(n)
    out: list[np.ndarray] = []
    start = 0
    for size in targets:
        out.append(patients[order[start : start + size]])
        start += size
    return out


def _partition_sizes(
    n: int, n_partitions: int, fractions, carry: np.ndarray | None = None
) -> list[int]:
    """Integer sizes summing to exactly ``n``.

    Cumulative rounding: accumulate ``fractions * n + carry``, take the floor,
    and carry the fractional remainder into the next stratum. The last partition
    absorbs whatever is left so the sizes always sum to ``n``.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if carry is None:
        carry = np.zeros(n_partitions, dtype=np.float64)

    ideal = fractions * n + carry
    sizes = [int(np.floor(value)) for value in ideal[:-1]]
    sizes.append(n - sum(sizes))
    # update the carry with what rounding discarded (the last partition is
    # already exact, so its carry stays 0)
    carry[:-1] = ideal[:-1] - np.asarray(sizes[:-1], dtype=np.float64)
    carry[-1] = 0.0
    return sizes


def get_best_stratified_split(
    df: pd.DataFrame,
    train_ratio: float | None = None,
    *,
    fractions: tuple[float, ...] | None = None,
    n_seeds: int = 1000,
) -> tuple[pd.DataFrame, ...]:
    """Stratified patient-level split, choosing the seed that best preserves label proportions.

    Finds the seed in ``range(n_seeds)`` whose partitions minimise the summed
    absolute difference in label proportions between partitions (the historical
    0-999 search), so a rare combination cannot end up absent from a partition.

    Call styles:

    * ``get_best_stratified_split(df, 0.7)`` — historical two-way ``(train, test)``
      split, unchanged;
    * ``get_best_stratified_split(df, fractions=(0.7, 0.1, 0.2))`` — any number of
      partitions, e.g. the 70/10/20 used by the cross-dataset transfer study.

    Args:
        df: Input dataframe with ``PatientID`` and ``Labels`` (list of labels per row).
        train_ratio: Proportion of patients in the first partition; only valid for
            a two-way split. Mutually exclusive with ``fractions``.
        fractions: Proportions per partition, summing to 1. Takes precedence.
        n_seeds: Number of seeds to search (default 1000, the historical bound).

    Returns:
        One dataframe per partition, in ``fractions`` order. Two-way calls return
        exactly ``(train_df, test_df)`` as before.
    """
    if fractions is None:
        if train_ratio is None:
            raise ValueError("pass either train_ratio (two-way) or fractions")
        fractions = (float(train_ratio), 1.0 - float(train_ratio))
    elif train_ratio is not None:
        raise ValueError("pass train_ratio or fractions, not both")

    fractions = tuple(float(f) for f in fractions)
    if len(fractions) < 2:
        raise ValueError(f"need at least two partitions, got {fractions!r}")
    if any(f <= 0.0 for f in fractions):
        raise ValueError(f"fractions must be positive, got {fractions!r}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions!r} (sum {sum(fractions)})")

    n_partitions = len(fractions)
    two_way = n_partitions == 2

    # Step 1: Compute label combinations, treating empty lists as 'Normal'
    def normalize_labels(labels):
        return _label_combo(labels)

    patient_labels = (
        df.groupby("PatientID")["Labels"]
        .apply(lambda x: normalize_labels(x.iloc[0]))  # assume 'Labels' is consistent per patient
        .reset_index(name="label_combo")
    )

    # Step 2: Compute label counts, including 'Normal' for empty lists
    def explode_or_split(row):
        labels = row["Labels"]
        return labels if (isinstance(labels, list) and labels) else ["Normal"]

    # create a copy and prepare labels for counting
    exploded_df = df.copy()
    exploded_df["Labels"] = exploded_df.apply(explode_or_split, axis=1)
    exploded = exploded_df.explode("Labels")
    patient_label_counts = exploded.groupby(["PatientID", "Labels"]).size().unstack(fill_value=0)

    # ensure 'Normal' column exists
    if "Normal" not in patient_label_counts.columns:
        patient_label_counts["Normal"] = 0

    # Step 3: Ensure all patients are included
    all_patients = df["PatientID"].unique()
    missing_patients = set(all_patients) - set(patient_label_counts.index)
    if missing_patients:
        missing_df = pd.DataFrame(0, index=missing_patients, columns=patient_label_counts.columns)
        patient_label_counts = pd.concat([patient_label_counts, missing_df])

    # Step 4: Stratify patients based on label combinations
    combo_counts = patient_labels["label_combo"].value_counts()
    count1_combos = combo_counts[combo_counts == 1].index
    count1_patients = patient_labels[patient_labels["label_combo"].isin(count1_combos)][
        "PatientID"
    ].values
    count2_patients_df = patient_labels[~patient_labels["label_combo"].isin(count1_combos)]

    # Step 5: Find the best split
    min_difference = float("inf")
    best_parts: list[np.ndarray] | None = None

    for seed in range(n_seeds):
        np.random.seed(seed)
        # fresh carry per candidate split, shared across that split's strata so
        # per-stratum rounding does not accumulate drift; initialised here so
        # the count1 assignment below cannot hit an unbound name when every
        # combination occurs once (count2_patients_df empty)
        carry = np.zeros(n_partitions, dtype=np.float64)

        # split patients with 2+ occurrences of label combinations
        if not count2_patients_df.empty:
            if two_way:
                # historical path: sklearn stratified shuffle keeps the label
                # combination distribution; unchanged for backward compatibility
                sss = StratifiedShuffleSplit(
                    n_splits=1, train_size=fractions[0], random_state=seed
                )
                for part_idx in sss.split(count2_patients_df, count2_patients_df["label_combo"]):
                    count2_parts = [
                        count2_patients_df.iloc[idx]["PatientID"].values for idx in part_idx
                    ]
            else:
                # N-way: stratify per combination, then slice each stratum to size
                rng = np.random.default_rng(seed)
                count2_parts = [np.array([], dtype=object) for _ in range(n_partitions)]
                for _combo, group in count2_patients_df.groupby("label_combo"):
                    assigned = _assign_patients_to_partitions(
                        group["PatientID"].values,
                        n_partitions,
                        fractions,
                        np.random.default_rng(rng.integers(2**32)),
                        carry=carry,
                    )
                    count2_parts = [
                        np.concatenate([acc, new_part])
                        for acc, new_part in zip(count2_parts, assigned)
                    ]
        else:
            count2_parts = [np.array([]) for _ in range(n_partitions)]

        # assign patients whose label combination occurs only once
        if len(count1_patients) > 0:
            count1_parts = _assign_patients_to_partitions(
                count1_patients,
                n_partitions,
                fractions,
                np.random.default_rng(seed + 1),
                carry=carry,
            )
        else:
            count1_parts = [np.array([]) for _ in range(n_partitions)]

        # combine patients per partition
        parts = [
            np.concatenate([c2, c1]) for c2, c1 in zip(count2_parts, count1_parts)
        ]

        # proportions of each label within each partition, on the same index
        # (patient_label_counts columns), then the summed pairwise deviation
        props = []
        for part in parts:
            counts = patient_label_counts.loc[part].sum()
            total = len(df[df["PatientID"].isin(part)])
            props.append((counts / total) if total > 0 else counts * 0.0)

        difference = 0.0
        for i in range(n_partitions):
            for j in range(i + 1, n_partitions):
                difference += float((props[i] - props[j]).abs().sum())

        # update best split
        if difference < min_difference:
            min_difference = difference
            best_parts = [part.copy() for part in parts]

    # Step 6: Return one dataframe per partition
    return tuple(df[df["PatientID"].isin(part)] for part in best_parts)
