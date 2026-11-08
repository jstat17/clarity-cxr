"""Filename and split-selection helpers for the ChestX-ray14 datasets."""
from __future__ import annotations

import os
from pathlib import Path
from tqdm import tqdm

from clarity.data import transform
from clarity.data.utils import get_iter_to_iter_dict


def partition_filename(filename: str, delim: str = ".") -> tuple[str, str]:
    """Split ``name.ext`` into ``(name, extension)`` (extension has no dot).

    ``str.partition`` keeps multi-part extensions intact: ``a.pk.xz`` becomes
    ``("a", "pk.xz")``.
    """
    name, _delim, extension = filename.partition(delim)
    return name, extension


def get_file_extension_info(
    filenames: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Return the stems (order-preserving, unique) and a stem→extension map."""
    name_to_extension: dict[str, str] = {}
    names: list[str] = []
    for filename in filenames:
        name, extension = partition_filename(filename)
        if name not in name_to_extension:
            names.append(name)
        name_to_extension[name] = extension
    return names, name_to_extension


def add_file_extensions(filenames: list[str], path: Path) -> list[str]:
    """Attach the on-disk extension for each stem.

    The historical implementation zipped the input stems against the
    directory's extensions *by position*, which silently mismatched names and
    extensions when a directory mixed file types. This looks each stem up by
    name instead (no behaviour change for the .png-only CXR14 tree) and lists
    the directory once instead of once per filename.
    """
    stored_filenames = os.listdir(path)
    _, stored_extensions = get_file_extension_info(stored_filenames)

    new_filenames = []
    for filename in filenames:
        name, extension = partition_filename(filename)
        stored = stored_extensions.get(name)
        if stored is not None:
            new_filenames.append(f"{name}.{stored}" if stored else name)
        else:
            new_filenames.append(filename if extension else name)
    return new_filenames


def select_split_filenames(
    df,
    images_path: Path,
    split: str,
    split_pct1: float,
    split_pct2: float,
) -> list[str]:
    """Select the filenames of ``split`` by the FNV-1a patient hash.

    The historical protocol hashes the patient ID into ``2**32`` buckets and
    assigns ``train``/``validate``/``test`` by the two percentage cut points.
    """
    split = split.casefold()
    possible_hashes = 2**32
    hash_split1 = int(possible_hashes * split_pct1)
    hash_split2 = int(possible_hashes * split_pct2)

    # filter images available in images_path by the filenames in df
    stored_filenames = os.listdir(images_path)
    stored_filenames, _ = get_file_extension_info(stored_filenames)

    option_filenames = df["Filename"].tolist()
    option_filenames, _ = get_file_extension_info(option_filenames)

    all_filenames = list(set(option_filenames) & set(stored_filenames))
    all_filenames = add_file_extensions(all_filenames, images_path)
    n = len(all_filenames)

    selected_filenames = []
    filename_to_patient_id = get_iter_to_iter_dict(
        iter1=df["Filename"], iter2=df["PatientID"]
    )

    pbar = tqdm(
        total=n,
        desc=f"Selecting {split} filenames based on hash value",
        unit="file",
    )
    for i in range(n):
        filename = all_filenames[i]
        try:
            patient_id = filename_to_patient_id[filename]
        except KeyError:
            pbar.update(1)
            continue

        value = transform.fnv1a_32(patient_id)
        if (
            (value <= hash_split1 and split == "train")
            or (hash_split1 < value <= hash_split2 and split == "validate")
            or (hash_split2 < value and split == "test")
        ):
            selected_filenames.append(filename)
        pbar.update(1)
    pbar.close()

    return selected_filenames
