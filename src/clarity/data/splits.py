"""Split-manifest loading and validation.

The companion repository does not distribute the paper's split lists or any
split generator.  This module only *loads and validates* a manifest supplied
by the user (``--splits-dir``) and provides the official NIH split out of the
box.  On-disk list names and record counts are frozen.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

#: historical paper manifest names inside ``--splits-dir``
MANIFEST_FILES = {
    "classification_train": "classification_train_list_70.txt",
    "classification_val": "classification_val_list_70.txt",
    "localization_val": "localization_val_list.txt",
    "localization_test": "localization_test_list.txt",
}

#: record counts of the paper's historical lists (a supplied manifest may be
#: any size; these are used by ``--verify-splits`` and the docs)
PAPER_COUNTS = {
    "classification_train": 75_697,
    "classification_val": 10_826,
    "localization_val": 442,
    "localization_test": 438,
}

#: official NIH split record counts (before the two ignored images)
OFFICIAL_COUNTS = {"train": 86_524, "test": 25_596}

OFFICIAL_TRAIN_FILE = "train_val_list.txt"
OFFICIAL_TEST_FILE = "test_list.txt"


class SplitValidationError(ValueError):
    """A split manifest is missing, malformed, overlapping or mis-sized."""


def patient_id(filename: str) -> str:
    """Patient identifier encoded in a ChestX-ray14 filename (prefix)."""
    return Path(filename).stem.split("_", 1)[0]


def filter_filenames(
    filenames: Iterable[str],
    images_path: str | Path | None = None,
    ignored: Iterable[str] = (),
) -> list[str]:
    """Drop ignored names and (optionally) files absent from ``images_path``.

    Manifest lists record the dataset filenames verbatim; the dataset readers
    still skip the two known-bad images and any image missing from the
    extracted directory, exactly like the historical list readers.
    """
    ignored = set(ignored)
    result = [str(name) for name in filenames if str(name) not in ignored]
    if images_path is not None:
        available = set(os.listdir(images_path))
        result = [name for name in result if name in available]
    return result


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise SplitValidationError(
            f"missing split file {path}; supply the four historical lists via "
            "--splits-dir (see docs/data.md)"
        )
    records = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SplitValidationError(f"split file {path} is empty")
    return records


def _check_patient_disjoint(
    name_a: str, records_a: Sequence[str], name_b: str, records_b: Sequence[str]
) -> None:
    patients_a = {patient_id(name) for name in records_a}
    patients_b = {patient_id(name) for name in records_b}
    overlap = patients_a & patients_b
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise SplitValidationError(f"patient overlap between {name_a} and {name_b}: {sample}")


@dataclass(frozen=True)
class SplitManifest:
    """The four historical split lists, validated for patient disjointness."""

    classification_train: tuple[str, ...]
    classification_val: tuple[str, ...]
    localization_val: tuple[str, ...]
    localization_test: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in MANIFEST_FILES}

    def validate_counts(self, expected: Mapping[str, int]) -> None:
        """Raise unless every requested split has the expected record count."""
        actual = self.counts
        wrong = {
            name: (actual[name], expected[name])
            for name in expected
            if actual.get(name) != expected[name]
        }
        if wrong:
            details = ", ".join(
                f"{name}: got {got}, expected {want}" for name, (got, want) in sorted(wrong.items())
            )
            raise SplitValidationError(f"split count mismatch: {details}")


def load_manifest(
    directory: str | Path, expected_counts: Mapping[str, int] | None = None
) -> SplitManifest:
    """Load and validate the four files from a ``--splits-dir`` directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise SplitValidationError(
            f"split manifest directory {directory} does not exist; pass the "
            "directory containing the four historical list files via "
            "--splits-dir (see docs/data.md)"
        )
    records = {name: _read_list(directory / filename) for name, filename in MANIFEST_FILES.items()}
    names = list(records)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            _check_patient_disjoint(name_a, records[name_a], name_b, records[name_b])
    manifest = SplitManifest(**{name: tuple(values) for name, values in records.items()})
    if expected_counts is not None:
        manifest.validate_counts(expected_counts)
    return manifest


@dataclass(frozen=True)
class OfficialSplit:
    """The official NIH train_val/test split read from the dataset root."""

    train: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "test": len(self.test)}


def load_official_split(directory: str | Path) -> OfficialSplit:
    """Read ``train_val_list.txt`` and ``test_list.txt`` from the data root."""
    directory = Path(directory)
    return OfficialSplit(
        train=tuple(_read_list(directory / OFFICIAL_TRAIN_FILE)),
        test=tuple(_read_list(directory / OFFICIAL_TEST_FILE)),
    )


def verify_official_split(
    split: OfficialSplit, expected_counts: Mapping[str, int] | None = None
) -> None:
    """Validate official-split counts (for ``--verify-splits``)."""
    expected = expected_counts or OFFICIAL_COUNTS
    wrong = {
        name: (split.counts[name], want)
        for name, want in expected.items()
        if split.counts.get(name) != want
    }
    if wrong:
        details = ", ".join(
            f"{name}: got {got}, expected {want}" for name, (got, want) in sorted(wrong.items())
        )
        raise SplitValidationError(f"official split count mismatch: {details}")
