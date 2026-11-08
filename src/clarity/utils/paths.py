"""Explicit path resolution for data, models and segmentation roots.

No path is read from the environment at import time.  Call
:func:`resolve_data_root` (or pass an explicit CLI path) when a path is
actually needed, so importing a library module never depends on the machine
it happens to run on.
"""

from __future__ import annotations

import os
from pathlib import Path

#: primary environment variable for the ChestX-ray14 dataset root
DATA_ROOT_ENV = "CLARITY_DATA_ROOT"
#: deprecated alias, kept for one release
DATA_ROOT_ENV_LEGACY = "CXR_HOME"
#: optional environment variable for the run-folder root
MODELS_ROOT_ENV = "CLARITY_MODELS_ROOT"


class DataRootNotConfigured(RuntimeError):
    """Raised when no data root was passed and no environment variable is set."""


def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the dataset root, preferring an explicit path over the env.

    ``CLARITY_DATA_ROOT`` wins over the deprecated ``CXR_HOME`` alias.  The
    path is not required to exist here; dataset readers validate it when they
    open files.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    value = os.environ.get(DATA_ROOT_ENV) or os.environ.get(DATA_ROOT_ENV_LEGACY)
    if not value:
        raise DataRootNotConfigured(
            "no dataset root configured: pass --data-root/--images or set "
            f"{DATA_ROOT_ENV} (legacy alias: {DATA_ROOT_ENV_LEGACY})"
        )
    return Path(value).expanduser()


def resolve_models_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the run-folder root (explicit path, env, then ``~/models``)."""
    if explicit is not None:
        return Path(explicit).expanduser()
    value = os.environ.get(MODELS_ROOT_ENV)
    if value:
        return Path(value).expanduser()
    return Path.home() / "models"


def resolve_cxr14_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the ChestX-ray14 root (explicit data root > env > default).

    ``data_root`` may be the dataset parent (the usual ``--data-root``) or the
    ChestX-ray14 directory itself; the latter is detected by its metadata CSV.
    """
    if data_root is not None:
        root = Path(data_root).expanduser()
        if (root / "Data_Entry_2017_v2020.csv").is_file() or (
            root / "BBox_List_2017.csv"
        ).is_file():
            return root
        return root / "ChestX-ray14"
    explicit = os.environ.get("CXR_CXR14_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return resolve_data_root() / "ChestX-ray14"


def resolve_padchest_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the PadChest root (optional dataset)."""
    if data_root is not None:
        return Path(data_root).expanduser() / "PadChest"
    explicit = os.environ.get("CXR_PADCHEST_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return resolve_data_root() / "PadChest"


def resolve_rsna_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the RSNA Pneumonia Detection Challenge root (explicit > env).

    ``data_root`` may be the dataset parent (the usual ``--data-root``) or the
    RSNA directory itself; the latter is detected by its label CSV.
    """
    if data_root is not None:
        root = Path(data_root).expanduser()
        if (root / "stage_2_train_labels.csv").is_file() or root.name.startswith("rsna-pneumonia"):
            return root
        return root / "rsna-pneumonia-detection-challenge"
    explicit = os.environ.get("CXR_RSNA_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return resolve_data_root() / "rsna-pneumonia-detection-challenge"


def resolve_vindr_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the VinDr-CXR root (explicit > env).

    ``data_root`` may be the dataset parent (the usual ``--data-root``) or the
    VinDr directory itself; the latter is detected by its ``annotations``
    directory or its ``VinDr-CXR`` name.
    """
    if data_root is not None:
        root = Path(data_root).expanduser()
        if (root / "annotations" / "annotations_test.csv").is_file() or root.name == "VinDr-CXR":
            return root
        return root / "VinDr-CXR"
    explicit = os.environ.get("CXR_VINDR_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return resolve_data_root() / "VinDr-CXR"
