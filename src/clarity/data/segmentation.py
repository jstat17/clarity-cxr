"""Segmentation-cache layout, validation and loading helpers."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from clarity.data.errors import InvalidDatasetError
from clarity.data.files import partition_filename

#: precomputed segmentation cache format (see clarity-prepare)
SEGM_CACHE_FORMAT_VERSION = 1
SEGM_CACHE_META_ENTRY = "meta.json"


def resolve_segmentation_paths(
    segmentation_path: Path,
) -> tuple[Path | None, Path]:
    """Classify a ``-seg`` directory as the npz cache or the legacy pk.xz source.

    Returns ``(cache_path, read_path)``; ``cache_path`` is ``None`` for the
    legacy source directory. Mixed or unknown contents are rejected: there is
    no implicit sibling-folder fallback.
    """
    stored_extensions = {
        partition_filename(filename)[1]
        for filename in os.listdir(segmentation_path)
    }
    if stored_extensions == {"pk.xz"}:
        return None, segmentation_path
    if stored_extensions == {"npz"}:
        return segmentation_path, segmentation_path
    raise InvalidDatasetError(
        "Segmentation directory must contain either only .pk.xz files (legacy "
        "source) or only .npz files (precomputed cache, see "
        "clarity-prepare); found extensions "
        f"{sorted(stored_extensions)} in {segmentation_path}"
    )


def source_dir_for_cache(segmentation_path: Path) -> Path | None:
    """Conventional ``<name>-<size>`` cache directory -> legacy source sibling."""
    import re

    suffix_match = re.fullmatch(r"(.+)-\d+", segmentation_path.name)
    if suffix_match is None:
        return None
    candidate = segmentation_path.parent / suffix_match.group(1)
    return candidate if candidate.is_dir() else None


def validate_cache_meta(
    cache_path: Path,
    segm_shape: tuple[int, int],
    segmentation_path: Path,
    cache_filename: str,
    strict: bool = True,
) -> None:
    """Validate one cache file's meta: version, mask shape and source sha256.

    A cheap one-time probe (first stored file only). Source hashing runs only
    when the legacy source directory is locally reachable.
    """
    def _fail(message: str) -> None:
        if strict:
            raise InvalidDatasetError(message)

    source_dir = source_dir_for_cache(segmentation_path)
    source_path = (
        source_dir / (cache_filename[: -len(".npz")] + ".pk.xz")
        if source_dir is not None
        else None
    )

    try:
        with zipfile.ZipFile(cache_path) as zf:
            meta = json.loads(zf.read(SEGM_CACHE_META_ENTRY))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError) as exc:
        _fail(f"Cannot read segmentation cache meta from {cache_path}: {exc}")
        return

    if meta.get("format_version") != SEGM_CACHE_FORMAT_VERSION:
        _fail(
            f"Segmentation cache format version mismatch in {cache_path}: "
            f"expected {SEGM_CACHE_FORMAT_VERSION}, "
            f"found {meta.get('format_version')}"
        )
        return

    stored_shape = tuple(meta.get("shape", ()))
    if stored_shape != tuple(segm_shape):
        _fail(
            f"Segmentation cache shape mismatch in {cache_path}: cache holds "
            f"{stored_shape} masks but segm_shape is {tuple(segm_shape)}. "
            "Re-run clarity-prepare with -s "
            f"{segm_shape[0]} or pass a matching -seg_s."
        )
        return

    if source_path is not None and source_path.exists():
        sha256 = hashlib.sha256()
        with open(source_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha256.update(chunk)
        if meta.get("source_sha256") != sha256.hexdigest():
            _fail(
                f"Segmentation cache {cache_path} is stale: recorded source "
                f"sha256 does not match {source_path}. Rebuild the cache."
            )


def unpack_cached_segmentation(
    cache_bytes: bytes, segm_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Inflate one cached npz into a ``region -> boolean mask`` dict."""
    height, width = segm_shape
    n_bits = height * width
    segmentation = {}
    with zipfile.ZipFile(io.BytesIO(cache_bytes)) as zf:
        for name in zf.namelist():
            if name == SEGM_CACHE_META_ENTRY:
                continue
            bits = np.frombuffer(zf.read(name), dtype=np.uint8)
            mask = np.unpackbits(bits, count=n_bits).reshape(height, width)
            segmentation[name] = mask.astype(bool, copy=False)
    return segmentation


def load_compressed_segmentations(
    read_path: Path,
    segmentation_filenames: list[str],
    split: str,
    cached: bool,
) -> list[bytes]:
    """Read the raw compressed segmentation files into memory (bytes)."""
    compressed = []
    pbar = tqdm(
        total=len(segmentation_filenames),
        desc=(
            f"Loading {'cached ' if cached else 'compressed '}segmentation "
            f"maps for {split} into memory"
        ),
        unit="image",
    )
    for filename in segmentation_filenames:
        with open(read_path / filename, "rb") as f:
            compressed.append(f.read())
        pbar.update(1)
    pbar.close()
    return compressed


def load_segmentation_dict(
    compressed: bytes,
    cached: bool,
    segm_shape: tuple[int, int],
    resize: bool = True,
) -> dict[str, np.ndarray]:
    """Load one compressed segmentation file into ``region -> mask``.

    The npz cache already holds masks at ``segm_shape``; the legacy pk.xz
    source is decoded (lzma + pickle) and resized on demand.
    """
    if cached:
        return unpack_cached_segmentation(compressed, segm_shape)

    import lzma
    import pickle as pk

    from clarity.data import transform

    with io.BytesIO(compressed) as f_compressed:
        with lzma.open(f_compressed, "rb") as f:
            segmentation = pk.load(f)

    if not resize:
        return segmentation
    return {
        key: transform.manipulate_boolean_mask(mask, func=transform.resize, shape=segm_shape)
        for key, mask in segmentation.items()
    }


def dilate_segmentation(
    segmentation: dict[str, np.ndarray],
    kernel_size: int | None,
    iterations: int | None,
) -> dict[str, np.ndarray]:
    """Optionally dilate every region mask."""
    if not (kernel_size and iterations):
        return segmentation

    from clarity.data import augment

    return {
        key: augment.dilate(mask, kernel_size=kernel_size, iterations=iterations)
        for key, mask in segmentation.items()
    }


def build_label_map(
    segmentation: dict[str, np.ndarray],
    segm_shape: tuple[int, int],
    possible_labels: list[str],
    label_anatomical_regions: dict[str, list[str]],
    use_background_label: bool,
) -> np.ndarray:
    """Combine anatomical-region masks into one boolean mask per finding.

    Labels sharing the same region list share the OR result (memoised by the
    sorted region signature: 11 of the 14 CXR14 labels use the same two lung
    regions). An all-False mask (failed segmentation) is replaced by all-True.
    """
    n_background = int(use_background_label)
    n_labels = len(possible_labels) + n_background
    # mask shape from the data (the legacy path may keep the source resolution
    # when resize=False); segm_shape is only the fallback for an empty dict
    mask_shape = next(iter(segmentation.values())).shape if segmentation else segm_shape
    segmentation_map = np.zeros((n_labels, *mask_shape), order="F", dtype=bool)

    if use_background_label:
        segmentation_map[0] = ~np.logical_or.reduce(list(segmentation.values()))

    memo: dict[tuple[str, ...], np.ndarray] = {}
    for i in range(n_background, n_labels):
        label = possible_labels[i - n_background]
        signature = tuple(sorted(label_anatomical_regions[label]))
        if signature not in memo:
            memo[signature] = np.logical_or.reduce([segmentation[r] for r in signature])
        segmentation_map[i] = memo[signature]

    for i in range(n_labels):
        if np.all(~segmentation_map[i]):
            segmentation_map[i] = np.ones_like(segmentation_map[i], dtype=np.bool_)

    return segmentation_map


def pad_label_map(segmentation: np.ndarray, border_padding: int) -> np.ndarray:
    """Pad every label mask with the configured border."""
    if not border_padding:
        return segmentation

    from clarity.data import transform

    padded = np.zeros_like(segmentation)
    for i in range(segmentation.shape[0]):
        padded[i] = transform.manipulate_boolean_mask(
            segmentation[i],
            transform.border_pad_image,
            border_padding=border_padding,
        )
    return padded
