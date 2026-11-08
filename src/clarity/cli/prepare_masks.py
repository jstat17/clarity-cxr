"""Precompute segmentation-map caches from compressed pk.xz segmentation files.

For each source ``<id>.pk.xz`` under the source directory, this script writes a
per-resolution cache file ``<id>.npz`` into a sibling directory named
``<source-dir-name>-<size>``. Each npz archive contains:

- one zip entry per anatomical region (entry name = region name, e.g.
  ``"Left Clavicle"``), whose payload is the ``np.packbits``-compressed
  boolean mask at the target resolution;
- a ``meta.json`` entry recording the format version, target shape, packbits
  payload size, and the sha256 of the source pk.xz bytes (used for staleness
  checks by ``--verify`` and by the loader).

Masks for a target size of 512 are stored bit-identically to the source
(verified: cv2 bilinear resize to the same size is an exact identity).
Other sizes go through the exact standard resize path
(``transform.manipulate_boolean_mask`` with ``transform.resize``), so cached
masks are bitwise identical to what the loader would compute per sample.

The archives are written deterministically (fixed zip timestamps), so
re-running the script on identical sources yields byte-identical npz files.

Example::

    clarity-prepare -seg <segmentation-root> -out <cache-root> -s 224 512 --workers 16
"""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import lzma
import pickle as pk
from tqdm import tqdm

from clarity.data import transform

# bump when the on-disk format changes; loaders assert on mismatch
CACHE_FORMAT_VERSION = 1

# fixed zip timestamp so identical sources produce byte-identical npz files
ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

META_ENTRY_NAME = "meta.json"


def cache_dir_for(source_dir: Path, out_root: Path, size: int) -> Path:
    """Return the cache directory for a given source directory and size."""
    return out_root / f"{source_dir.name}-{size}"


def pack_segmentation(
    segmentation: dict[str, np.ndarray], segm_shape: tuple[int, int] | None
) -> dict[str, bytes]:
    """Convert a dict of boolean masks to a dict of packbits payloads.

    If ``segm_shape`` is given and differs from the mask shape, each mask is
    resized with the exact standard path (uint8 * 255 -> cv2 bilinear ->
    threshold at 127) so cached masks equal what the loader would compute.
    """
    packed = dict()
    for key, mask in segmentation.items():
        if segm_shape is not None and tuple(mask.shape) != tuple(segm_shape):
            mask = transform.manipulate_boolean_mask(mask, func=transform.resize, shape=segm_shape)
        mask = np.asarray(mask, dtype=bool)
        packed[key] = np.packbits(mask).tobytes()
    return packed


def build_npz_bytes(packed: dict[str, bytes], meta: dict) -> bytes:
    """Serialize packbits payloads and metadata into a deterministic npz."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for key, payload in packed.items():
            info = zipfile.ZipInfo(key, date_time=ZIP_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)

        info = zipfile.ZipInfo(META_ENTRY_NAME, date_time=ZIP_FIXED_DATE_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, json.dumps(meta, sort_keys=True))

    return buffer.getvalue()


def convert_file(task: tuple[str, str, int | None]) -> str:
    """Convert one source pk.xz file into a cache npz file.

    Args:
        task (tuple): (source path, output path, segm_size or None).
            ``segm_size = None`` means no resize is performed (used for
            verification against a source of the same resolution).

    Returns:
        str: The output path (for progress reporting).
    """
    source_path, output_path, segm_size = task

    with open(source_path, "rb") as f:
        source_bytes = f.read()

    with lzma.open(io.BytesIO(source_bytes), "rb") as f:
        segmentation: dict[str, np.ndarray] = pk.load(f)

    if segm_size is not None:
        segm_shape = (segm_size, segm_size)
    else:
        segm_shape = None

    packed = pack_segmentation(segmentation, segm_shape)

    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "shape": list(segm_shape) if segm_shape is not None else None,
        "sizes": {key: len(payload) for key, payload in packed.items()},
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }

    npz_bytes = build_npz_bytes(packed, meta)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(npz_bytes)
    os.replace(tmp_path, output_path)

    return output_path


def verify_file(task: tuple[str, str, int | None]) -> tuple[str, str | None]:
    """Verify a cache npz against its source pk.xz file.

    Re-opens the npz, checks the meta staleness fields, and recomputes the
    expected masks from the source (full recompute + compare).

    Args:
        task (tuple): (source path, cache npz path, segm_size or None).

    Returns:
        tuple[str, str | None]: (npz path, error message or None on success).
    """
    source_path, npz_path, segm_size = task

    with open(source_path, "rb") as f:
        source_bytes = f.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    with open(npz_path, "rb") as f:
        npz_bytes = f.read()

    with zipfile.ZipFile(io.BytesIO(npz_bytes)) as zf:
        meta = json.loads(zf.read(META_ENTRY_NAME))

        if meta["format_version"] != CACHE_FORMAT_VERSION:
            return npz_path, f"format version mismatch: {meta['format_version']}"
        if meta["source_sha256"] != source_sha256:
            return npz_path, "source file changed since the cache was built"

        with lzma.open(io.BytesIO(source_bytes), "rb") as f:
            segmentation: dict[str, np.ndarray] = pk.load(f)

        if sorted(zf.namelist()) != sorted(list(segmentation.keys()) + [META_ENTRY_NAME]):
            return npz_path, "npz entries do not match the source region names"

        for key, mask in segmentation.items():
            if segm_size is not None and tuple(mask.shape) != (segm_size, segm_size):
                mask = transform.manipulate_boolean_mask(
                    mask, func=transform.resize, shape=(segm_size, segm_size)
                )
            mask = np.asarray(mask, dtype=bool)
            cached = (
                np.unpackbits(np.frombuffer(zf.read(key), dtype=np.uint8), count=mask.size)
                .reshape(mask.shape)
                .astype(bool)
            )

            if not np.array_equal(cached, mask):
                return npz_path, f"cached mask differs from source for region '{key}'"

    return npz_path, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute per-resolution segmentation-map caches "
        "from compressed pk.xz segmentation files."
    )
    parser.add_argument(
        "-seg",
        "--segmentation_path",
        type=Path,
        required=True,
        help="Path to the source segmentation directory (containing <id>.pk.xz files)",
    )
    parser.add_argument(
        "-out",
        "--output_path",
        type=Path,
        required=True,
        help="Directory in which cache folders <source-dir-name>-<size> are created",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="Target resolution(s) for the caches (e.g. 224 512)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel worker processes (default 1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cache files even if they already exist and are fresh",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After building, re-open each npz and compare against the source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir: Path = args.segmentation_path
    out_root: Path = args.output_path
    if not source_dir.is_dir():
        raise SystemExit(f"Source segmentation directory not found: {source_dir}")

    source_filenames = sorted(
        filename for filename in os.listdir(source_dir) if filename.endswith(".pk.xz")
    )
    if not source_filenames:
        raise SystemExit(f"No .pk.xz files found in {source_dir}")

    for size in args.sizes:
        cache_dir = cache_dir_for(source_dir, out_root, size)
        cache_dir.mkdir(parents=True, exist_ok=True)

        tasks: list[tuple[str, str, int]] = []
        n_skipped = 0
        for filename in source_filenames:
            source_path = source_dir / filename
            output_path = str(cache_dir / f"{filename[:-len('.pk.xz')]}.npz")

            # cheap staleness check: skip if the cache exists, has the right
            # format version, and the source sha256 recorded in meta matches
            if not args.overwrite and os.path.exists(output_path):
                try:
                    with open(source_path, "rb") as f:
                        source_sha256 = hashlib.sha256(f.read()).hexdigest()
                    with zipfile.ZipFile(output_path) as zf:
                        meta = json.loads(zf.read(META_ENTRY_NAME))
                    fresh = (
                        meta["format_version"] == CACHE_FORMAT_VERSION
                        and meta["source_sha256"] == source_sha256
                    )
                except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError):
                    fresh = False

                if fresh:
                    n_skipped += 1
                    continue

            tasks.append((str(source_path), output_path, size))

        print(
            f"[size {size}] {len(tasks)} file(s) to convert, "
            f"{n_skipped} already fresh, cache dir: {cache_dir}"
        )

        if tasks:
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(convert_file, task) for task in tasks]
                    with tqdm(
                        total=len(futures), desc=f"Converting (size {size})", unit="file"
                    ) as pbar:
                        for future in as_completed(futures):
                            future.result()
                            pbar.update(1)
            else:
                for task in tqdm(tasks, desc=f"Converting (size {size})", unit="file"):
                    convert_file(task)

        if args.verify:
            verify_tasks = [
                (
                    str(source_dir / filename),
                    str(cache_dir / f"{filename[:-len('.pk.xz')]}.npz"),
                    size,
                )
                for filename in source_filenames
            ]
            n_errors = 0
            with tqdm(
                total=len(verify_tasks), desc=f"Verifying (size {size})", unit="file"
            ) as pbar:
                if args.workers > 1:
                    with ProcessPoolExecutor(max_workers=args.workers) as executor:
                        futures = [executor.submit(verify_file, task) for task in verify_tasks]
                        for future in as_completed(futures):
                            npz_path, error = future.result()
                            if error is not None:
                                n_errors += 1
                                print(f"VERIFY FAILED: {npz_path}: {error}", file=sys.stderr)
                            pbar.update(1)
                else:
                    for task in verify_tasks:
                        npz_path, error = verify_file(task)
                        if error is not None:
                            n_errors += 1
                            print(f"VERIFY FAILED: {npz_path}: {error}", file=sys.stderr)
                        pbar.update(1)

            if n_errors:
                raise SystemExit(f"Verification failed for {n_errors} file(s)")
            print(f"[size {size}] verification passed for {len(verify_tasks)} file(s)")


if __name__ == "__main__":
    main()
