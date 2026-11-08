"""Model-agnostic contracts for localization layer-count sweeps.

This module holds the small, shared vocabulary used by the optional
localization sweep feature (see docs/onebox.md):

- an immutable sweep specification (:class:`LocalizationSweepSpec`);
- strict, round-trippable variant keys (``n_attn_layers_avg=2``);
- documented result/selection schemas (:class:`VariantLocalizationResult`,
  :class:`EpochSweepSelection`, :class:`BestOverallState`,
  :class:`SweepManifest`);
- the deterministic sweep score and tie-breaking helpers;
- the per-epoch selection state (:class:`SweepSelectionState`) and the
  artifact manager for snapshots, ``best-selection.json``, and the relative
  ``model_best.pth`` symlink;
- a minimal :func:`unwrap_model` capability-access helper.

The module contains no model code and no imports of ``clarity.models.clarity``; it
serializes tensors only through :func:`torch.save` snapshots. Later steps
(training integration, standalone evaluation, reporting) consume these
definitions; ad-hoc key formatting, score/tie-breaking, or manifest handling
must not be re-implemented elsewhere.
"""

from __future__ import annotations

import logging

import json
import math
import numbers
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch
from torch import nn

logger = logging.getLogger(__name__)

from clarity.localization.sweep.selection import SweepSelectionState
from clarity.localization.sweep.spec import (
    BEST_EPOCHS_DIRNAME,
    ManifestValidationError,
    BEST_MODEL_FILENAME,
    BEST_SELECTION_FILENAME,
    LocalizationSweepSpec,
    SELECTION_METRIC_DESCRIPTION,
    SELECTION_SCHEMA_VERSION,
    parse_variant_key,
    variant_key,
)

SNAPSHOT_STEM_PATTERN = re.compile(r"^epoch_(\d+)\.pth$")


def snapshot_filename(epoch: int) -> str:
    """Return the canonical snapshot file name ``epoch_NNN.pth``.

    The file name is zero-padded to at least three digits (matching the
    historical ``epoch_011`` style) but never truncated.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"epoch must be an integer, got {epoch!r}")
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch!r}")
    return f"epoch_{epoch:03d}.pth"


def parse_snapshot_filename(filename: str) -> int | None:
    """Parse ``epoch_NNN.pth`` into its epoch, or ``None`` if not a snapshot.

    Only plain integer-stemmed ``.pth`` files are recognized; anything else
    (including subdirectory paths) returns ``None`` so garbage collection
    never touches unexpected files.
    """
    if not isinstance(filename, str):
        return None
    match = SNAPSHOT_STEM_PATTERN.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_relative_posix_path(run_dir: Path, snapshot_path: Path) -> str:
    """Return the relative POSIX checkpoint path recorded in the manifest."""
    return snapshot_path.relative_to(run_dir).as_posix()


def _resolve_checkpoint_within_run_dir(run_dir: Path, checkpoint: Any) -> Path:
    """Resolve a manifest checkpoint path to a snapshot inside ``run_dir``.

    The canonical manifest stores exactly ``best-epochs/epoch_NNN.pth``;
    anything else (absolute paths, ``..``/``.`` components, backslashes,
    other directories, non-snapshot file names, or missing files) is
    rejected as malformed.
    """
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ManifestValidationError(f"checkpoint must be a non-empty string, got {checkpoint!r}")
    if os.path.isabs(checkpoint) or checkpoint.startswith("/"):
        raise ManifestValidationError(
            f"absolute checkpoint paths are not allowed, got {checkpoint!r}"
        )
    if "\\" in checkpoint:
        raise ManifestValidationError(
            f"checkpoint must be a relative POSIX path, got {checkpoint!r}"
        )
    relative = PurePosixPath(checkpoint)
    expected_parts = 2
    if len(relative.parts) != expected_parts or relative.parts[0] != BEST_EPOCHS_DIRNAME:
        raise ManifestValidationError(
            f"checkpoint must be a path under {BEST_EPOCHS_DIRNAME!r}/, got " f"{checkpoint!r}"
        )
    for part in relative.parts:
        if part in ("", ".", ".."):
            raise ManifestValidationError(
                f"checkpoint must not contain {part!r} components, got " f"{checkpoint!r}"
            )
    if parse_snapshot_filename(relative.parts[1]) is None:
        raise ManifestValidationError(
            f"checkpoint must name an epoch_NNN.pth snapshot, got " f"{checkpoint!r}"
        )
    resolved = run_dir.joinpath(*relative.parts)
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:  # pragma: no cover - defense in depth
        raise ManifestValidationError(
            f"checkpoint escapes the run folder, got {checkpoint!r}"
        ) from exc
    if not resolved.is_file():
        raise ManifestValidationError(
            f"checkpoint file does not exist inside the run folder: " f"{checkpoint!r}"
        )
    return resolved


def _require_non_negative_int(container: Mapping[str, Any], key: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestValidationError(
            f"manifest field {key!r} must be a non-negative integer, got " f"{value!r}"
        )
    return value


class ExistingBestModelError(RuntimeError):
    """Raised when sweep artifacts would overwrite a regular file.

    ``model_best.pth`` historically holds a full regular checkpoint in
    non-sweep runs. Sweep artifact initialization refuses to replace such a
    file unless the caller explicitly opted in (after migrating/backing it
    up), to prevent accidental data loss.
    """


def _require_finite_float(container: Mapping[str, Any], key: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ManifestValidationError(f"manifest field {key!r} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ManifestValidationError(f"manifest field {key!r} must be finite, got {value!r}")
    return value


def _validate_manifest_structure(manifest: Any) -> None:
    """Validate the schema/structure of a parsed manifest document.

    Raises :class:`ManifestValidationError` on unknown schema versions,
    missing best-overall fields, malformed paths, or a non-mapping
    ``variants`` section. Snapshot existence is validated by the caller
    (``load_best_selection``) which knows the run directory.
    """
    if not isinstance(manifest, dict):
        raise ManifestValidationError(
            f"manifest must be a JSON object, got {type(manifest).__name__}"
        )
    version = manifest.get("schema_version")
    if version != SELECTION_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"unsupported manifest schema_version {version!r}; expected "
            f"{SELECTION_SCHEMA_VERSION}"
        )

    best_overall = manifest.get("best_overall")
    if not isinstance(best_overall, dict):
        raise ManifestValidationError("manifest is missing a best_overall object")
    for field in ("epoch", "score", "checkpoint"):
        if field not in best_overall:
            raise ManifestValidationError(
                f"manifest best_overall is missing required field {field!r}"
            )
    if "attention_layer_count" not in best_overall and "n_attn_layers_avg" not in best_overall:
        raise ManifestValidationError(
            "manifest best_overall is missing required field "
            "'attention_layer_count' (or 'n_attn_layers_avg')"
        )

    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        raise ManifestValidationError(
            f"manifest variants must be an object, got {type(variants).__name__}"
        )
    for count_key, entry in variants.items():
        if not isinstance(count_key, str) or not count_key.isdigit():
            raise ManifestValidationError(
                f"manifest variant keys must be decimal layer counts, got " f"{count_key!r}"
            )
        if not isinstance(entry, dict):
            raise ManifestValidationError(f"manifest variant {count_key!r} must be an object")
        for field in ("epoch", "score", "checkpoint"):
            if field not in entry:
                raise ManifestValidationError(
                    f"manifest variant {count_key!r} is missing required " f"field {field!r}"
                )


def _validate_manifest_values(manifest: dict, run_dir: Path) -> None:
    """Validate manifest field values, including checkpoint file existence."""
    best_overall = manifest["best_overall"]
    _require_non_negative_int(best_overall, "epoch")
    _require_finite_float(best_overall, "score")
    _resolve_checkpoint_within_run_dir(run_dir, best_overall["checkpoint"])

    for count_key, entry in manifest["variants"].items():
        _require_non_negative_int(entry, "epoch")
        _require_finite_float(entry, "score")
        _resolve_checkpoint_within_run_dir(run_dir, entry["checkpoint"])


def _best_overall_attention_layer_count(manifest: dict) -> int:
    best_overall = manifest["best_overall"]
    if "attention_layer_count" in best_overall:
        return _require_non_negative_int(best_overall, "attention_layer_count")
    return _require_non_negative_int(best_overall, "n_attn_layers_avg")


class SweepArtifactManager:
    """Maintain best-per-variant snapshots, the manifest, and the symlink.

    The manager owns exactly three artifact families inside one run
    directory:

    - ``best-epochs/epoch_NNN.pth`` — one inference snapshot per distinct
      selected epoch (no optimizer state; written atomically);
    - ``best-selection.json`` — the canonical selection manifest (written
      atomically with flush/fsync + ``os.replace``);
    - ``model_best.pth`` — a *relative* symlink to the best-overall
      snapshot (replaced atomically via a temporary symlink).

    Update order per epoch (crash-recoverable; see the module docstring):

    1. ensure every newly-referenced snapshot exists on disk;
    2. atomically update the manifest, then the symlink;
    3. on every load/update, repair a stale or missing symlink from the
       canonical manifest;
    4. only then garbage-collect unreferenced snapshots.

    A crash between these steps may leave the manifest and symlink
    temporarily inconsistent; :meth:`recover` (called on every update and
    available standalone) converges them without ever deleting a referenced
    snapshot.
    """

    def __init__(self, run_dir: str | os.PathLike, spec: LocalizationSweepSpec) -> None:
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.state = SweepSelectionState(spec)
        # Tracks which epochs already have a snapshot on disk, so that callers can
        # verify "at most one snapshot per epoch".
        self.written_snapshot_epochs: set[int] = set()
        # Explicit opt-in for initializing sweep artifacts in a directory
        # that already holds a real (non-symlink) ``model_best.pth``. The
        # caller must have migrated/backed the file up first; see
        # :meth:`initialize_run_dir`.
        self._overwrite_authorized = False

    # -- paths -----------------------------------------------------------------

    @property
    def best_epochs_dir(self) -> Path:
        return self.run_dir / BEST_EPOCHS_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / BEST_SELECTION_FILENAME

    @property
    def best_model_path(self) -> Path:
        return self.run_dir / BEST_MODEL_FILENAME

    def snapshot_path(self, epoch: int) -> Path:
        return self.best_epochs_dir / snapshot_filename(epoch)

    # -- initialization ---------------------------------------------------------

    def initialize_run_dir(self) -> None:
        """Prepare the run directory for sweep artifacts.

        Creates ``best-epochs/`` and verifies that ``model_best.pth`` is
        either absent or already a symlink. A real regular file at
        ``model_best.pth`` is never silently overwritten: the call
        fails with :class:`ExistingBestModelError` unless the caller
        explicitly opted into sweep artifact initialization (after migrating
        or backing the file up) via
        :meth:`authorize_existing_best_model_overwrite`.
        """
        self.best_epochs_dir.mkdir(parents=True, exist_ok=True)
        if self.best_model_path.is_symlink() or not self.best_model_path.exists():
            return
        if self._overwrite_authorized:
            return
        raise ExistingBestModelError(
            f"refusing to initialize sweep artifacts: {self.best_model_path} "
            "exists and is a regular file, not a symlink. Back it up or "
            "migrate it manually, then call "
            "authorize_existing_best_model_overwrite() to confirm the file "
            "was deliberately handled."
        )

    def authorize_existing_best_model_overwrite(self) -> None:
        """Explicitly allow initializing over an existing regular file.

        The caller is responsible for having backed up or migrated the file
        beforehand; this only acknowledges that responsibility.
        """
        self._overwrite_authorized = True

    # -- per-epoch update ---------------------------------------------------------

    def update(
        self,
        epoch: int,
        model: nn.Module,
        auc_roc_macro: float,
        variant_iou_accuracy: Mapping[str, float],
    ) -> dict[str, Any]:
        """Process one finished epoch and persist any new selection.

        Exactly one snapshot is written for the epoch when *any* variant
        improved; nothing is written otherwise. The manifest is
        updated before the symlink, and unreferenced snapshots are garbage
        collected last.

        Returns a plain dict with ``overall_improved``, the improved variant
        keys, and the updated best-overall pair.
        """
        self.state.update(
            epoch=epoch,
            auc_roc_macro=auc_roc_macro,
            variant_iou_accuracy=variant_iou_accuracy,
        )

        # A regular (non-symlink) model_best.pth must never be silently
        # replaced by an update either; authorize first if truly intended.
        if (
            self.best_model_path.exists()
            and not self.best_model_path.is_symlink()
            and not self._overwrite_authorized
        ):
            raise ExistingBestModelError(
                f"refusing to update sweep artifacts: {self.best_model_path} "
                "is a regular file, not a symlink. Back it up or migrate it "
                "manually, then call authorize_existing_best_model_overwrite() "
                "to confirm the file was deliberately handled."
            )

        if self.state.any_improvement:
            # First step of the recovery order: make the snapshot exist before
            # anything references it.
            self._write_snapshot_if_missing(epoch, model)

        # Second step: atomically persist the manifest, then move the symlink.
        self.write_manifest()
        if self.state.overall_improved:
            self._replace_best_model_symlink(
                _checkpoint_relative_posix_path(self.run_dir, self.snapshot_path(epoch))
            )

        # Steps 3+4 of the recovery order: converge any pre-existing
        # inconsistency, then drop snapshots nothing references anymore.
        self.repair_symlink_from_manifest()
        self._collect_garbage()

        return self.state.snapshot_state()

    # -- snapshot writing -------------------------------------------------------

    def _write_snapshot_if_missing(self, epoch: int, model: nn.Module) -> Path:
        """Write ``epoch_NNN.pth`` unless it already exists (deduplication).

        All variants selecting the same epoch share one file, so a snapshot
        that already exists is left untouched. The payload contains only
        ``epoch``, ``model_state_dict``, and the ``selection`` metadata
        block — no optimizer state — so it loads through the existing
        ``parse_weights()`` helper.
        """
        path = self.snapshot_path(epoch)
        if path.exists():
            return path
        self.best_epochs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "selection": {
                "improved_attention_layer_counts": [
                    parse_variant_key(key) for key in self.state.improved_variants
                ],
                "epoch_scores": dict(self.state.epoch_scores),
            },
        }
        self._atomic_torch_save(payload, path)
        self.written_snapshot_epochs.add(epoch)
        return path

    @staticmethod
    def _atomic_torch_save(payload: dict, path: Path) -> None:
        """Save through a same-directory temporary file and ``os.replace``."""
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with open(temp_path, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    # -- manifest and symlink ------------------------------------------------------

    def write_manifest(self) -> None:
        """Atomically write the canonical manifest from the in-memory state."""
        manifest = self.render_manifest()
        temp_path = self.manifest_path.with_name(f".{BEST_SELECTION_FILENAME}.tmp-{os.getpid()}")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.manifest_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def render_manifest(self) -> dict[str, Any]:
        """Render the manifest document from the current selection state."""
        return self._render_manifest(self.run_dir)

    def _render_manifest(self, run_dir: Path) -> dict[str, Any]:
        state = self.state
        if state.best_overall is None:
            raise ManifestValidationError(
                "cannot render a manifest before the first epoch has improved"
            )

        def checkpoint_for(count: int, epoch: int) -> str:
            return _checkpoint_relative_posix_path(run_dir, self.snapshot_path(epoch))

        best = state.best_overall
        variants: dict[str, dict[str, Any]] = {}
        for count in self.spec.attention_layer_counts:
            key = variant_key(count)
            selection = state.best_variants.get(key)
            if selection is None:
                continue
            variants[str(count)] = {
                "epoch": selection.epoch,
                "score": selection.score,
                "checkpoint": checkpoint_for(count, selection.epoch),
            }

        best_selection = state.best_variants[variant_key(best["attention_layer_count"])]
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_metric": SELECTION_METRIC_DESCRIPTION,
            "early_stopping_policy": self.spec.early_stopping_policy,
            "exact_localization": True,
            "primary_attention_layer_count": self.spec.primary_attention_layer_count,
            "attention_layer_counts": list(self.spec.attention_layer_counts),
            "best_overall": {
                "epoch": best["epoch"],
                "attention_layer_count": best["attention_layer_count"],
                "n_attn_layers_avg": best["attention_layer_count"],
                "score": best["score"],
                "auc_roc_macro": best_selection.auc_roc_macro,
                "IoU_accuracy_macro": best_selection.iou_accuracy_macro,
                "checkpoint": checkpoint_for(best["attention_layer_count"], best["epoch"]),
            },
            "variants": variants,
        }

    def _replace_best_model_symlink(self, relative_target: str) -> None:
        """Atomically point ``model_best.pth`` at a relative snapshot path.

        Creates a temporary symlink in the run directory and ``os.replace``s
        it into place, so the link is never absent or half-updated. The
        target is always the relative POSIX path recorded in the manifest —
        never absolute.
        """
        temp_link = self.best_model_path.with_name(f".{BEST_MODEL_FILENAME}.tmp-{os.getpid()}")
        try:
            if temp_link.is_symlink() or temp_link.exists():
                temp_link.unlink()
            os.symlink(relative_target, temp_link)
            os.replace(temp_link, self.best_model_path)
        except BaseException:
            if temp_link.is_symlink() or temp_link.exists():
                temp_link.unlink()
            raise

    # -- recovery / repair ---------------------------------------------------------

    def repair_symlink_from_manifest(self) -> bool:
        """Converge the symlink with the canonical manifest.

        A crash between manifest and symlink updates can leave
        ``model_best.pth`` stale, missing, or pointing at a snapshot that is
        no longer best-overall. The manifest is authoritative: if it is
        readable, the symlink is (re)created from its ``best_overall``
        checkpoint. Returns ``True`` when the symlink was (re)paired.
        """
        if not self.manifest_path.exists():
            return False
        manifest = self.read_manifest()
        target = manifest["best_overall"]["checkpoint"]
        desired = os.readlink(self.best_model_path) if self.best_model_path.is_symlink() else None
        if desired == target:
            return False
        self._replace_best_model_symlink(target)
        return True

    def recover(self) -> None:
        """Full recovery pass: repair the symlink, then garbage-collect.

        Safe to call after a crash at any point of :meth:`update`. A missing
        or unreadable manifest leaves the artifacts untouched apart from
        symlink repair (which itself needs the manifest). Never deletes a
        referenced snapshot.
        """
        try:
            self.repair_symlink_from_manifest()
        except ManifestValidationError:
            # Without a valid manifest there is no canonical selection to
            # converge to; never garbage-collect on a guess.
            return
        self._collect_garbage()

    # -- garbage collection ---------------------------------------------------------

    def _referenced_checkpoints(self) -> set[str]:
        referenced: set[str] = set()
        if self.manifest_path.exists():
            try:
                manifest = self.read_manifest()
            except ManifestValidationError:
                # The manifest exists but is unreadable/incomplete (crash
                # mid-write is impossible thanks to os.replace, but a
                # hand-edited or truncated file must not trigger deletion).
                return referenced
            referenced.add(manifest["best_overall"]["checkpoint"])
            for entry in manifest["variants"].values():
                referenced.add(entry["checkpoint"])
        return referenced

    def _collect_garbage(self) -> None:
        """Delete only unreferenced ``epoch_NNN.pth`` snapshots.

        Never deletes ``model_checkpoint.pth``, files outside
        ``best-epochs/``, referenced snapshots, or anything that is not a
        plain ``epoch_NNN.pth`` file inside the snapshot directory. When no
        valid manifest exists (lost or unreadable), nothing is deleted: GC
        must never guess which snapshots are unreferenced.
        """
        if not self.best_epochs_dir.is_dir():
            return
        if not self.manifest_path.is_file():
            return
        referenced = self._referenced_checkpoints()
        if not referenced:
            # Manifest exists but could not be validated; be conservative.
            return
        referenced_names = {PurePosixPath(path).name for path in referenced}
        for entry in sorted(self.best_epochs_dir.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                continue
            if parse_snapshot_filename(entry.name) is None:
                continue
            if entry.name in referenced_names:
                continue
            entry.unlink()

    # -- strict loading ----------------------------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        """Strictly load and validate the manifest from disk.

        See :func:`load_best_selection` for the validation contract. Raises
        :class:`ManifestValidationError` for unknown schema versions,
        malformed or escaping checkpoint paths, missing best-overall fields,
        and missing snapshot files — but never for a stale symlink (that is
        repaired instead, test 9).
        """
        return load_best_selection(self.run_dir)

    def load_snapshot(self, epoch: int) -> dict:
        """Load one snapshot through ``torch.load`` (weights-safe context)."""
        return load_sweep_snapshot(self.snapshot_path(epoch))


def load_sweep_snapshot(path: str | os.PathLike) -> dict:
    """Load a sweep snapshot from ``path`` with strict structure checks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sweep snapshot not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ManifestValidationError(f"snapshot {path} does not contain a model_state_dict")
    if "epoch" not in payload:
        raise ManifestValidationError(f"snapshot {path} does not record its epoch")
    return payload


def load_best_selection(
    run_dir: str | os.PathLike,
    *,
    repair_symlink: bool = True,
) -> dict[str, Any]:
    """Strictly load and validate ``best-selection.json`` from ``run_dir``.

    The loader validates the schema version, required best-overall fields,
    per-variant entries, and every checkpoint path (relative, in-run-folder,
    existing). It never invents or repairs selection values; the only repair
    it performs is re-pointing a stale or missing ``model_best.pth`` symlink
    at the manifest's canonical checkpoint (when ``repair_symlink=True``).

    Raises :class:`ManifestValidationError` (a ``ValueError``) on any
    violation, with a message describing the exact problem.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / BEST_SELECTION_FILENAME
    if not manifest_path.is_file():
        raise ManifestValidationError(f"selection manifest not found: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            f"selection manifest is not valid JSON: {manifest_path}"
        ) from exc

    _validate_manifest_structure(manifest)
    _validate_manifest_values(manifest, run_dir)

    if repair_symlink:
        best_model_path = run_dir / BEST_MODEL_FILENAME
        target = manifest["best_overall"]["checkpoint"]
        current = os.readlink(best_model_path) if best_model_path.is_symlink() else None
        if current != target:
            if best_model_path.exists() and not best_model_path.is_symlink():
                raise ExistingBestModelError(
                    f"{best_model_path} is a regular file; refusing to repair "
                    "the symlink over it (back up or migrate the file first)."
                )
            temp_link = best_model_path.with_name(f".{BEST_MODEL_FILENAME}.tmp-{os.getpid()}")
            try:
                if temp_link.is_symlink() or temp_link.exists():
                    temp_link.unlink()
                os.symlink(target, temp_link)
                os.replace(temp_link, best_model_path)
            except BaseException:
                if temp_link.is_symlink() or temp_link.exists():
                    temp_link.unlink()
                raise

    return manifest
