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
import sys

import math
import numbers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from torch import nn

logger = logging.getLogger(__name__)

from clarity.localization.sweep.spec import (
    BEST_MODEL_FILENAME,
    BEST_SELECTION_FILENAME,
    LocalizationSweepSpec,
    ManifestValidationError,
    parse_variant_key,
)

NON_MANIFEST_WEIGHT_NAMES = frozenset({"dino", "dinov3"})


def sweep_score(auc_roc_macro: float, iou_accuracy_macro: float) -> float:
    """Return the deterministic sweep score.

    The score is the historical checkpoint-selection score::

        0.5 * (auc_roc_macro + IoU_accuracy_macro)

    Both inputs must be finite numbers; ``nan`` and infinities are rejected so
    that a broken evaluation can never win checkpoint selection.
    """
    for name, value in (
        ("auc_roc_macro", auc_roc_macro),
        ("iou_accuracy_macro", iou_accuracy_macro),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"{name} must be a real number, got {type(value).__name__} " f"{value!r}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    return 0.5 * (float(auc_roc_macro) + float(iou_accuracy_macro))


def is_better_sweep_candidate(
    challenger_epoch: int,
    challenger_attention_layer_count: int,
    challenger_score: float,
    incumbent_epoch: int | None = None,
    incumbent_attention_layer_count: int | None = None,
    incumbent_score: float | None = None,
) -> bool:
    """Deterministic comparison of a challenger against an incumbent pair.

    Ordering rules, applied in order:

    1. higher score wins;
    2. exact tie on score: the earlier epoch wins;
    3. same score and epoch: the smaller attention-layer count wins.

    This is the single source of truth for training-time selection and for
    reporting, so ties never depend on dictionary iteration order.
    """
    for name, value in (
        ("challenger_epoch", challenger_epoch),
        ("challenger_attention_layer_count", challenger_attention_layer_count),
        ("incumbent_epoch", incumbent_epoch),
        (
            "incumbent_attention_layer_count",
            incumbent_attention_layer_count,
        ),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")

    if incumbent_score is None or incumbent_epoch is None:
        return True
    if incumbent_score != incumbent_score:  # NaN guard
        return True

    if challenger_score > incumbent_score:
        return True
    if challenger_score < incumbent_score:
        return False
    if challenger_epoch != incumbent_epoch:
        return challenger_epoch < incumbent_epoch
    return challenger_attention_layer_count < incumbent_attention_layer_count


def manifest_selection_for_weights(
    weights_path: str | os.PathLike,
    explicit_count: int | None = None,
) -> dict[str, Any] | None:
    """Resolve the localization attention-layer count for a weights path.

    This is the single discovery/resolution entry point used by standalone
    evaluation. The rules are:

    - ``explicit_count`` (an explicit ``--attention-layer-count`` override)
      always wins. It is validated as a positive integer against the
      manifest's recorded sweep (when a manifest applies) purely to report
      ``override_outside_recorded_sweep``; no manifest is required.
    - otherwise, when the weights file is named ``model_best.pth`` (checked
      on the given path name, *without* resolving away a possible symlink
      first), ``best-selection.json`` is loaded strictly from that file's
      parent directory (the run folder). No ancestor directories are
      searched, and named pretrained weights never trigger a search.
    - a manifest that exists but is malformed or inconsistent raises
      :class:`ManifestValidationError` (a ``ValueError``): failing clearly is
      preferred over silently falling back to the model default and
      producing a mislabeled evaluation.

    ``model_best.pth`` must resolve (through the symlink) to the manifest's
    best-overall checkpoint; a stale or missing symlink is repaired only
    through the tested loader (:func:`load_best_selection` with
    ``repair_symlink=True``).

    Returns ``None`` when no manifest applies (historical behavior:
    the caller keeps the model/default ``n_attn_layers_avg``), otherwise a
    plain dict::

        {
            "n_attn_layers_avg": 6,           # selected count
            "manifest_path": str | None,      # None for explicit overrides
            "manifest_best_epoch": int | None,
            "recorded_sweep_counts": tuple[int, ...] | None,
            "override_outside_recorded_sweep": bool,
        }
    """
    weights_path = Path(weights_path)

    ## Explicit override: validate and report without requiring a manifest
    if explicit_count is not None:
        if isinstance(explicit_count, bool) or not isinstance(explicit_count, int):
            raise ValueError(
                f"explicit attention layer count must be an integer, got " f"{explicit_count!r}"
            )
        if explicit_count < 1:
            raise ValueError(
                f"explicit attention layer count must be >= 1, got " f"{explicit_count!r}"
            )
        selection: dict[str, Any] | None = None
        if weights_path.name == BEST_MODEL_FILENAME:
            manifest_path = weights_path.parent / BEST_SELECTION_FILENAME
            if manifest_path.is_file():
                from clarity.localization.sweep.manager import load_best_selection

                manifest = load_best_selection(weights_path.parent)
                counts = tuple(manifest.get("attention_layer_counts", ()))
                selection = {
                    "n_attn_layers_avg": explicit_count,
                    "manifest_path": str(manifest_path),
                    "manifest_best_epoch": manifest["best_overall"]["epoch"],
                    "recorded_sweep_counts": counts or None,
                    "override_outside_recorded_sweep": bool(counts)
                    and (explicit_count not in counts),
                }
        if selection is None:
            selection = {
                "n_attn_layers_avg": explicit_count,
                "manifest_path": None,
                "manifest_best_epoch": None,
                "recorded_sweep_counts": None,
                "override_outside_recorded_sweep": False,
            }
        return selection

    ## No explicit override: discover the run-folder manifest, if any
    if weights_path.name != BEST_MODEL_FILENAME:
        # Named pretrained weights or arbitrary checkpoints: never search.
        return None

    manifest_path = weights_path.parent / BEST_SELECTION_FILENAME
    if not manifest_path.is_file():
        # Legacy real checkpoint without sweep artifacts: historical behavior.
        return None

    # The strict loader repairs a stale/missing symlink from the manifest and
    # fails loudly on malformed manifests, missing snapshots, or schema
    # errors — never silently falling back to a default count.
    from clarity.localization.sweep.manager import load_best_selection

    manifest = load_best_selection(weights_path.parent)

    # Consistency: the requested weights file must be the manifest's
    # best-overall checkpoint (or a symlink to it).
    best_checkpoint = manifest["best_overall"]["checkpoint"]
    resolved_weights = weights_path.resolve()
    resolved_best = (weights_path.parent / best_checkpoint).resolve()
    if resolved_weights != resolved_best:
        raise ManifestValidationError(
            f"{weights_path} does not resolve to the manifest's best-overall "
            f"checkpoint {best_checkpoint!r} (resolved to "
            f"{resolved_best}); refusing to evaluate it with the manifest's "
            "selected attention layer count"
        )

    return {
        "n_attn_layers_avg": int(manifest["best_overall"]["attention_layer_count"]),
        "manifest_path": str(manifest_path),
        "manifest_best_epoch": int(manifest["best_overall"]["epoch"]),
        "recorded_sweep_counts": tuple(manifest.get("attention_layer_counts", ())) or None,
        "override_outside_recorded_sweep": False,
    }


def resolve_localization_attention_layer_count(
    model: nn.Module,
    weights_path: str | os.PathLike,
    explicit_count: int | None = None,
    legacy_model_kwarg_count: int | None = None,
) -> dict[str, Any]:
    """Resolve the localization attention-layer count for one evaluation.

    Precedence:

    1. ``explicit_count`` — the explicit ``--attention-layer-count`` CLI
       argument;
    2. ``legacy_model_kwarg_count`` — an equivalent localization kwarg
       (e.g. ``-mkw/-lckw n_attn_layers_avg=...``) retained for API callers;
       a conflict with the explicit count is rejected loudly (the explicit
       argument wins, with a clear warning printed to stderr for CLI flows
       that catch the error);
    3. the ``best-selection.json`` best-overall count when the weights path
       is a sweep run's ``model_best.pth`` (see
       :func:`manifest_selection_for_weights`);
    4. otherwise the model's own ``n_attn_layers_avg`` (historical behavior).

    Returns a plain dict combining the selection decision with the metadata
    required by the evaluation output::

        {
            "n_attn_layers_avg": int,         # count to use for this call
            "source": "manifest" | "explicit_cli" | "legacy_model_kwarg"
                      | "default",
            "manifest_path": str | None,
            "manifest_best_epoch": int | None,
            "override_outside_recorded_sweep": bool,
        }

    An explicit count is accepted for any value up to the model depth,
    including counts outside the recorded sweep (reported through
    ``override_outside_recorded_sweep``).
    """
    if (
        explicit_count is not None
        and legacy_model_kwarg_count is not None
        and explicit_count != legacy_model_kwarg_count
    ):
        print(
            f"WARNING: conflicting localization attention-layer counts: the "
            f"explicit attention layer count {explicit_count} wins over the "
            f"legacy model/localization kwarg n_attn_layers_avg="
            f"{legacy_model_kwarg_count}; the model kwarg is NOT used.",
            file=sys.stderr,
        )

    ## 1. explicit CLI override
    if explicit_count is not None:
        selection = manifest_selection_for_weights(weights_path, explicit_count=explicit_count)
        return {
            "n_attn_layers_avg": explicit_count,
            "source": "explicit_cli",
            "manifest_path": selection["manifest_path"],
            "manifest_best_epoch": selection["manifest_best_epoch"],
            "override_outside_recorded_sweep": selection["override_outside_recorded_sweep"],
        }

    ## 2. legacy equivalent localization kwarg (API callers)
    if legacy_model_kwarg_count is not None:
        return {
            "n_attn_layers_avg": legacy_model_kwarg_count,
            "source": "legacy_model_kwarg",
            "manifest_path": None,
            "manifest_best_epoch": None,
            "override_outside_recorded_sweep": False,
        }

    ## 3. manifest selection for a sweep run's model_best.pth
    selection = manifest_selection_for_weights(weights_path)
    if selection is not None:
        return {
            "n_attn_layers_avg": selection["n_attn_layers_avg"],
            "source": "manifest",
            "manifest_path": selection["manifest_path"],
            "manifest_best_epoch": selection["manifest_best_epoch"],
            "override_outside_recorded_sweep": selection["override_outside_recorded_sweep"],
        }

    ## 4. historical model/default behavior
    return {
        "n_attn_layers_avg": int(getattr(model, "n_attn_layers_avg", 8)),
        "source": "default",
        "manifest_path": None,
        "manifest_best_epoch": None,
        "override_outside_recorded_sweep": False,
    }


def localization_selection_metadata(
    model: nn.Module,
    weights_path: str | os.PathLike,
    explicit_count: int | None = None,
    legacy_model_kwarg_count: int | None = None,
) -> dict[str, Any]:
    """Resolve the selection and return the pickle-ready metadata block.

    This is the metadata half of :func:`resolve_localization_attention_layer_count`
    (same precedence), shaped exactly for the evaluation output's
    ``localization_selection`` key. Classification-only evaluation may call
    this to report selection metadata without generating attention maps.
    """
    selection = resolve_localization_attention_layer_count(
        model=model,
        weights_path=weights_path,
        explicit_count=explicit_count,
        legacy_model_kwarg_count=legacy_model_kwarg_count,
    )
    return {
        "source": selection["source"],
        "n_attn_layers_avg": selection["n_attn_layers_avg"],
        "manifest_path": selection["manifest_path"],
        "manifest_best_epoch": selection["manifest_best_epoch"],
        "override_outside_recorded_sweep": selection["override_outside_recorded_sweep"],
    }


@dataclass
class _VariantBest:
    """Best-so-far selection for one attention-layer count variant."""

    epoch: int
    score: float
    auc_roc_macro: float
    iou_accuracy_macro: float


@dataclass
class _EpochObservation:
    """Raw per-epoch scores for all variants, before comparison."""

    epoch: int
    auc_roc_macro: float
    variant_scores: dict[str, float]  # variant_key -> combined score
    variant_iou_accuracy: dict[str, float]  # variant_key -> IoU_accuracy_macro


class SweepSelectionState:
    """Best-per-variant and best-overall selection state for one run.

    Initialized from a validated :class:`LocalizationSweepSpec` and updated
    once per epoch with :meth:`update`. All comparisons go through the
    deterministic :func:`is_better_sweep_candidate`, so ties prefer the
    earlier epoch and then the smaller attention-layer count regardless of
    dictionary iteration order. Epoch 0 is eligible; the historical
    ``epoch > 0`` exclusion of the ordinary training loop is deliberately
    not reproduced here.
    """

    def __init__(self, spec: LocalizationSweepSpec) -> None:
        if not isinstance(spec, LocalizationSweepSpec):
            raise ValueError(f"spec must be a LocalizationSweepSpec, got {type(spec).__name__}")
        self.spec = spec
        #: ``None`` until the first epoch observation has been recorded.
        self.epoch: int | None = None
        self.epoch_auc_roc_macro: float | None = None
        self.epoch_scores: dict[str, float] = {}
        self.epoch_iou_accuracy: dict[str, float] = {}
        #: variant_key -> _VariantBest
        self.best_variants: dict[str, _VariantBest] = {}
        #: ``None`` until a first best-overall pair exists.
        self.best_overall: dict[str, Any] | None = None
        #: Set by :meth:`update`: whether the overall pair improved.
        self.overall_improved: bool = False
        #: Variant keys improved at the current epoch (sorted by layer count).
        self.improved_variants: list[str] = []

    # -- per-epoch update ----------------------------------------------------

    def update(
        self,
        epoch: int,
        auc_roc_macro: float,
        variant_iou_accuracy: Mapping[str, float],
    ) -> None:
        """Record one epoch's scores and advance the selection state.

        ``variant_iou_accuracy`` maps each canonical variant key
        (:func:`variant_key`) to that variant's ``IoU_accuracy_macro``; the
        combined score is computed once per variant with
        :func:`sweep_score`. Every configured variant must be present.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
        if self.epoch is not None and epoch <= self.epoch:
            raise ValueError(
                f"epochs must be strictly increasing, got {epoch} after " f"{self.epoch}"
            )
        expected_keys = set(self.spec.variant_keys)
        provided_keys = set(variant_iou_accuracy)
        if provided_keys != expected_keys:
            missing = sorted(expected_keys - provided_keys)
            unknown = sorted(provided_keys - expected_keys)
            raise ValueError(
                "variant_iou_accuracy keys must exactly match the sweep "
                f"variants; missing={missing}, unknown={unknown}"
            )

        scores: dict[str, float] = {}
        for key in self.spec.variant_keys:
            scores[key] = sweep_score(auc_roc_macro, variant_iou_accuracy[key])

        self.epoch = int(epoch)
        self.epoch_auc_roc_macro = float(auc_roc_macro)
        self.epoch_scores = scores
        self.epoch_iou_accuracy = {
            key: float(variant_iou_accuracy[key]) for key in self.spec.variant_keys
        }

        improved: list[str] = []
        for key in self.spec.variant_keys:
            count = parse_variant_key(key)
            score = scores[key]
            incumbent = self.best_variants.get(key)
            if is_better_sweep_candidate(
                challenger_epoch=epoch,
                challenger_attention_layer_count=count,
                challenger_score=score,
                incumbent_epoch=None if incumbent is None else incumbent.epoch,
                incumbent_attention_layer_count=None if incumbent is None else count,
                incumbent_score=None if incumbent is None else incumbent.score,
            ):
                self.best_variants[key] = _VariantBest(
                    epoch=epoch,
                    score=score,
                    auc_roc_macro=float(auc_roc_macro),
                    iou_accuracy_macro=float(variant_iou_accuracy[key]),
                )
                improved.append(key)
        # Deterministic order (ascending layer count), not insertion order.
        self.improved_variants = improved

        # Best overall: compare every variant of this epoch against the
        # incumbent pair with the single deterministic comparator.
        overall_improved = False
        if self.best_overall is None:
            challenger_epoch = None
            challenger_count = None
            challenger_score = None
        else:
            challenger_epoch = self.best_overall["epoch"]
            challenger_count = self.best_overall["attention_layer_count"]
            challenger_score = self.best_overall["score"]
        for key in self.spec.variant_keys:
            count = parse_variant_key(key)
            if is_better_sweep_candidate(
                challenger_epoch=epoch,
                challenger_attention_layer_count=count,
                challenger_score=scores[key],
                incumbent_epoch=challenger_epoch,
                incumbent_attention_layer_count=challenger_count,
                incumbent_score=challenger_score,
            ):
                self.best_overall = {
                    "epoch": epoch,
                    "attention_layer_count": count,
                    "score": scores[key],
                }
                challenger_epoch = epoch
                challenger_count = count
                challenger_score = scores[key]
                overall_improved = True
        self.overall_improved = overall_improved

    # -- reporting helpers ----------------------------------------------------

    @property
    def any_improvement(self) -> bool:
        """Whether any variant improved at the current epoch."""
        return bool(self.improved_variants)

    def best_variant_epoch(self, key: str) -> int:
        """Return the best epoch recorded for a variant key."""
        return self.best_variants[key].epoch

    def snapshot_state(self) -> dict[str, Any]:
        """Plain-Python summary for ``performance_curves`` logging."""
        return {
            "epoch": self.epoch,
            "improved_variants": list(self.improved_variants),
            "epoch_scores": dict(self.epoch_scores),
            "best_scores": {key: best.score for key, best in self.best_variants.items()},
            "best_epochs": {key: best.epoch for key, best in self.best_variants.items()},
            "best_overall": None if self.best_overall is None else dict(self.best_overall),
        }
