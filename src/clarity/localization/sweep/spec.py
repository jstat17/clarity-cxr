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

import numbers
from dataclasses import dataclass
from typing import Any, Dict, Iterable, TypedDict

from torch import nn

logger = logging.getLogger(__name__)


BEST_SELECTION_FILENAME = "best-selection.json"


BEST_EPOCHS_DIRNAME = "best-epochs"


BEST_MODEL_FILENAME = "model_best.pth"


SELECTION_SCHEMA_VERSION = 1


VARIANT_KEY_PREFIX = "n_attn_layers_avg="


SELECTION_METRIC_MEAN_AUC_IOU = "mean_auc_iou"


SUPPORTED_SELECTION_METRICS = frozenset({SELECTION_METRIC_MEAN_AUC_IOU})


EARLY_STOPPING_BEST_OVERALL = "best_overall"


SUPPORTED_EARLY_STOPPING_POLICIES = frozenset({EARLY_STOPPING_BEST_OVERALL})


SWEEP_CAPABILITY_ATTRIBUTE = "supports_localization_sweep"


SELECTION_METRIC_DESCRIPTION = "0.5*(auc_roc_macro+IoU_accuracy_macro)"


@dataclass(frozen=True)
class LocalizationSweepSpec:
    """Immutable specification of an attention-layer-count sweep.

    ``attention_layer_counts`` must already be a sorted, unique tuple of
    positive integers; use :meth:`from_counts` to normalize a parsed
    list/tuple, or :meth:`for_clarity_s` for the default CLARiTy-S sweep.

    ``exact=True`` is required: the current feature only supports exact
    training-hyperparameter semantics (a pure evaluation sweep over shared
    weights). Approximate or sampled sweeps must not be silently enabled.
    """

    attention_layer_counts: tuple[int, ...]
    primary_attention_layer_count: int
    selection_metric: str = SELECTION_METRIC_MEAN_AUC_IOU
    early_stopping_policy: str = EARLY_STOPPING_BEST_OVERALL
    exact: bool = True

    def __post_init__(self) -> None:
        counts = self.attention_layer_counts
        if not isinstance(counts, tuple):
            raise ValueError(f"attention_layer_counts must be a tuple, got {type(counts).__name__}")
        if len(counts) == 0:
            raise ValueError("attention_layer_counts must not be empty")
        for count in counts:
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    "attention_layer_counts must contain only integers, got " f"{count!r}"
                )
            if count < 1:
                raise ValueError(
                    "attention_layer_counts must contain only values >= 1, got " f"{count!r}"
                )
        for earlier, later in zip(counts, counts[1:]):
            if later <= earlier:
                raise ValueError(
                    "attention_layer_counts must be sorted and unique, got " f"{counts!r}"
                )

        primary = self.primary_attention_layer_count
        if isinstance(primary, bool) or not isinstance(primary, int):
            raise ValueError(
                "primary_attention_layer_count must be an integer, got " f"{primary!r}"
            )
        if primary not in counts:
            raise ValueError(
                f"primary_attention_layer_count {primary} is not part of the "
                f"sweep attention_layer_counts {counts!r}"
            )

        if self.exact is not True:
            raise ValueError("only exact sweeps are supported, got exact=" f"{self.exact!r}")

        if self.selection_metric not in SUPPORTED_SELECTION_METRICS:
            raise ValueError(
                f"unsupported selection_metric {self.selection_metric!r}; "
                f"supported values: {sorted(SUPPORTED_SELECTION_METRICS)}"
            )

        if self.early_stopping_policy not in SUPPORTED_EARLY_STOPPING_POLICIES:
            raise ValueError(
                "unsupported early_stopping_policy "
                f"{self.early_stopping_policy!r}; supported values: "
                f"{sorted(SUPPORTED_EARLY_STOPPING_POLICIES)}"
            )

    # -- factories ----------------------------------------------------------

    @classmethod
    def from_counts(
        cls,
        counts: Iterable[Any],
        primary_attention_layer_count: int,
        *,
        selection_metric: str = SELECTION_METRIC_MEAN_AUC_IOU,
        early_stopping_policy: str = EARLY_STOPPING_BEST_OVERALL,
        exact: bool = True,
    ) -> "LocalizationSweepSpec":
        """Normalize a parsed list/tuple of counts into a valid spec.

        The counts are normalized to a sorted, unique tuple. Duplicates are
        rejected rather than silently hidden so that a typo (for example
        ``[2, 2, 4]``) fails loudly before any training begins.
        """
        if counts is None or isinstance(counts, (str, bytes)):
            raise ValueError(
                f"attention_layer_counts must be an iterable of integers, got " f"{counts!r}"
            )
        if not isinstance(counts, Iterable):
            raise ValueError(
                f"attention_layer_counts must be an iterable of integers, got "
                f"{type(counts).__name__} {counts!r}"
            )

        values = list(counts)
        if len(values) == 0:
            raise ValueError("attention_layer_counts must not be empty")

        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "attention_layer_counts must contain only integers, got " f"{value!r}"
                )

        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(
                f"attention_layer_counts contains duplicate values {duplicates!r}; "
                "every count must appear exactly once"
            )

        return cls(
            attention_layer_counts=tuple(sorted(set(values))),
            primary_attention_layer_count=primary_attention_layer_count,
            selection_metric=selection_metric,
            early_stopping_policy=early_stopping_policy,
            exact=exact,
        )

    @classmethod
    def for_clarity_s(cls, depth: int = 12) -> "LocalizationSweepSpec":
        """Default CLARiTy-S sweep: every count ``1..depth`` with primary 8.

        For ``depth=12`` this is
        ``(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)`` with primary count 8.
        For a shallower model the primary count is clamped to the depth so
        that it always remains part of the sweep.
        """
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"model depth must be an integer, got {depth!r}")
        if depth < 1:
            raise ValueError(f"model depth must be >= 1, got {depth!r}")
        return cls.from_counts(
            tuple(range(1, depth + 1)),
            primary_attention_layer_count=min(8, depth),
        )

    # -- validation ---------------------------------------------------------

    def validate_for_depth(self, depth: int) -> None:
        """Raise ``ValueError`` if any count exceeds the model depth."""
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"model depth must be an integer, got {depth!r}")
        if depth < 1:
            raise ValueError(f"model depth must be >= 1, got {depth!r}")
        for count in self.attention_layer_counts:
            if count > depth:
                raise ValueError(f"attention layer count {count} exceeds model depth " f"{depth}")

    # -- derived views ------------------------------------------------------

    @property
    def variant_keys(self) -> tuple[str, ...]:
        """Variant keys in sweep order (ascending layer count)."""
        return tuple(variant_key(count) for count in self.attention_layer_counts)

    @property
    def primary_variant_key(self) -> str:
        """Variant key of the primary (historical) layer count."""
        return variant_key(self.primary_attention_layer_count)


def variant_key(attention_layer_count: int) -> str:
    """Return the canonical variant key ``n_attn_layers_avg=<count>``."""
    if isinstance(attention_layer_count, bool) or not isinstance(attention_layer_count, int):
        raise ValueError(
            f"attention layer count must be an integer, got " f"{attention_layer_count!r}"
        )
    if attention_layer_count < 1:
        raise ValueError(f"attention layer count must be >= 1, got {attention_layer_count!r}")
    return f"{VARIANT_KEY_PREFIX}{attention_layer_count}"


def parse_variant_key(key: str) -> int:
    """Parse a canonical variant key into its attention-layer count.

    Parsing is strict: only the exact ``n_attn_layers_avg=<digits>`` format is
    accepted, with no whitespace, signs, leading zeros, or extra characters.
    """
    if not isinstance(key, str):
        raise ValueError(f"variant key must be a string, got {type(key).__name__} {key!r}")
    if not key.startswith(VARIANT_KEY_PREFIX):
        raise ValueError(f"invalid variant key {key!r}: must start with {VARIANT_KEY_PREFIX!r}")
    suffix = key[len(VARIANT_KEY_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(
            f"invalid variant key {key!r}: count suffix {suffix!r} must be a "
            "positive decimal integer"
        )
    if len(suffix) > 1 and suffix[0] == "0":
        raise ValueError(
            f"invalid variant key {key!r}: count suffix {suffix!r} must not "
            "contain leading zeros"
        )
    count = int(suffix)
    if count < 1:
        raise ValueError(f"invalid variant key {key!r}: attention layer count must be >= 1")
    return count


def variant_keys(attention_layer_counts: Iterable[int]) -> tuple[str, ...]:
    """Return canonical variant keys for the given counts, in order."""
    return tuple(variant_key(count) for count in attention_layer_counts)


ScalarMetric = float | int | None


class LocalizationMetricBlock(TypedDict, total=False):
    """Macro localization metrics for one metric family (IoBB or IoU).

    Keys mirror ``evaluation.classification._localization_macro_metrics``; the block may
    also carry ``<proxy>_macro`` proxy entries. All values are JSON numbers or
    ``None`` for absent entries.
    """

    IoU_accuracy_macro: float
    IoU_atp_macro: float
    IoU_afp_macro: float
    IoU_tpr_macro: float
    IoU_fpr_macro: float
    IoBB_accuracy_macro: float
    IoBB_atp_macro: float
    IoBB_afp_macro: float
    IoBB_tpr_macro: float
    IoBB_fpr_macro: float


LocalizationFamilyResult = tuple[
    LocalizationMetricBlock,
    dict,
    dict,
    object,
]


class VariantLocalizationResult(TypedDict):
    """Complete per-variant localization result for one epoch.

    ``IoBB`` and ``IoU`` each hold the historical localization 4-tuple
    ``(metrics, best_detection, mean_proxy_abnorm, artifact)`` produced by
    the evaluator for that metric family (``artifact`` is the legacy
    ``heatmap_thr`` dict or the shared ``LocalizationCalibration``).
    """

    variant_key: str
    attention_layer_count: int
    IoBB: LocalizationFamilyResult
    IoU: LocalizationFamilyResult


class EpochSweepSelection(TypedDict):
    """Lightweight per-epoch selection state appended to
    ``performance_curves['localization_sweep_selection']``."""

    epoch: int
    primary_variant: str
    epoch_scores: Dict[str, float]
    best_scores: Dict[str, float]
    best_epochs: Dict[str, int]
    best_overall: "BestOverallState"
    timing_seconds: Dict[str, float]


class BestVariantState(TypedDict):
    """Best-so-far state for one variant."""

    epoch: int
    score: float


class BestOverallState(TypedDict):
    """Best ``(epoch, variant)`` pair across the whole sweep."""

    variant: str
    attention_layer_count: int
    epoch: int
    score: float


class ManifestVariantEntry(TypedDict):
    """Per-variant entry in ``best-selection.json``."""

    epoch: int
    score: float
    checkpoint: str


class SweepManifest(TypedDict):
    """Contents of ``best-selection.json``.

    ``sweep_counts`` holds the sweep specification; ``selection_metric`` is the
    human-readable description of the score; ``primary_variant`` records the
    designated primary variant key; ``variants`` is keyed by the decimal layer
    count as a string (``"2"``, ``"4"``, ...) for JSON friendliness.
    """

    schema_version: int
    selection_metric: str
    sweep_counts: list[int]
    primary_variant: str
    best_overall: BestOverallState
    variants: Dict[str, ManifestVariantEntry]


USES_FUSED_VIT_ATTENTION_METHOD = "uses_fused_vit_attention_for_training"


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module, transparently unwrapping DataParallel.

    Supports plain modules and ``nn.DataParallel``. Later code may inspect a
    documented capability attribute (see
    :data:`SWEEP_CAPABILITY_ATTRIBUTE`) on the returned model. No other
    wrapper types are handled; anything else is returned unchanged.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def has_sweep_capability(model: nn.Module) -> bool:
    """Return whether the unwrapped model advertises the sweep capability.

    The capability attribute (:data:`SWEEP_CAPABILITY_ATTRIBUTE`) must be
    truthy on the unwrapped model.
    """
    unwrapped = unwrap_model(model)
    return bool(getattr(unwrapped, SWEEP_CAPABILITY_ATTRIBUTE, False))


def uses_fused_vit_attention(loss: nn.Module) -> bool:
    """Whether a single loss module actively trains on the fused attention.

    The decision is never inferred from class names or tuple positions: the
    loss must explicitly implement the documented method
    (:data:`USES_FUSED_VIT_ATTENTION_METHOD`). Unknown losses default to
    ``False`` (not claiming the capability).
    """
    checker = getattr(loss, USES_FUSED_VIT_ATTENTION_METHOD, None)
    if not callable(checker):
        return False
    return bool(checker())


def unwrap_label_smoothing(loss: nn.Module) -> nn.Module:
    """Unwrap one level of a wrapping multi-label loss if present."""
    inner = getattr(loss, "classification_loss", None)
    return inner if isinstance(inner, nn.Module) else loss


def describe_criterion(loss: nn.Module) -> str:
    """Human-readable description of a criterion for rejection messages."""
    name = type(loss).__name__
    inner = unwrap_label_smoothing(loss)
    if inner is not loss:
        return f"{name} wrapping {type(inner).__name__}"
    return name


def attention_deltas_description(loss: nn.Module) -> str | None:
    """Active delta values of a criterion, when it exposes them.

    Returns ``None`` for losses that do not expose fused-attention deltas;
    otherwise a compact description such as ``delta1=1.0``.
    """
    inner = unwrap_label_smoothing(loss)
    deltas = getattr(inner, "effective_attention_deltas", None)
    if not callable(deltas):
        return None
    try:
        values = dict(deltas())
    except Exception:  # pragma: no cover - defensive
        return None
    active = {
        name: value
        for name, value in values.items()
        if isinstance(value, numbers.Real) and value != 0
    }
    if not active:
        return None
    return ", ".join(f"{name}={value:g}" for name, value in sorted(active.items()))


def find_attention_dependent_criterion(
    criteria: Iterable[nn.Module],
) -> nn.Module | None:
    """Return the first criterion that actively trains on fused attention.

    When multiple criteria rotate by epoch, a shared inference sweep is
    incompatible if **any** of them is attention-dependent, so the first
    offending criterion is returned (``None`` when all are safe).
    """
    for loss in criteria:
        if uses_fused_vit_attention(loss):
            return loss
    return None


class ManifestValidationError(ValueError):
    """Raised when a ``best-selection.json`` manifest is invalid.

    Raised for unknown schema versions, malformed or absolute/escaping
    checkpoint paths, missing required best-overall fields, and referenced
    snapshot files that do not exist. The strict loader never invents or
    repairs selection values; it may only repair the symlink (see
    :func:`load_best_selection`).
    """
