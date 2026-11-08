"""Attention-layer sweep spec construction for the training CLI."""

from __future__ import annotations

from typing import Any

from torch import nn


def _reject_sweep_configuration(message: str) -> None:
    """Reject the command with a user-facing message before training starts.

    Raises ``SystemExit`` carrying the message: when run as a script, the
    interpreter prints the message to stderr and exits with status 1, so a
    rejected command is both visible and clearly diagnosable. This must be
    called before any run folder is created.
    """
    raise SystemExit(message)


from clarity.localization.sweep import (
    LocalizationSweepSpec,
    attention_deltas_description,
    describe_criterion,
    find_attention_dependent_criterion,
    has_sweep_capability,
    unwrap_model,
)


def build_localization_sweep_spec(
    model: nn.Module,
    criteria: list[nn.Module],
    counts: list[int],
    primary: int | None,
    validate_localization: bool,
    localization_kwargs: dict[str, Any],
) -> LocalizationSweepSpec:
    """Build and fully validate the sweep spec; reject bad configurations.

    Performs every static check (primary membership, spec validity, model
    capability and depth, localization validation and method, inference
    attention type, exact mode, and active fused-attention training losses)
    and raises ``SystemExit`` with a user-facing message before any run
    folder is created.

    Note: the exact-mode and best-overall-policy requirements are enforced
    by :class:`LocalizationSweepSpec` itself (``exact=True`` and
    ``early_stopping_policy='best_overall'`` are the fixed defaults and any
    other value is rejected).
    """
    # primary is required with the sweep and must be one of the counts
    if primary is None:
        _reject_sweep_configuration(
            "-lasp/--localization_attention_primary is required when "
            "-las/--localization_attention_layer_sweep is given; the "
            "recommended value is 5 (e.g. '-las 1 2 3 4 5 6 7 8 9 10 11 12 "
            "-lasp 5')"
        )
    if primary not in counts:
        _reject_sweep_configuration(
            f"-lasp/--localization_attention_primary {primary} must be one "
            f"of the -las/--localization_attention_layer_sweep counts "
            f"{sorted(set(counts))}"
        )

    # spec-level validation (sorted/unique counts, metric, policy, exact)
    try:
        spec = LocalizationSweepSpec.from_counts(counts, primary_attention_layer_count=primary)
    except ValueError as exc:
        _reject_sweep_configuration(
            f"invalid localization attention-layer sweep configuration: {exc}"
        )

    # localization validation must be enabled with the configured method
    if not validate_localization:
        _reject_sweep_configuration(
            "-las/--localization_attention_layer_sweep requires localization "
            "validation (-vl/--validate_localization True); the sweep is "
            "evaluated on the localization validation split after every epoch"
        )
    localization_method = localization_kwargs.get("localization_method", "model_attention")
    if localization_method != "model_attention":
        _reject_sweep_configuration(
            "-las/--localization_attention_layer_sweep requires the "
            f"localization method 'model_attention', got "
            f"{localization_method!r} (from -lckw/--localization_kwargs)"
        )

    # the inference attention type used by the sweep evaluator is the one
    # configured through -lckw/--localization_kwargs (the extractor forwards it
    # as the single-call override; default 'attention', standard 'mask')
    # and must be supported by the model
    unwrapped_model = unwrap_model(model)
    inference_attention_type = str(
        localization_kwargs.get("inference_attention_type", "attention")
    ).casefold()
    supported_types = getattr(unwrapped_model, "SUPPORTED_INFERENCE_ATTENTION_TYPES", frozenset())
    if supported_types and inference_attention_type not in supported_types:
        _reject_sweep_configuration(
            "-las/--localization_attention_layer_sweep requires a supported "
            "inference_attention_type in -lckw/--localization_kwargs (the "
            "standard command uses 'mask'); got "
            f"{inference_attention_type!r}, supported values: "
            f"{sorted(supported_types)}"
        )

    # model must be CLARiTy and advertise the sweep capability
    if not str(getattr(model, "fullname", "")).startswith("CLARiTy") or not has_sweep_capability(
        unwrapped_model
    ):
        _reject_sweep_configuration(
            "-las/--localization_attention_layer_sweep requires a CLARiTy "
            "model with the localization sweep capability; got model with "
            f"fullname {getattr(model, 'fullname', None)!r}"
        )

    # counts must fit the actual model depth
    depth = len(unwrapped_model.blocks)
    try:
        spec.validate_for_depth(depth)
    except ValueError as exc:
        _reject_sweep_configuration(
            f"invalid localization attention-layer sweep configuration: {exc} "
            f"(model depth is {depth})"
        )

    # hard rejection: an active training loss on the fused ViT attention is
    # incompatible with a shared inference sweep over layer counts
    offender = find_attention_dependent_criterion(criteria)
    if offender is not None:
        _reject_sweep_configuration(
            _attention_loss_rejection_message(offender, spec, inference_attention_type)
        )

    return spec


def _attention_loss_rejection_message(
    offender: nn.Module,
    spec: LocalizationSweepSpec,
    inference_attention_type: str,
) -> str:
    """User-facing hard-rejection message for attention-dependent losses."""
    counts = list(spec.attention_layer_counts)
    deltas = attention_deltas_description(offender)
    deltas_part = f" (active deltas: {deltas})" if deltas else ""
    return (
        "A shared localization attention-layer sweep is incompatible with an "
        "active training loss on the fused ViT attention map: training "
        "back-propagates through the fused attention map, so its layer count "
        "cannot be varied at inference time without changing the trained "
        "hyperparameters.\n"
        f"Offending criterion: {describe_criterion(offender)}{deltas_part}.\n"
        f"Requested sweep counts: {counts} (primary "
        f"{spec.primary_attention_layer_count}, inference attention type "
        f"{inference_attention_type!r}).\n"
        "Choose one of:\n"
        "  1. disable the attention-loss branch (e.g. drop the delta "
        "coefficients) so training "
        "no longer consumes the fused ViT attention;\n"
        "  2. remove the sweep (-las/-lasp) and train with a single fixed "
        "attention-layer count;\n"
        "  3. run independent training jobs (one per attention-layer count) "
        "for an exact training-hyperparameter comparison."
    )
