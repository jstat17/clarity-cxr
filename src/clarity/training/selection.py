"""Localization controls, sweep-selection logging and sweep table printing."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from clarity.localization.artifacts import calibration_thresholds, localization_method_block
from clarity.localization.sweep import LocalizationSweepSpec, parse_variant_key

logger = logging.getLogger("clarity.training.loop")


def _resolve_localization_controls(localization_kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """Split the explicit method controls out of the free-form kwargs.

    Returns ``(extractor_kwargs, controls)`` where ``controls`` holds
    ``bbox_method``/``selection``/``optimize_metrics_separately``/
    ``fit_workers``/``only_correctly_classified`` for the evaluator wrappers.
    The defaults keep the earlier behaviour when a caller supplies only
    the legacy ``-lckw`` dictionary.
    """
    kwargs = dict(localization_kwargs or {})
    kwargs.pop("use_amp", None)
    controls = {
        "bbox_method": kwargs.pop("bbox_method", "onebox"),
        "selection": kwargs.pop("selection", None),
        "optimize_metrics_separately": bool(kwargs.pop("optimize_metrics_separately", False)),
        "fit_workers": kwargs.pop("fit_workers", None),
        "only_correctly_classified": bool(kwargs.pop("only_correctly_classified", False)),
    }
    return kwargs, controls


def _localization_log_fields(artifact) -> tuple[dict, dict, dict | None]:
    """Method metadata, per-finding params and legacy thresholds for a log.

    ``localization_heatmap_thr`` is ``None`` for OneBox: the params block
    is authoritative.  Legacy ``heatmap_thr`` dicts and SingleCut
    calibrations keep their scalar dicts.
    """
    block = localization_method_block(artifact)
    method = {
        "bbox_method": block.get("bbox_method"),
        "selection": block.get("selection"),
        "selection_metric": block.get("selection_metric"),
        "fit_seconds": block.get("fit_seconds"),
        "fit_workers": block.get("fit_workers"),
        "val5": block.get("val5"),
    }
    if block.get("optimize_metrics_separately"):
        method["optimize_metrics_separately"] = True
    params = {int(k): v for k, v in (block.get("params") or {}).items()}
    if block.get("bbox_method") == "onebox":
        thresholds = None
    else:
        thresholds = calibration_thresholds(artifact) or None
    return method, params, thresholds


def _legacy_localization_log_entry(family_results: dict[str, tuple]) -> defaultdict:
    """Build the historical one-variant localization log entry.

    ``family_results`` maps ``'IoBB'``/``'IoU'`` to the localization
    evaluator's ``(metrics, class_metrics, proxy_metrics, heatmap_thr)``
    4-tuple. The returned structure matches the pre-sweep
    ``performance_curves['localization']`` schema exactly, so existing
    one-variant consumers keep working in sweep mode.
    """
    localization_eval = defaultdict(dict)

    for metric, (
        local_metrics,
        local_metrics_class,
        local_proxy_class,
        local_artifact,
    ) in family_results.items():
        method, params, thresholds = _localization_log_fields(local_artifact)
        localization_eval[metric]["localization_class"] = local_metrics_class
        localization_eval[metric]["localization_proxy_class"] = local_proxy_class
        localization_eval[metric]["localization_heatmap_thr"] = thresholds
        localization_eval[metric]["localization_method"] = method
        localization_eval[metric]["localization_params"] = params

        for local_metric, value in local_metrics.items():
            localization_eval[metric][local_metric] = value

    return localization_eval


def _sweep_log_entry(
    spec: LocalizationSweepSpec,
    sweep_results: dict[str, dict[str, tuple]],
) -> dict:
    """Complete per-variant IoBB/IoU result blocks for the sweep log.

    One entry per canonical variant key, each holding the variant's layer
    count and the full historical 4-tuple content (macro metrics, class
    metrics, proxy metrics, and heatmap thresholds) for both metric
    families.
    """
    entry: dict[str, Any] = {}
    for key in spec.variant_keys:
        variant_entry: dict[str, Any] = {"attention_layer_count": parse_variant_key(key)}
        for metric, (
            local_metrics,
            local_metrics_class,
            local_proxy_class,
            local_artifact,
        ) in sweep_results[key].items():
            method, params, thresholds = _localization_log_fields(local_artifact)
            block = {
                "localization_class": local_metrics_class,
                "localization_proxy_class": local_proxy_class,
                "localization_heatmap_thr": thresholds,
                "localization_method": method,
                "localization_params": params,
            }
            block.update(local_metrics)
            variant_entry[metric] = block
        entry[key] = variant_entry
    return entry


def _sweep_selection_log_entry(
    spec: LocalizationSweepSpec,
    state: Any,
    timings: dict[str, float] | None,
) -> dict:
    """Per-epoch selection summary for ``localization_sweep_selection``.

    Holds the epoch, primary variant, per-variant epoch scores, per-variant
    best scores and epochs (after the update), the best-overall state, the
    variants improved this epoch, and the sweep evaluator timing metadata.
    """
    entry = state.snapshot_state()
    entry["primary_variant"] = spec.primary_variant_key
    entry["timing_seconds"] = None if timings is None else dict(timings)
    return entry


def _print_sweep_table(
    epoch: int,
    spec: LocalizationSweepSpec,
    sweep_results: dict[str, dict[str, tuple]],
    state: Any,
) -> None:
    """Print one concise row per sweep variant for the current epoch."""
    logger.info(f"Exact localization sweep (epoch {epoch})")
    logger.info(
        f"{'count':>5}  {'IoBB_acc':>8}  {'IoU_acc':>8}  "
        f"{'score':>8}  {'best':>8}  {'best_ep':>7}"
    )
    for key in spec.variant_keys:
        count = parse_variant_key(key)
        iobb_acc = sweep_results[key]["IoBB"][0]["IoBB_accuracy_macro"]
        iou_acc = sweep_results[key]["IoU"][0]["IoU_accuracy_macro"]
        score = state.epoch_scores[key]
        best = state.best_variants[key]
        logger.info(
            f"{count:>5}  {iobb_acc:>8.4f}  {iou_acc:>8.4f}  "
            f"{score:>8.4f}  {best.score:>8.4f}  {best.epoch:>7d}"
        )
    overall = state.best_overall
    logger.info(
        f"Best overall: epoch {overall['epoch']}, "
        f"count {overall['attention_layer_count']}, "
        f"score {overall['score']:.4f}"
    )
