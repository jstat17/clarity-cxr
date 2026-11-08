"""Localization evaluation: OneBox fitting/scoring and the method dispatcher.

The public OneBox path lives here.  The legacy SingleCut bodies are imported
from :mod:`clarity.localization.legacy` only when the caller explicitly selects
``bbox_method="singlecut"``.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from typing import Literal

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clarity.localization.constants import THRESHOLDS_SCORE
from clarity.localization.sweep import variant_key
from clarity.localization.fitting import (
    LocalizationCalibration,
    LocalizationMethodSpec,
    OneBoxFitPool,
    OneBoxFitter,
    build_calibration,
    resolve_method_spec,
    score_fixed_params,
    select_onebox,
    val5_macros,
)
from clarity.localization.metrics import (
    calculate_average_euclidean_distance,
    calculate_masked_intensity_ratio,
)
from clarity.localization.onebox import Instance
from clarity.localization.errors import DetectionWorkerError

def _resolve_localization_map_extractor(localization_method: str):
    """Resolve the map extractor for a method name.

    ``model_attention`` is the OneBox default and ``cam`` is HiResCAM
    (:mod:`clarity.localization.hirescam`), both public.  Other extractors are imported lazily, so importing this module stays free
    of the legacy stubs; they are not part of this release and raise
    ``NotImplementedError`` when requested.
    """
    method = localization_method.casefold()
    if method == "model_attention":
        return get_model_attention
    from clarity.localization.hirescam import get_cam_heatmaps
    from clarity.localization.legacy import (
        get_cam_fused_attention,
        get_mctfp_attention,
    )

    match method:
        case "cam":
            return get_cam_heatmaps
        case "mctfp_attention":
            return get_mctfp_attention
        case "cam_fused_attention":
            return get_cam_fused_attention
    raise ValueError(
        f"unknown localization_method {localization_method!r}; expected one of "
        "'cam', 'model_attention'"
    )

from clarity.localization.attention import get_model_attention, get_model_attention_sweep
from clarity.localization.artifacts import _params_representative_threshold, _params_to_dict


def _resolve_fit_workers(fit_workers: int | None, max_workers: int | None) -> int:
    """Actual OneBox fit pool size (default: ``min(-wmp, cpus)``).

    ``-fw`` wins when given; otherwise the evaluator's ``-wmp`` budget is
    capped by the CPUs actually available to the process (cgroup/affinity
    aware, so a scheduler allocation smaller than the host does not fork a
    host-sized pool).  The resolved count is persisted in the calibration's
    ``fit_workers`` field.
    """
    if fit_workers is not None:
        return max(1, int(fit_workers))
    budget = int(max_workers) if max_workers else 1
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    return max(1, min(budget, int(available)))


def _score_index_in_val5(score_thr: float) -> int:
    """Index of ``score_thr`` in the val5 score grid (the OneBox fit axis)."""
    scores = np.asarray(THRESHOLDS_SCORE, dtype=np.float64)
    idx = int(np.argmin(np.abs(scores - float(score_thr))))
    if not np.isclose(scores[idx], float(score_thr), rtol=0.0, atol=1e-9):
        raise ValueError(
            "OneBox reporting is defined on the val5 score grid "
            f"{list(THRESHOLDS_SCORE)}; got score threshold {score_thr!r}"
        )
    return idx


def _merge_partial_sums(target: dict, ab: int, partial) -> None:
    """Accumulate one worker partial (5 exact-integer arrays) per finding."""
    acc = target.get(int(ab))
    if acc is None:
        target[int(ab)] = [np.array(a, dtype=np.float64) for a in partial]
        return
    for dst, src in zip(acc, partial):
        dst += src


def _merge_fixed_counts(target: dict, ab: int, scored: dict) -> None:
    """Accumulate one instance's fixed-params counts per finding/score."""
    entry = target.setdefault(int(ab), {"n_inst": 0, "scores": {}})
    entry["n_inst"] += 1
    for score_thr, counts in scored["scores"].items():
        acc = entry["scores"].setdefault(
            float(score_thr),
            {
                "iou_num_detected": 0,
                "iou_num_missed": 0,
                "iou_is_detected": 0,
                "iobb_num_detected": 0,
                "iobb_num_missed": 0,
                "iobb_is_detected": 0,
            },
        )
        for key in acc:
            acc[key] += counts[key]


def _valstats_to_fixed_counts(
    stats_iou, stats_iobb, params_by_ab, n_inst, score_thresholds
) -> dict:
    """Build fixed-params counts from the fitted val sums at the selected config.

    ``sum_det`` is the per-instance detection indicator, so the number of
    detected boxes is recovered as ``sum_boxes - sum_missed`` (clamped at 0:
    an empty map carries ``sum_boxes=0`` and the standard empty-map miss of
    1).
    """
    index = {cfg.key: i for i, cfg in enumerate(stats_iou.configs)}
    out: dict = {}
    for ab in stats_iou.findings:
        fi = stats_iou.finding_pos[ab]
        ci = index[params_by_ab[ab].key]
        boxes = float(stats_iou.sum_boxes[ci, fi])
        entry = {"n_inst": int(n_inst[ab]), "scores": {}}
        for score_thr in score_thresholds:
            j = _score_index_in_val5(score_thr)
            iou_missed = float(stats_iou.sum_missed[ci, fi, j])
            iobb_missed = float(stats_iobb.sum_missed[ci, fi, j])
            entry["scores"][float(score_thr)] = {
                "iou_num_detected": max(0.0, boxes - iou_missed),
                "iou_num_missed": iou_missed,
                "iou_is_detected": float(stats_iou.sum_det[ci, fi, j]),
                "iobb_num_detected": max(0.0, boxes - iobb_missed),
                "iobb_num_missed": iobb_missed,
                "iobb_is_detected": float(stats_iobb.sum_det[ci, fi, j]),
            }
        out[int(ab)] = entry
    return out


def _build_fixed_params_best_detection(
    counts_by_ab: dict,
    params_by_ab: dict,
    score_thresholds: Iterable[float],
) -> dict:
    """Historical ``best_detection`` layout at frozen parameters.

    ``counts_by_ab[ab]`` holds ``n_inst`` and a ``scores`` dict keyed by score
    threshold with the aggregated per-instance counts of both families
    (``iou_*``/``iobb_*`` ``num_detected``/``num_missed``/``is_detected``).
    The representative ``HeatmapThreshold`` is ``threshold`` for SingleCut and
    ``t_hi`` for OneBox (documented; the real OneBox vector lives in the
    calibration artifact).  Metric formulas are identical to
    :func:`get_best_detection_by_heatmap_threshold`.
    """
    score_thresholds = list(score_thresholds)
    best_detection: dict = {}
    for ab, entry in counts_by_ab.items():
        params = params_by_ab[ab]
        rep = _params_representative_threshold(params)
        n_inst = max(int(entry["n_inst"]), 1)
        best_detection[int(ab)] = {}
        for score_thr in score_thresholds:
            counts = entry["scores"][float(score_thr)]
            metrics: dict = {}
            for family, key in (("IoU", "iou"), ("IoBB", "iobb")):
                nd = float(counts[f"{key}_num_detected"])
                missed = float(counts[f"{key}_num_missed"])
                is_detected = float(counts[f"{key}_is_detected"])
                denom = nd + missed
                metrics[f"{family}_NumDetected"] = nd
                metrics[f"{family}_NumMissed"] = missed
                metrics[f"{family}_IsDetected"] = is_detected
                metrics[f"{family}_Accuracy"] = is_detected / n_inst
                metrics[f"{family}_AverageTruePositives"] = nd / n_inst
                metrics[f"{family}_AverageFalsePositives"] = missed / n_inst
                metrics[f"{family}_TruePositiveRate"] = nd / denom if denom > 0 else 0.0
                metrics[f"{family}_FalsePositiveRate"] = missed / denom if denom > 0 else 0.0
            best_detection[int(ab)][float(score_thr)] = {
                "HeatmapThreshold": rep,
                "AverageBestHeatmapThreshold": rep,
                "Metrics": metrics,
            }
    return best_detection


def _evaluate_model_localization_onebox(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    localization_method: str,
    spec: LocalizationMethodSpec,
    calibration: LocalizationCalibration | None,
    fit_workers: int | None,
    score_thresholds: list[float],
    return_per_instance: bool,
    only_correctly_classified: bool,
    use_amp: bool,
    localization_kwargs: dict,
) -> tuple:
    """OneBox localization evaluation (val fit or frozen-params scoring).

    Streams the heatmaps once.  On val (``calibration is None``) every
    instance is submitted to a fork pool; the per-finding partials
    are merged deterministically and the val5-IoU ``JointArgmax`` selection
    produces one :class:`LocalizationCalibration` used to build **both** the
    IoU and IoBB ``best_detection`` tables.  On test (``calibration`` given)
    no fit pool is created: each instance is scored inline at the frozen
    params with :func:`localization.fitting.score_fixed_params`.
    """
    spec = spec.resolved()
    get_localization_maps = _resolve_localization_map_extractor(localization_method)
    pbar = tqdm(dataloader, desc="Computing CAM maps", unit="batch")

    is_fit = calibration is None
    if not is_fit and not isinstance(calibration, LocalizationCalibration):
        raise TypeError(
            "localization_calibration must be a LocalizationCalibration "
            f"for OneBox evaluation; got {type(calibration).__name__}"
        )
    if not is_fit and calibration.bbox_method != "onebox":
        raise ValueError(
            "localization_calibration.bbox_method="
            f"{calibration.bbox_method!r} does not match bbox_method='onebox'"
        )
    params_by_ab = None if is_fit else calibration.params

    all_abnorm_idxs: set[int] = set()
    all_detected_abnorm_idxs: set[int] = set()
    proxy_abnorm = {"MIR": defaultdict(list), "AEuD": defaultdict(list)}
    futures_meta: list[dict] = []
    per_instance_scores: list[dict | None] = []
    skipped_instances: list[dict] = []
    image_level: dict[str, dict] = {}

    partials: dict[int, list[np.ndarray]] = {}
    n_inst: dict[int, int] = {}
    fixed_counts: dict[int, dict] = {}
    image_size: int | None = None
    fit_seconds = 0.0

    with ExitStack() as stack:
        pool: OneBoxFitPool | None = None
        for batch in pbar:
            images, labels, filenames, abnorm_bboxes, segmentations = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if image_size is None:
                image_size = int(min(images.shape[-2], images.shape[-1]))
            if is_fit and pool is None:
                pool = stack.enter_context(
                    OneBoxFitPool(
                        image_size=image_size,
                        workers=fit_workers,
                    )
                )
                fit_start = time.perf_counter()

            map_kwargs = dict(localization_kwargs)
            if localization_method.casefold() != "cam":
                map_kwargs["use_amp"] = use_amp
            cam_dict = get_localization_maps(
                model=model,
                images=images,
                labels=labels,
                filenames=filenames,
                **map_kwargs,
            )

            for filename, abnorm_bbox, segmentation in zip(filenames, abnorm_bboxes, segmentations):
                cam_maps = cam_dict[filename]
                img_labels = cam_maps["labels"]
                img_preds = cam_maps["preds"]

                if return_per_instance and filename not in image_level:
                    label_row = img_labels
                    pred = img_preds
                    if th.is_tensor(label_row):
                        label_row = label_row.cpu().numpy()
                    if th.is_tensor(pred):
                        pred = pred.cpu().numpy()
                    image_level[filename] = {
                        "labels": np.asarray(label_row),
                        "preds": np.asarray(pred),
                    }

                for abnorm_idx in abnorm_bbox.keys():
                    all_abnorm_idxs.add(abnorm_idx)
                    correct_cls = bool(img_labels[abnorm_idx] == img_preds[abnorm_idx])
                    if only_correctly_classified and not correct_cls:
                        if return_per_instance:
                            skipped_instances.append(
                                {
                                    "filename": filename,
                                    "abnorm_idx": int(abnorm_idx),
                                    "correct_cls": False,
                                    "MIR": None,
                                    "AEuD": None,
                                    "_image_labels": np.asarray(img_labels, dtype=int),
                                    "_image_preds": np.asarray(img_preds, dtype=int),
                                    "score": None,
                                }
                            )
                        continue

                    # a test finding absent from the frozen calibration has no
                    # params; mirror the legacy threshold-dict filtering and
                    # surface it as undetected instead of scoring it
                    if not is_fit and abnorm_idx not in params_by_ab:
                        continue

                    all_detected_abnorm_idxs.add(abnorm_idx)
                    heatmap = cam_maps[abnorm_idx]
                    bbox = abnorm_bbox[abnorm_idx]
                    mask = segmentation[abnorm_idx].numpy()

                    aeud_abnorm = calculate_average_euclidean_distance(heatmap, mask)
                    proxy_abnorm["AEuD"][abnorm_idx].append(aeud_abnorm)
                    mir_abnorm = calculate_masked_intensity_ratio(heatmap, mask)
                    proxy_abnorm["MIR"][abnorm_idx].append(mir_abnorm)

                    n_inst[abnorm_idx] = n_inst.get(abnorm_idx, 0) + 1
                    score_map = None
                    if is_fit:
                        pool.submit(
                            abnorm_idx,
                            [Instance(filename, int(abnorm_idx), heatmap, bbox)],
                        )
                    else:
                        params = params_by_ab[abnorm_idx]
                        scored = score_fixed_params(
                            heatmap, bbox, params, score_thresholds,
                            # AP-style metrics need the ranked detection list;
                            # only pay for it when per-instance records are kept
                            return_boxes=return_per_instance,
                        )
                        _merge_fixed_counts(fixed_counts, abnorm_idx, scored)
                        score_map = scored["scores"]

                    if return_per_instance:
                        futures_meta.append(
                            {
                                "filename": filename,
                                "abnorm_idx": int(abnorm_idx),
                                "correct_cls": correct_cls,
                                "MIR": float(mir_abnorm),
                                "AEuD": float(aeud_abnorm),
                                "_image_labels": np.asarray(img_labels, dtype=int),
                                "_image_preds": np.asarray(img_preds, dtype=int),
                                # ranked detections must travel WITH the
                                # instance; reading them from loop-local
                                # variables after the loop repeats the last one
                                "boxes": (
                                    None if score_map is None
                                    else np.asarray(scored["boxes"], dtype=np.float64)
                                ),
                                "box_scores": (
                                    None if score_map is None
                                    else np.asarray(scored["box_scores"], dtype=np.float64)
                                ),
                            }
                        )
                        per_instance_scores.append(score_map)

            if is_fit:
                try:
                    for ab, partial in pool.drain():
                        _merge_partial_sums(partials, ab, partial)
                except DetectionWorkerError:
                    raise
                except BaseException as exc:
                    raise DetectionWorkerError(str(exc)) from exc

        if is_fit:
            if pool is not None:
                try:
                    for ab, partial in pool.flush():
                        _merge_partial_sums(partials, ab, partial)
                except DetectionWorkerError:
                    raise
                except BaseException as exc:
                    raise DetectionWorkerError(str(exc)) from exc
                fit_seconds = time.perf_counter() - fit_start
            if image_size is None or not partials:
                raise RuntimeError("cannot fit OneBox on an empty validation split")

    if is_fit:
        findings = sorted(partials)
        fitter = OneBoxFitter(
            spec,
            image_size,
            findings,
            {ab: n_inst[ab] for ab in findings},
            search_space=pool.search_space,
            configs=pool.configs,
        )
        for ab in findings:
            fitter.add_partial(ab, partials[ab])
        stats_iou, stats_iobb = fitter.finalize()
        params_by_ab = select_onebox(stats_iou, stats_iobb, spec)
        calibration = build_calibration(
            spec,
            params_by_ab,
            image_size=image_size,
            search_space=pool.search_space,
            val5=val5_macros(stats_iou, params_by_ab),
            fit_seconds=fit_seconds,
            fit_workers=(fit_workers or 0),
        )
        fixed_counts = _valstats_to_fixed_counts(
            stats_iou, stats_iobb, params_by_ab, n_inst, score_thresholds
        )

    best_detection = _build_fixed_params_best_detection(
        fixed_counts, params_by_ab, score_thresholds
    )

    mean_proxy_abnorm = defaultdict(dict)
    for proxy, d_abnorm in proxy_abnorm.items():
        for abnorm_idx, proxy_vals in d_abnorm.items():
            mean_proxy_abnorm[proxy][abnorm_idx] = np.mean(proxy_vals)

    all_undetected_abnorm_idxs = sorted(all_abnorm_idxs - all_detected_abnorm_idxs)

    instances: list[dict] | None = None
    if return_per_instance:
        instances = []
        for meta, score_map in zip(futures_meta, per_instance_scores):
            abnorm_idx = meta["abnorm_idx"]
            instances.append(
                {
                    "filename": meta["filename"],
                    "abnorm_idx": abnorm_idx,
                    "correct_cls": meta["correct_cls"],
                    "MIR": meta["MIR"],
                    "AEuD": meta["AEuD"],
                    "_image_labels": meta["_image_labels"],
                    "_image_preds": meta["_image_preds"],
                    "heatmap_thr_used": None,
                    "localization_params_used": _params_to_dict(params_by_ab[abnorm_idx]),
                    "score": score_map,
                    # ranked detections (boxes + pseudo-confidence) so an
                    # AP/mAP pass can run post hoc without re-extracting maps
                    "boxes": meta["boxes"],
                    "box_scores": meta["box_scores"],
                }
            )
        empty_score = {
            float(s): {
                "iou_is_detected": 0,
                "iou_num_detected": 0,
                "iou_num_missed": 0,
                "iobb_is_detected": 0,
                "iobb_num_detected": 0,
                "iobb_num_missed": 0,
            }
            for s in score_thresholds
        }
        for skip in skipped_instances:
            skip = dict(skip)
            skip["heatmap_thr_used"] = None
            skip["localization_params_used"] = None
            skip["score"] = empty_score
            instances.append(skip)
        _ = image_level

    base = (
        best_detection,
        mean_proxy_abnorm,
        all_undetected_abnorm_idxs,
        calibration,
    )
    if return_per_instance:
        return base + (instances,)
    return base


def evaluate_model_localization_both(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    localization_method: str = "cam",
    max_workers: int | None = None,
    localization_calibration: LocalizationCalibration | None = None,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    optimize_metrics_separately: bool = False,
    fit_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    iou_thr_merge: float = 1.0,
    proximity_thr: float | None = None,
    return_per_instance: bool = False,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    return_timings: bool = False,
    **localization_kwargs,
) -> tuple:
    """Default evaluation: one val5-IoU calibration, both metric families.

    Streams the split once, fits (val) or loads (test) one
    :class:`LocalizationCalibration` optimized on val5 IoU, and returns the
    historical per-family 4-tuple for both ``"IoBB"`` and ``"IoU"`` from the
    same frozen parameters::

        ({metric: (best_detection, mean_proxy_abnorm, undetected, cal)},
         calibration)

    ``return_per_instance`` appends the shared per-instance records;
    ``return_timings`` appends a phase-timing dict.  The legacy
    ``optimize_metrics_separately=True`` mode (only
    ``singlecut`` + ``mean_argmax``) restores the two-call, per-metric
    sequence and returns the two family-specific artifacts.
    """
    localization_kwargs = dict(localization_kwargs)
    score_thresholds = list(score_thresholds)
    spec = resolve_method_spec(
        bbox_method,
        selection,
        optimize_metrics_separately=optimize_metrics_separately,
    )

    timings: dict[str, float] = {}
    instances = None

    if optimize_metrics_separately:
        start = time.perf_counter()
        iobb_result = evaluate_model_localization(
            model=model,
            dataloader=dataloader,
            device=device,
            localization_method=localization_method,
            max_workers=max_workers,
            bbox_method=spec.bbox_method,
            selection=spec.selection,
            metric="IoBB",
            optimize_metrics_separately=True,
            score_thresholds=score_thresholds,
            iou_thr_merge=iou_thr_merge,
            proximity_thr=proximity_thr,
            return_per_instance=False,
            only_correctly_classified=only_correctly_classified,
            use_amp=use_amp,
            **localization_kwargs,
        )
        iou_result = evaluate_model_localization(
            model=model,
            dataloader=dataloader,
            device=device,
            localization_method=localization_method,
            max_workers=max_workers,
            bbox_method=spec.bbox_method,
            selection=spec.selection,
            metric="IoU",
            optimize_metrics_separately=True,
            score_thresholds=score_thresholds,
            iou_thr_merge=iou_thr_merge,
            proximity_thr=proximity_thr,
            return_per_instance=False,
            only_correctly_classified=only_correctly_classified,
            use_amp=use_amp,
            **localization_kwargs,
        )
        iou_best, iou_proxy, iou_undet, iou_thr = iou_result
        iobb_best, iobb_proxy, iobb_undet, iobb_thr = iobb_result
        # the legacy dict path does not expose the map short side; the
        # per-family threshold dicts remain authoritative for reproduction.
        from clarity.localization.legacy import _singlecut_calibration

        cal = _singlecut_calibration(
            spec=spec,
            heatmap_thr=iou_thr,
            best_detection=iou_best,
            image_size=0,
            score_thresholds=score_thresholds,
            params_iobb=iobb_thr,
            fit_workers=(fit_workers or 0),
        )
        timings["total"] = time.perf_counter() - start
        family_results = {
            "IoBB": (iobb_best, iobb_proxy, iobb_undet, iobb_thr),
            "IoU": (iou_best, iou_proxy, iou_undet, iou_thr),
        }
        result = (family_results, cal)
        if return_timings:
            result = result + (timings,)
        return result

    start = time.perf_counter()
    result_tuple = evaluate_model_localization(
        model=model,
        dataloader=dataloader,
        device=device,
        localization_method=localization_method,
        max_workers=max_workers,
        localization_calibration=localization_calibration,
        bbox_method=spec.bbox_method,
        selection=spec.selection,
        metric="IoU",
        optimize_metrics_separately=False,
        fit_workers=fit_workers,
        score_thresholds=score_thresholds,
        iou_thr_merge=iou_thr_merge,
        proximity_thr=proximity_thr,
        return_per_instance=return_per_instance,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        **localization_kwargs,
    )
    if return_per_instance:
        best, proxy, undet, calibration, instances = result_tuple
    else:
        best, proxy, undet, calibration = result_tuple
    if not isinstance(calibration, LocalizationCalibration):
        raise RuntimeError(
            "the combined evaluator requires a LocalizationCalibration "
            f"artifact; got {type(calibration).__name__}"
        )
    family = (best, proxy, undet, calibration)
    family_results = {"IoBB": family, "IoU": family}
    timings["total"] = time.perf_counter() - start
    out = (family_results, calibration)
    if return_per_instance:
        out = out + (instances,)
    if return_timings:
        out = out + (timings,)
    return out


def _evaluate_model_localization_sweep_onebox(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    attention_layer_counts: tuple[int, ...],
    *,
    spec: LocalizationMethodSpec,
    fit_workers: int | None,
    score_thresholds: list[float],
    only_correctly_classified: bool,
    use_amp: bool,
    localization_kwargs: dict,
) -> tuple[dict[str, dict[str, tuple]], dict[str, float]]:
    """OneBox multi-variant sweep: one forward, one shared fork pool.

    All variants are extracted from a single multi-variant forward per batch
    and streamed into one :class:`~localization.fitting.OneBoxFitPool`
    tagged by variant.  When the pass ends each variant gets its own exact
    0.05-grid ``JointArgmax`` fit, both metric families are built from the
    same selected params, and the 4th tuple element is the per-variant
    shared :class:`LocalizationCalibration`.
    """
    spec = spec.resolved()
    pbar = tqdm(dataloader, desc="Computing sweep CAM maps", unit="batch")
    variant_keys = [variant_key(count) for count in attention_layer_counts]

    all_abnorm_idxs: dict[str, set] = {vk: set() for vk in variant_keys}
    all_detected_abnorm_idxs: dict[str, set] = {vk: set() for vk in variant_keys}
    proxy_abnorm: dict[str, dict[str, defaultdict]] = {
        vk: {"MIR": defaultdict(list), "AEuD": defaultdict(list)} for vk in variant_keys
    }

    partials: dict[str, dict[int, list[np.ndarray]]] = {vk: {} for vk in variant_keys}
    n_inst: dict[str, dict[int, int]] = {vk: {} for vk in variant_keys}
    image_size: int | None = None

    timings: dict[str, float] = {
        "map_extraction": 0.0,
        "onebox_fit": 0.0,
        "selection": 0.0,
        "total": 0.0,
    }
    timings["n_batches"] = 0
    timings["n_model_forwards"] = 0
    total_start = time.perf_counter()

    with ExitStack() as stack:
        pool: OneBoxFitPool | None = None
        fit_start = 0.0
        extraction_start = time.perf_counter()
        for batch in pbar:
            images, labels, filenames, abnorm_bboxes, segmentations = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if image_size is None:
                image_size = int(min(images.shape[-2], images.shape[-1]))
            if pool is None:
                pool = stack.enter_context(
                    OneBoxFitPool(
                        image_size=image_size,
                        workers=fit_workers,
                    )
                )
                fit_start = time.perf_counter()

            timings["n_batches"] += 1
            timings["n_model_forwards"] += 1
            sweep_dict = get_model_attention_sweep(
                model=model,
                images=images,
                labels=labels,
                filenames=filenames,
                attention_layer_counts=attention_layer_counts,
                use_amp=use_amp,
                **localization_kwargs,
            )

            for count in attention_layer_counts:
                vk = variant_key(count)
                cam_dict = sweep_dict[vk]
                for filename, abnorm_bbox, segmentation in zip(
                    filenames, abnorm_bboxes, segmentations
                ):
                    cam_maps = cam_dict[filename]
                    img_labels = cam_maps["labels"]
                    img_preds = cam_maps["preds"]
                    for abnorm_idx in abnorm_bbox.keys():
                        all_abnorm_idxs[vk].add(abnorm_idx)
                        correct_cls = bool(img_labels[abnorm_idx] == img_preds[abnorm_idx])
                        if only_correctly_classified and not correct_cls:
                            continue
                        all_detected_abnorm_idxs[vk].add(abnorm_idx)
                        heatmap = cam_maps[abnorm_idx]
                        bbox = abnorm_bbox[abnorm_idx]
                        mask = segmentation[abnorm_idx].numpy()
                        proxy_abnorm[vk]["AEuD"][abnorm_idx].append(
                            calculate_average_euclidean_distance(heatmap, mask)
                        )
                        proxy_abnorm[vk]["MIR"][abnorm_idx].append(
                            calculate_masked_intensity_ratio(heatmap, mask)
                        )
                        pool.submit(
                            abnorm_idx,
                            [Instance(filename, int(abnorm_idx), heatmap, bbox)],
                            tag=vk,
                        )
                        n_inst[vk][abnorm_idx] = n_inst[vk].get(abnorm_idx, 0) + 1

            try:
                for tag, ab, partial in pool.drain_tagged():
                    _merge_partial_sums(partials[tag], ab, partial)
            except DetectionWorkerError:
                raise
            except BaseException as exc:
                raise DetectionWorkerError(str(exc)) from exc

        timings["map_extraction"] = time.perf_counter() - extraction_start
        if pool is not None:
            try:
                for tag, ab, partial in pool.flush_tagged():
                    _merge_partial_sums(partials[tag], ab, partial)
            except DetectionWorkerError:
                raise
            except BaseException as exc:
                raise DetectionWorkerError(str(exc)) from exc
            timings["onebox_fit"] = time.perf_counter() - fit_start

    if image_size is None:
        raise RuntimeError("cannot fit OneBox on an empty sweep validation split")

    sweep_results: dict[str, dict[str, tuple]] = {}
    for count in attention_layer_counts:
        vk = variant_key(count)
        selection_start = time.perf_counter()
        findings = sorted(partials[vk])
        if not findings:
            raise RuntimeError(
                f"cannot fit OneBox for sweep variant {vk!r} on an empty " "validation split"
            )
        fitter = OneBoxFitter(
            spec,
            image_size,
            findings,
            {ab: n_inst[vk][ab] for ab in findings},
            search_space=pool.search_space,
            configs=pool.configs,
        )
        for ab in findings:
            fitter.add_partial(ab, partials[vk][ab])
        stats_iou, stats_iobb = fitter.finalize()
        params = select_onebox(stats_iou, stats_iobb, spec)
        calibration = build_calibration(
            spec,
            params,
            image_size=image_size,
            search_space=pool.search_space,
            val5=val5_macros(stats_iou, params),
            fit_seconds=timings["onebox_fit"],
            fit_workers=(fit_workers or 0),
        )
        counts = _valstats_to_fixed_counts(
            stats_iou, stats_iobb, params, n_inst[vk], score_thresholds
        )
        best_detection = _build_fixed_params_best_detection(counts, params, score_thresholds)

        mean_proxy_abnorm = defaultdict(dict)
        for proxy, d_abnorm in proxy_abnorm[vk].items():
            for abnorm_idx, proxy_vals in d_abnorm.items():
                mean_proxy_abnorm[proxy][abnorm_idx] = np.mean(proxy_vals)
        all_undetected = sorted(all_abnorm_idxs[vk] - all_detected_abnorm_idxs[vk])

        family = (best_detection, mean_proxy_abnorm, all_undetected, calibration)
        sweep_results[vk] = {"IoBB": family, "IoU": family}
        timings["selection"] += time.perf_counter() - selection_start

    timings["total"] = time.perf_counter() - total_start
    return sweep_results, timings


def evaluate_model_localization(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    localization_method: str = "cam",
    max_workers: int | None = None,
    heatmap_thr: dict[int, float] | None = None,
    localization_calibration: LocalizationCalibration | None = None,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    metric: Literal["IoBB", "IoU"] = "IoBB",
    optimize_metrics_separately: bool = False,
    fit_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    iou_thr_merge: float = 1.0,
    proximity_thr: float | None = None,
    return_per_instance: bool = False,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    **localization_kwargs,
) -> tuple:
    # Reserved controls are handled explicitly by this function. Remove any
    # duplicate copies before forwarding method-specific kwargs to the map
    # extractor, especially when callers build kwargs from a CLI dictionary.
    localization_kwargs = dict(localization_kwargs)
    localization_kwargs.pop("use_amp", None)
    localization_kwargs.pop("only_correctly_classified", None)

    score_thresholds = list(score_thresholds)
    spec = resolve_method_spec(
        bbox_method,
        selection,
        optimize_metrics_separately=optimize_metrics_separately,
    )

    # The 4th tuple slot is polymorphic: callers that round-trip the val
    # artifact through the historical ``heatmap_thr`` keyword are accepted and
    # routed to the method artifact (keeps older callers working).
    if isinstance(heatmap_thr, LocalizationCalibration):
        if localization_calibration is not None and localization_calibration is not heatmap_thr:
            raise ValueError(
                "both heatmap_thr and localization_calibration were given as "
                "calibrations; pass only one"
            )
        localization_calibration = heatmap_thr
        heatmap_thr = None

    # OneBox shares the legacy kernel's supported-options contract.
    if spec.bbox_method == "onebox" and (iou_thr_merge < 1.0 or proximity_thr is not None):
        raise ValueError(
            "OneBox evaluation only supports iou_thr_merge >= 1.0 and "
            "proximity_thr=None (the historical defaults); bbox merging is "
            f"not supported (got iou_thr_merge={iou_thr_merge}, "
            f"proximity_thr={proximity_thr})"
        )

    if spec.bbox_method == "onebox":
        if heatmap_thr is not None and localization_calibration is None:
            raise ValueError(
                "heatmap_thr is a legacy SingleCut artifact; OneBox uses "
                "localization_calibration (pass bbox_method='singlecut' for "
                "the legacy path)"
            )
        return _evaluate_model_localization_onebox(
            model=model,
            dataloader=dataloader,
            device=device,
            localization_method=localization_method,
            spec=spec,
            calibration=localization_calibration,
            fit_workers=_resolve_fit_workers(fit_workers, max_workers),
            score_thresholds=score_thresholds,
            return_per_instance=return_per_instance,
            only_correctly_classified=only_correctly_classified,
            use_amp=use_amp,
            localization_kwargs=localization_kwargs,
        )

    from clarity.localization.legacy import evaluate_model_localization_singlecut

    return evaluate_model_localization_singlecut(
        model=model,
        dataloader=dataloader,
        device=device,
        localization_method=localization_method,
        max_workers=max_workers,
        heatmap_thr=heatmap_thr,
        localization_calibration=localization_calibration,
        bbox_method=bbox_method,
        selection=selection,
        metric=metric,
        optimize_metrics_separately=optimize_metrics_separately,
        fit_workers=fit_workers,
        score_thresholds=score_thresholds,
        iou_thr_merge=iou_thr_merge,
        proximity_thr=proximity_thr,
        return_per_instance=return_per_instance,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        **localization_kwargs,
    )


def evaluate_model_localization_sweep(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    attention_layer_counts: tuple[int, ...],
    localization_method: str = "model_attention",
    max_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    iou_thr_merge: float = 1.0,
    proximity_thr: float | None = None,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    fit_workers: int | None = None,
    **localization_kwargs,
) -> tuple[dict[str, dict[str, tuple]], dict[str, float]]:
    """Exact multi-variant localization evaluation over all layer counts.

    Receives all CLARiTy attention variants from one multi-variant forward per
    batch (:func:`get_model_attention_sweep`).  The **default method is OneBox
    + joint_argmax**: each variant gets its own exact 0.05-grid fit
    in one shared fork process pool and both families are reported from the
    same selected params; the 4th tuple element is then the per-variant
    :class:`~localization.fitting.LocalizationCalibration`.

    The legacy ``singlecut`` path runs raw localization detection
    (the per-instance detection tables) once per variant map in one
    ``ThreadPoolExecutor`` and derives both IoBB- and IoU-optimized outputs
    from the same raw tables; model inference and raw detection are therefore
    never repeated per metric. Exact full-grid semantics are fixed: the
    unchanged 91 ``THRESHOLDS_HEATMAP`` values and every supplied score
    threshold are always used.

    Args:
        model: Sweep-capable model (e.g. ``CLARiTy``), optionally wrapped in
            ``nn.DataParallel``; see :func:`get_model_attention_sweep`.
        dataloader: Localization validation loader yielding batches of
            ``(images, labels, filenames, abnorm_bboxes, segmentations)``.
        device: Torch device string for model inference.
        attention_layer_counts: Tuple of suffix-layer counts to evaluate.
        localization_method: Only ``"model_attention"`` is supported; any
            other method fails loudly rather than degrading silently.
        max_workers: Worker count for the legacy detection thread pool
            (``None`` uses the executor default).
        bbox_method: ``"onebox"`` (default) or ``"singlecut"`` (legacy).
        selection: ``None`` (method default), ``"joint_argmax"`` or
            ``"mean_argmax"``.
        fit_workers: Fork-pool size for the OneBox per-variant fits (``None``
            uses all CPUs).
        score_thresholds: Score thresholds dictating successful detection.
            Defaults to the standard ``THRESHOLDS_SCORE`` grid.
        iou_thr_merge: Optional IoU merging of predicted bounding boxes.
        proximity_thr: Optional proximity merging of predicted boxes.
        only_correctly_classified: Keep the historical behaviour of evaluating
            only correctly classified abnormalities (default evaluates all
            boxed abnormalities, matching the ordinary evaluator).
        use_amp: Optional bfloat16 autocast around model forwards.
        **localization_kwargs: Forwarded to the map extractor (e.g.
            ``inference_attention_type``, ``interpolation_mode``).

    Returns:
        ``(sweep_results, timings)`` where ``sweep_results`` is keyed by
        canonical variant key (``n_attn_layers_avg=2``) and then by metric
        family (``"IoBB"``/``"IoU"``), holding the historical 4-tuple
        ``(metrics, best_detection, mean_proxy_abnorm, artifact)`` per family,
        where ``artifact`` is the legacy ``heatmap_thr`` dict for SingleCut or
        the shared ``LocalizationCalibration`` for OneBox (abnormalities that
        were never evaluated hold ``None``)::

            {
              "n_attn_layers_avg=2": {
                "IoBB": (metrics, best_detection, mean_proxy_abnorm, heatmap_thr),
                "IoU":  (metrics, best_detection, mean_proxy_abnorm, heatmap_thr),
              },
              ...
            }

        and ``timings`` is a plain dict of phase durations in seconds with
        keys ``"map_extraction"`` (GPU/inference phase), ``"cpu_detection"``
        (raw detection phase), ``"metric_reduction"`` and ``"total"``. The
        timing metadata is separate so the result tuples remain byte-for-byte
        comparable with the ordinary evaluator.

    Raises:
        ValueError: Unsupported localization method or a failing sweep
            capability/shape check of the extractor.
        DetectionWorkerError: A detection worker raised; the original
            exception is chained and the message carries the variant key,
            filename, and abnormality index of the failing job. No partial
            results are returned.
    """
    localization_method = localization_method.casefold()
    if localization_method != "model_attention":
        raise ValueError(
            "the multi-variant localization sweep currently supports only "
            f"localization_method='model_attention', got {localization_method!r}"
        )

    score_thresholds = list(score_thresholds)
    localization_kwargs = dict(localization_kwargs)
    # reserved controls are handled explicitly by this function
    localization_kwargs.pop("use_amp", None)
    localization_kwargs.pop("only_correctly_classified", None)

    spec = resolve_method_spec(bbox_method, selection)
    if spec.bbox_method == "onebox":
        return _evaluate_model_localization_sweep_onebox(
            model=model,
            dataloader=dataloader,
            device=device,
            attention_layer_counts=attention_layer_counts,
            spec=spec,
            fit_workers=_resolve_fit_workers(fit_workers, max_workers),
            score_thresholds=score_thresholds,
            only_correctly_classified=only_correctly_classified,
            use_amp=use_amp,
            localization_kwargs=localization_kwargs,
        )

    from clarity.localization.legacy import evaluate_model_localization_sweep_singlecut

    return evaluate_model_localization_sweep_singlecut(
        model=model,
        dataloader=dataloader,
        device=device,
        attention_layer_counts=attention_layer_counts,
        localization_method=localization_method,
        max_workers=max_workers,
        score_thresholds=score_thresholds,
        iou_thr_merge=iou_thr_merge,
        proximity_thr=proximity_thr,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        bbox_method=bbox_method,
        selection=selection,
        fit_workers=fit_workers,
        **localization_kwargs,
    )


def _localization_macro_metrics(
    best_detection: dict,
    mean_proxy_abnorm: dict,
    score_thresholds: Iterable[float],
) -> dict:
    """Macro-average localization metrics over abnorms × score thresholds."""
    metrics = defaultdict(float)
    n_localized_abnorms = len(best_detection)
    n_score_thresholds = len(list(score_thresholds))
    n_atomic_metrics = n_score_thresholds * n_localized_abnorms
    if n_atomic_metrics == 0:
        return metrics

    for _abnorm_idx, abnorm_metrics in best_detection.items():
        for _score_thr, abnorm_score_results in abnorm_metrics.items():
            abnormality_score_metrics = abnorm_score_results["Metrics"]

            iou_accuracy = abnormality_score_metrics["IoU_Accuracy"]
            iou_atp = abnormality_score_metrics["IoU_AverageTruePositives"]
            iou_afp = abnormality_score_metrics["IoU_AverageFalsePositives"]
            iou_tpr = abnormality_score_metrics["IoU_TruePositiveRate"]
            iou_fpr = abnormality_score_metrics["IoU_FalsePositiveRate"]

            iobb_accuracy = abnormality_score_metrics["IoBB_Accuracy"]
            iobb_atp = abnormality_score_metrics["IoBB_AverageTruePositives"]
            iobb_afp = abnormality_score_metrics["IoBB_AverageFalsePositives"]
            iobb_tpr = abnormality_score_metrics["IoBB_TruePositiveRate"]
            iobb_fpr = abnormality_score_metrics["IoBB_FalsePositiveRate"]

            metrics["IoU_accuracy_macro"] += iou_accuracy / n_atomic_metrics
            metrics["IoU_atp_macro"] += iou_atp / n_atomic_metrics
            metrics["IoU_afp_macro"] += iou_afp / n_atomic_metrics
            metrics["IoU_tpr_macro"] += iou_tpr / n_atomic_metrics
            metrics["IoU_fpr_macro"] += iou_fpr / n_atomic_metrics

            metrics["IoBB_accuracy_macro"] += iobb_accuracy / n_atomic_metrics
            metrics["IoBB_atp_macro"] += iobb_atp / n_atomic_metrics
            metrics["IoBB_afp_macro"] += iobb_afp / n_atomic_metrics
            metrics["IoBB_tpr_macro"] += iobb_tpr / n_atomic_metrics
            metrics["IoBB_fpr_macro"] += iobb_fpr / n_atomic_metrics

    if n_localized_abnorms > 0:
        for proxy, d_abnorm in mean_proxy_abnorm.items():
            for mean_val in d_abnorm.values():
                metrics[f"{proxy}_macro"] += mean_val / n_localized_abnorms

    return metrics


def run_localization_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_workers: int,
    heatmap_thr: dict[int, float] | None = None,
    localization_calibration=None,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    metric: Literal["IoBB", "IoU"] = "IoBB",
    optimize_metrics_separately: bool = False,
    fit_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    return_per_instance: bool = False,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    **localization_kwargs,
):
    """Evaluate localization and return the historical 4-tuple by default.

    Localization is evaluated over all bbox-annotated abnormalities by default,
    including incorrectly classified abnormalities. Set
    ``only_correctly_classified=True`` to use the historical filtered behavior.

    The default method is OneBox + ``joint_argmax``; pass
    ``bbox_method="singlecut", selection="mean_argmax"`` for the legacy path
    and ``optimize_metrics_separately=True`` for the historical per-metric
    thresholds.  The 4th tuple element is the method artifact
    (``dict[int, float]`` legacy thresholds or a ``LocalizationCalibration``).

    When ``return_per_instance=True``, also returns a fifth element: a list of
    per-(image, pathology) instance records (see
    ``localization.evaluation.run_localization_evaluation``).
    """
    score_thresholds = list(score_thresholds)
    # Reserved evaluation controls are passed explicitly below. Remove copies
    # supplied through the free-form localization kwargs before forwarding to
    # avoid duplicate-keyword errors.
    localization_kwargs = dict(localization_kwargs)
    localization_method = localization_kwargs.pop("localization_method", "cam")
    only_correctly_classified = bool(
        localization_kwargs.pop("only_correctly_classified", only_correctly_classified)
    )
    localization_kwargs.pop("use_amp", None)

    output = evaluate_model_localization(
        model=model,
        dataloader=dataloader,
        device=device,
        max_workers=max_workers,
        heatmap_thr=heatmap_thr,
        localization_calibration=localization_calibration,
        bbox_method=bbox_method,
        selection=selection,
        localization_method=localization_method,
        metric=metric,
        optimize_metrics_separately=optimize_metrics_separately,
        fit_workers=fit_workers,
        score_thresholds=score_thresholds,
        return_per_instance=return_per_instance,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        **localization_kwargs,
    )
    if return_per_instance:
        best_detection, mean_proxy_abnorm, all_undetected_abnorm_idxs, artifact, instances = output
    else:
        best_detection, mean_proxy_abnorm, all_undetected_abnorm_idxs, artifact = output
        instances = None

    metrics = _localization_macro_metrics(best_detection, mean_proxy_abnorm, score_thresholds)

    for undetected_abnorm_idx in all_undetected_abnorm_idxs:
        best_detection[undetected_abnorm_idx] = None
        for proxy in mean_proxy_abnorm.keys():
            mean_proxy_abnorm[proxy][undetected_abnorm_idx] = None

    result = (
        metrics,
        best_detection,
        mean_proxy_abnorm,
        artifact,
    )
    if return_per_instance:
        return result + (instances,)
    return result


def run_localization_evaluation_both(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_workers: int | None = None,
    localization_calibration=None,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    optimize_metrics_separately: bool = False,
    fit_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    return_per_instance: bool = False,
    return_timings: bool = False,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    **localization_kwargs,
):
    """Default localization evaluation: one IoU fit, both families.

    Thin wrapper over :func:`localization.evaluation.run_localization_evaluation_both`
    that converts each raw family tuple to the historical macro-metric schema
    produced by :func:`evaluate_model_localization` (the metric formulas are
    not duplicated).  Returns ``(family_results, calibration)`` where
    ``family_results`` maps ``"IoBB"``/``"IoU"`` to the historical 4-tuple
    ``(metrics, best_detection, mean_proxy_abnorm, calibration)``.  With
    ``return_per_instance`` the shared per-instance records are appended, and
    with ``return_timings`` the phase-timing dict is appended last.
    """
    score_thresholds = list(score_thresholds)
    localization_kwargs = dict(localization_kwargs)
    localization_method = localization_kwargs.pop("localization_method", "cam")
    only_correctly_classified = bool(
        localization_kwargs.pop("only_correctly_classified", only_correctly_classified)
    )
    localization_kwargs.pop("use_amp", None)

    raw = evaluate_model_localization_both(
        model=model,
        dataloader=dataloader,
        device=device,
        localization_method=localization_method,
        max_workers=max_workers,
        localization_calibration=localization_calibration,
        bbox_method=bbox_method,
        selection=selection,
        optimize_metrics_separately=optimize_metrics_separately,
        fit_workers=fit_workers,
        score_thresholds=score_thresholds,
        return_per_instance=return_per_instance,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        return_timings=return_timings,
        **localization_kwargs,
    )

    family_raw, calibration = raw[0], raw[1]
    extra = list(raw[2:])
    family_results: dict[str, tuple] = {}
    # Both families share the same objects on the default OneBox path, so all
    # macro metrics must be computed *before* the undetected ``None`` entries
    # are installed (the macro denominator is the detected-only key count).
    for metric, (
        best_detection,
        mean_proxy_abnorm,
        _all_undetected,
        artifact,
    ) in family_raw.items():
        metrics = _localization_macro_metrics(best_detection, mean_proxy_abnorm, score_thresholds)
        family_results[metric] = (
            metrics,
            best_detection,
            mean_proxy_abnorm,
            artifact,
        )
    for _metric, (
        best_detection,
        mean_proxy_abnorm,
        all_undetected_abnorm_idxs,
        _artifact,
    ) in family_raw.items():
        for undetected_abnorm_idx in all_undetected_abnorm_idxs:
            best_detection[undetected_abnorm_idx] = None
            for proxy in mean_proxy_abnorm.keys():
                mean_proxy_abnorm[proxy][undetected_abnorm_idx] = None

    out = (family_results, calibration)
    return out + tuple(extra)


def run_localization_evaluation_sweep(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    attention_layer_counts: tuple[int, ...],
    max_workers: int | None = None,
    score_thresholds: Iterable[float] = THRESHOLDS_SCORE,
    return_timings: bool = False,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    bbox_method: str | None = "onebox",
    selection: str | None = None,
    fit_workers: int | None = None,
    **localization_kwargs,
) -> dict[str, dict[str, tuple]] | tuple[dict[str, dict[str, tuple]], dict[str, float]]:
    """Evaluate the exact localization sweep and return macro-metric results.

    Thin wrapper over
    :func:`localization.evaluation.run_localization_evaluation_sweep`: converts each
    raw per-variant/per-metric tuple to the same macro metric schema produced
    by :func:`evaluate_model_localization` (via
    ``_localization_macro_metrics``; no metric formulas are duplicated).

    Args:
        model: Sweep-capable model (e.g. ``CLARiTy``), optionally wrapped in
            ``nn.DataParallel``.
        dataloader: Localization validation loader yielding ``(images,
            labels, filenames, abnorm_bboxes, segmentations)`` batches.
        device: Torch device string for model inference.
        attention_layer_counts: Tuple of suffix-layer counts to evaluate.
        max_workers: Worker count for the legacy detection thread pool. The
            OneBox fit uses its own fork process pool (``fit_workers``); the
            OneBox-only path never starts the thread pool.
        score_thresholds: Score thresholds dictating successful detection
            (standard default: five ``THRESHOLDS_SCORE`` values).
        return_timings: When ``True``, also return the timing metadata dict
            (map extraction / CPU detection / metric reduction / total, in
            seconds) instead of hiding it.
        only_correctly_classified: Historical filtered behaviour; default
            evaluates all bbox-annotated abnormalities.
        use_amp: Optional bfloat16 autocast around model forwards.
        **localization_kwargs: Forwarded to the sweep evaluator (e.g.
            ``inference_attention_type``, ``interpolation_mode``).

    Returns:
        By default::

            {
              variant_key: {
                "IoBB": (metrics, class_metrics, proxy_metrics, heatmap_thresholds),
                "IoU":  (metrics, class_metrics, proxy_metrics, heatmap_thresholds),
              },
              ...
            }

        where each 4-tuple matches :func:`evaluate_model_localization`'s
        default output. With ``return_timings=True``, returns
        ``(sweep_results, timings)``.
    """
    score_thresholds = list(score_thresholds)
    # Reserved evaluation controls are passed explicitly below. Remove copies
    # supplied through the free-form localization kwargs before forwarding to
    # avoid duplicate-keyword errors.
    localization_kwargs = dict(localization_kwargs)
    localization_method = localization_kwargs.pop("localization_method", "model_attention")
    only_correctly_classified = bool(
        localization_kwargs.pop("only_correctly_classified", only_correctly_classified)
    )
    localization_kwargs.pop("use_amp", None)

    sweep_raw, timings = evaluate_model_localization_sweep(
        model=model,
        dataloader=dataloader,
        device=device,
        attention_layer_counts=attention_layer_counts,
        localization_method=localization_method,
        max_workers=max_workers,
        score_thresholds=score_thresholds,
        only_correctly_classified=only_correctly_classified,
        use_amp=use_amp,
        bbox_method=bbox_method,
        selection=selection,
        fit_workers=fit_workers,
        **localization_kwargs,
    )

    sweep_results: dict[str, dict[str, tuple]] = {}
    for variant_key, metric_families in sweep_raw.items():
        family_results: dict[str, tuple] = {}
        for metric, (
            best_detection,
            mean_proxy_abnorm,
            all_undetected_abnorm_idxs,
            heatmap_thr,
        ) in metric_families.items():
            metrics = _localization_macro_metrics(
                best_detection, mean_proxy_abnorm, score_thresholds
            )
            family_results[metric] = (
                metrics,
                best_detection,
                mean_proxy_abnorm,
                heatmap_thr,
            )
        sweep_results[variant_key] = family_results

    if return_timings:
        return sweep_results, timings
    return sweep_results
