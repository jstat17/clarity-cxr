"""Evaluation loop: device/compile, evaluator calls and pickle writing."""

from __future__ import annotations

import logging

import pickle as pk
from pathlib import Path

import torch as th
import torch.nn as nn

from clarity.evaluation import classification as evaluate_utils
from clarity.localization import evaluation as localization_eval
from clarity.localization.artifacts import (
    calibration_thresholds,
    localization_calibration_from_block,
    localization_method_block,
)
from clarity.utils.io import convert_defaultdict_to_dict
from clarity.utils.runtime import set_primary_device, wrap_parallel

logger = logging.getLogger(__name__)
LOCALIZATION_TEST_SCORE_THRESHOLDS = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
]


def run_evaluation(
    device: str,
    parallel: bool,
    jit_compile: bool,
    model: nn.Module,
    criterion: nn.Module | None,
    evaluate_loader: list[th.utils.data.DataLoader],
    save_path: Path,
    split: str,
    mp_workers: int,
    iou_thr_merge: float,
    proximity_thr: float | None,
    localization_selection: dict | None = None,
    only_correctly_classified: bool = False,
    use_amp: bool = False,
    bbox_method: str = "onebox",
    selection: str | None = None,
    optimize_metrics_separately: bool = False,
    fit_workers: int | None = None,
    frozen_calibration: dict | None = None,
    **localization_kwargs,
):
    """Evaluate one model on one split and write the evaluation pickle.

    ``frozen_calibration`` is a schema-1 ``localization_method`` block (as
    persisted by :func:`clarity.localization.artifacts.localization_method_block`
    and shipped as ``a1_onebox_nih_vindr6_{224,512}.json``).  When given, the
    validation pass is scored with that frozen artifact instead of being
    refitted, which is arm A1 (**NIH-calibrated**, strict zero-shot) of the
    transfer study: the parameters learned on NIH are ported unchanged to the
    target dataset.  Without it the calibration is refitted on the target
    validation split, which is arm A2 (**Target-calibrated**); see
    :mod:`clarity.evaluation.arms`.
    """
    # Pin the main process's intra-op CPU pool exactly like the training path
    # (training/loop.py::train_and_evaluate): the evaluator's real CPU
    # parallelism lives in the explicitly-sized numba detection thread pool,
    # while an unset pool spans every core and is inherited by the forked
    # th.utils.data.DataLoader/fit-pool workers. Loader workers re-pin themselves in
    # data/loader.py::dataloader_worker_init_fn. Keeping the evaluator's
    # implicit pools at one thread is what lets two whole-node evaluations
    # run concurrently without oversubscribing the node.
    th.set_num_threads(1)

    # ``use_amp`` and ``only_correctly_classified`` are explicit controls in
    # the calls below; remove duplicates supplied through free-form kwargs.
    localization_kwargs = dict(localization_kwargs)
    localization_kwargs.pop("use_amp", None)
    localization_kwargs.pop("only_correctly_classified", None)

    # use DataParallel if parallel is enabled
    # A frozen calibration is scored, never refitted (arm A1, NIH-calibrated).
    frozen = (
        localization_calibration_from_block(frozen_calibration)
        if frozen_calibration is not None
        else None
    )
    if frozen is not None:
        if split != "localize":
            raise SystemExit(
                "a frozen OneBox calibration only applies to '-spl localize'; "
                f"got split {split!r}"
            )
        # the ported artifact carries the method it was fitted with
        bbox_method = frozen.bbox_method
        selection = frozen.selection
        optimize_metrics_separately = frozen.optimize_metrics_separately
        if frozen.image_size and int(frozen.image_size) != int(localization_kwargs.get("size", frozen.image_size)):
            logger.warning(
                "frozen calibration was fitted at image_size=%s; evaluating at a "
                "different resolution invalidates the pixel-scale parameters",
                frozen.image_size,
            )

    model, device = wrap_parallel(model, parallel, device)

    # Set the current CUDA device when applicable. CPU execution must not
    # call torch.cuda.set_device().
    device = set_primary_device(device)

    # send model to device
    model = model.to(device)

    # Match the training path: compile after wrapping the model and moving it
    # to the primary device.
    if jit_compile:
        model.compile()

    match split:
        case "classify":
            # get classification results on validation set
            val_metrics, val_raw = evaluate_utils.evaluate_model_classification(
                model=model,
                loader=evaluate_loader[0],
                split="validate",
                device=device,
                criterion=criterion,
                use_amp=use_amp,
            )
            logger.info("Validation Classification")
            evaluate_utils.print_metrics(val_metrics)

            # get classification results on test set
            metrics, raw = evaluate_utils.evaluate_model_classification(
                model=model,
                loader=evaluate_loader[1],
                split="test",
                device=device,
                criterion=criterion,
                use_amp=use_amp,
            )
            logger.info("Test Classification")
            evaluate_utils.print_metrics(metrics)

            output = {
                "metrics": metrics,
                "raw": raw,
                "val_metrics": val_metrics,
                "val_raw": val_raw,
            }

        case "localize":
            # the localization selection decides which attention-layer count
            # the ordinary model-attention extractor uses
            if localization_selection is not None:
                selected_count = localization_selection["n_attn_layers_avg"]
                source = localization_selection["source"]
                print(
                    f"Localization attention-layer count: {selected_count} "
                    f"(source: {source})"
                )
                if localization_selection.get("override_outside_recorded_sweep"):
                    print(
                        "Note: the explicit count is outside the recorded "
                        "sweep counts of the run's best-selection.json"
                    )
                localization_kwargs.setdefault("attention_layer_count", selected_count)

            # get localization results by optimizing both IoBB and IoU
            output = {}
            if localization_selection is not None:
                source = localization_selection["source"]
                selected_count = localization_selection["n_attn_layers_avg"]
                if source in ("manifest", "explicit_cli"):
                    # Call-level selection: the ordinary model-attention
                    # extractor uses the sweep forward with a
                    # single-element count tuple; the model is never mutated.
                    localization_kwargs.setdefault("attention_layer_count", selected_count)
                elif source == "legacy_model_kwarg":
                    # Pass the count through the historical alias kwarg; it is
                    # honoured call-level for sweep-capable models and ignored
                    # (historical behavior) otherwise.
                    localization_kwargs.setdefault("n_attn_layers_avg", selected_count)

            if optimize_metrics_separately:
                # Legacy per-metric sequence: two independent val fits (IoBB,
                # IoU); the test pass reuses each family's frozen scalar
                # threshold.  Only valid for singlecut + mean_argmax (the
                # evaluator raises otherwise).
                family_results, val_calibration = (
                    localization_eval.run_localization_evaluation_both(
                        model=model,
                        dataloader=evaluate_loader[0],
                        device=device,
                        max_workers=mp_workers,
                        localization_calibration=frozen,
                        bbox_method=bbox_method,
                        selection=selection,
                        optimize_metrics_separately=True,
                        fit_workers=fit_workers,
                        iou_thr_merge=iou_thr_merge,
                        proximity_thr=proximity_thr,
                        only_correctly_classified=only_correctly_classified,
                        use_amp=use_amp,
                        **localization_kwargs,
                    )
                )
                thresholds = {metric: family_results[metric][3] for metric in ("IoBB", "IoU")}
                for metric in ["IoBB", "IoU"]:
                    logger.info(f"--Optimizing Localization by {metric}--")
                    val_metrics = family_results[metric][0]
                    val_raw = family_results[metric][1]
                    val_proxy = family_results[metric][2]
                    logger.info("Validation Localization")
                    evaluate_utils.print_metrics(val_metrics)
                    metrics, raw, proxy, _test_heatmap_thr = (
                        localization_eval.run_localization_evaluation(
                            model=model,
                            dataloader=evaluate_loader[1],
                            device=device,
                            max_workers=mp_workers,
                            heatmap_thr=thresholds[metric],
                            metric=metric,
                            bbox_method=bbox_method,
                            selection=selection,
                            optimize_metrics_separately=True,
                            fit_workers=fit_workers,
                            score_thresholds=LOCALIZATION_TEST_SCORE_THRESHOLDS,
                            iou_thr_merge=iou_thr_merge,
                            proximity_thr=proximity_thr,
                            only_correctly_classified=only_correctly_classified,
                            use_amp=use_amp,
                            **localization_kwargs,
                        )
                    )
                    logger.info("Test Localization")
                    evaluate_utils.print_metrics(metrics)
                    output[metric] = {
                        "metrics": metrics,
                        "raw": raw,
                        "proxy": proxy,
                        "heatmap_thr": calibration_thresholds(thresholds[metric]),
                        "val_metrics": val_metrics,
                        "val_raw": val_raw,
                        "val_proxy": val_proxy,
                    }
                output["localization_method"] = localization_method_block(val_calibration)
            else:
                # Default: one val5-IoU optimization producing both families
                #; the test pass reuses the frozen calibration and never
                # fits.
                family_results, val_calibration = (
                    localization_eval.run_localization_evaluation_both(
                        model=model,
                        dataloader=evaluate_loader[0],
                        device=device,
                        max_workers=mp_workers,
                        localization_calibration=frozen,
                        bbox_method=bbox_method,
                        selection=selection,
                        fit_workers=fit_workers,
                        iou_thr_merge=iou_thr_merge,
                        proximity_thr=proximity_thr,
                        only_correctly_classified=only_correctly_classified,
                        use_amp=use_amp,
                        **localization_kwargs,
                    )
                )
                test_results, _test_calibration = (
                    localization_eval.run_localization_evaluation_both(
                        model=model,
                        dataloader=evaluate_loader[1],
                        device=device,
                        max_workers=mp_workers,
                        localization_calibration=val_calibration,
                        bbox_method=val_calibration.bbox_method,
                        selection=val_calibration.selection,
                        score_thresholds=LOCALIZATION_TEST_SCORE_THRESHOLDS,
                        iou_thr_merge=iou_thr_merge,
                        proximity_thr=proximity_thr,
                        only_correctly_classified=only_correctly_classified,
                        use_amp=use_amp,
                        **localization_kwargs,
                    )
                )
                for metric in ["IoBB", "IoU"]:
                    val_metrics, val_raw, val_proxy, val_artifact = family_results[metric]
                    metrics, raw, proxy, _artifact = test_results[metric]
                    logger.info(f"--{metric} Localization--")
                    logger.info("Validation Localization")
                    evaluate_utils.print_metrics(val_metrics)
                    logger.info("Test Localization")
                    evaluate_utils.print_metrics(metrics)
                    output[metric] = {
                        "metrics": metrics,
                        "raw": raw,
                        "proxy": proxy,
                        "heatmap_thr": (
                            None
                            if getattr(val_artifact, "bbox_method", None) == "onebox"
                            else calibration_thresholds(val_artifact)
                        ),
                        "val_metrics": val_metrics,
                        "val_raw": val_raw,
                        "val_proxy": val_proxy,
                    }
                output["localization_method"] = localization_method_block(val_calibration)

    # save evaluation outputs as pickle
    output = convert_defaultdict_to_dict(output)
    if localization_selection is not None:
        output["localization_selection"] = dict(localization_selection)
    with open(save_path, "wb") as f:
        pk.dump(output, f)
