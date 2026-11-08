"""Training/evaluation epoch loop.

The per-epoch mechanics live in :mod:`clarity.training.epoch`, artifacts in
:mod:`clarity.training.artifacts`, selection logging in
:mod:`clarity.training.selection` and runtime helpers in
:mod:`clarity.training.runtime`.  ``train_and_evaluate`` stays here and the
historical public names are re-exported so existing callers keep importing
from this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from clarity.data.loader import update_dataloader
from clarity.evaluation import classification as evaluate
from clarity.localization import evaluation as localization_eval
from clarity.localization.sweep import (
    LocalizationSweepSpec,
    SweepArtifactManager,
    has_sweep_capability,
)
from clarity.training.artifacts import get_next_run_folder, save_checkpoint, save_performance_log
from clarity.training.epoch import train_one_epoch
from clarity.training.runtime import set_global_seed, unwrap_parallel, wrap_parallel
from clarity.training.selection import (
    _legacy_localization_log_entry,
    _localization_log_fields as _localization_log_fields,
    _print_sweep_table,
    _resolve_localization_controls,
    _sweep_log_entry,
    _sweep_selection_log_entry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_next_run_folder",
    "save_checkpoint",
    "save_performance_log",
    "set_global_seed",
    "train_and_evaluate",
    "train_one_epoch",
    "unwrap_parallel",
    "wrap_parallel",
]


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    evaluation_loader: DataLoader,
    evaluation_split: str,
    optimizer: optim.Optimizer,
    criterion: nn.Module | list[nn.Module],
    device: str,
    num_epochs: int,
    accumulation_factor: int,
    save_dir: Path,
    scheduler: LambdaLR | None = None,
    localization_loader: DataLoader | None = None,
    mp_workers: int | None = None,
    localization_kwargs: dict[str, Any] = dict(),
    early_stop_num_epochs: int | float = float("inf"),
    terminate_epoch: int = float("inf"),
    use_amp: bool = False,
    localization_sweep: LocalizationSweepSpec | None = None,
) -> None:
    # handle multiple loss functions
    if not isinstance(criterion, list):
        criterion = [criterion]

    # Localization metrics use all bbox-annotated images by default. Preserve
    # the opt-in historical filter for callers that explicitly request it.
    # The explicit method controls (bbox_method/selection/fit_workers/
    # optimize_metrics_separately) are routed to the evaluator wrappers.
    localization_kwargs, localization_controls = _resolve_localization_controls(localization_kwargs)
    only_correctly_classified = localization_controls["only_correctly_classified"]

    criterion_primary = criterion[0]
    num_criteria = len(criterion)

    # validate the optional localization sweep before any training happens
    if localization_sweep is not None:
        if localization_loader is None:
            raise ValueError(
                "localization_sweep requires localization_loader to be set: "
                "the sweep evaluates the full exact localization metrics for "
                "every configured attention-layer count after every epoch"
            )
        if not has_sweep_capability(model):
            raise ValueError(
                "localization_sweep requires a sweep-capable model (the "
                "model or its unwrapped module must advertise the "
                "localization sweep capability); the ordinary training path "
                "does not evaluate multiple attention-layer counts"
            )
        # owns best-per-variant/best-overall selection, best-epochs/ snapshots,
        # best-selection.json, and the relative model_best.pth symlink
        artifact_manager = SweepArtifactManager(save_dir, localization_sweep)
        artifact_manager.initialize_run_dir()

    # set up file paths for checkpoints and logs
    last_checkpoint_path = save_dir / "model_checkpoint.pth"
    best_checkpoint_path = save_dir / "model_best.pth"
    log_path = save_dir / "log.pk.xz"

    # initialize tracking variables
    performance_curves = defaultdict(list)
    best_score = float("-inf")
    epochs_since_best_score = 0

    # pin the main process's intra-op CPU pool: this process's real CPU work
    # (loss reduction, AMP bookkeeping) is trivial, while an unset pool spans
    # every core and is inherited by any forked pool members (the sweep's
    # detection threads), which livelocks multi-chain whole-node jobs. Callers should
    # cap OMP/MKL/OpenBLAS in the job environment. Loader workers re-pin themselves in
    # data/loader.py::dataloader_worker_init_fn, so this is never inherited
    # where it would hurt.
    th.set_num_threads(1)

    # training loop
    for epoch in range(num_epochs):
        # set epoch for dataset
        train_loader = update_dataloader(train_loader, epoch)

        # train the model for one epoch
        train_loss = train_one_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion[epoch % num_criteria],
            device=device,
            accumulation_factor=accumulation_factor,
            use_amp=use_amp,
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        ## evaluate model
        # evaluate the model's classification performance
        classif_metrics, classif_metrics_class = evaluate.evaluate_model_classification(
            model=model,
            loader=evaluation_loader,
            split=evaluation_split,
            device=device,
            criterion=criterion_primary,
            use_amp=use_amp,
        )
        evaluate.print_metrics(classif_metrics)

        # evaluate the model's localization performance
        sweep_results: dict[str, dict[str, tuple]] | None = None
        sweep_timings: dict[str, float] | None = None
        if localization_sweep is not None:
            # exact sweep evaluation: one multi-variant forward per batch and
            # one raw-detection pass per variant; no separate ordinary IoBB
            # or IoU model passes are run in sweep mode
            sweep_results, sweep_timings = localization_eval.run_localization_evaluation_sweep(
                model=model,
                dataloader=localization_loader,
                device=device,
                attention_layer_counts=localization_sweep.attention_layer_counts,
                max_workers=mp_workers,
                bbox_method=localization_controls["bbox_method"],
                selection=localization_controls["selection"],
                fit_workers=localization_controls["fit_workers"],
                only_correctly_classified=only_correctly_classified,
                return_timings=True,
                use_amp=use_amp,
                **localization_kwargs,
            )
        elif localization_loader is not None:
            # Default: one val5-IoU optimization per epoch producing both
            # metric families.  ``optimize_metrics_separately=True``
            # keeps the historical two-call, per-metric sequence (SingleCut +
            # mean_argmax only).
            family_results, _val_calibration = localization_eval.run_localization_evaluation_both(
                model=model,
                dataloader=localization_loader,
                device=device,
                max_workers=mp_workers,
                bbox_method=localization_controls["bbox_method"],
                selection=localization_controls["selection"],
                optimize_metrics_separately=localization_controls["optimize_metrics_separately"],
                fit_workers=localization_controls["fit_workers"],
                only_correctly_classified=only_correctly_classified,
                use_amp=use_amp,
                **localization_kwargs,
            )
            output_iobb = family_results["IoBB"]
            output_iou = family_results["IoU"]
            local_iobb_metrics = output_iobb[0]
            logger.info("IoBB Localization")
            evaluate.print_metrics(local_iobb_metrics)
            local_iou_metrics = output_iou[0]
            logger.info("IoU Localization")
            evaluate.print_metrics(local_iou_metrics)

        ## save checkpoints
        # save the latest resumable checkpoint (every epoch, in both modes)
        save_checkpoint(epoch, model, optimizer, last_checkpoint_path)

        if localization_sweep is not None:
            # Sweep selection: the artifact manager first computes the
            # in-memory selection for the completed epoch, then persists the
            # epoch snapshot, manifest, and relative model_best.pth symlink.
            # It raises on failure, so a partial sweep result can never commit
            # checkpoint, manifest, symlink, or log state.
            variant_iou_accuracy = {
                key: sweep_results[key]["IoU"][0]["IoU_accuracy_macro"]
                for key in localization_sweep.variant_keys
            }
            artifact_manager.update(
                epoch=epoch,
                model=model,
                auc_roc_macro=classif_metrics["auc_roc_macro"],
                variant_iou_accuracy=variant_iou_accuracy,
            )
            _print_sweep_table(epoch, localization_sweep, sweep_results, artifact_manager.state)

            # Early stopping follows the best-overall (epoch, count) pair
            # only; epoch 0 is eligible to establish the initial best.
            best_score = float(artifact_manager.state.best_overall["score"])
            if artifact_manager.state.overall_improved:
                epochs_since_best_score = 0
                logger.info("-- New best overall sweep pair --")
            else:
                epochs_since_best_score += 1
        else:
            # get current evaluation score
            curr_epoch_score = classif_metrics["auc_roc_macro"]
            if localization_loader is not None:
                curr_epoch_score = (curr_epoch_score + output_iou[0]["IoU_accuracy_macro"]) / 2

            # save best overall checkpoint if current score is higher
            if curr_epoch_score > best_score and epoch > 0:
                best_score = float(curr_epoch_score)
                epochs_since_best_score = 0
                save_checkpoint(epoch, model, optimizer, best_checkpoint_path)
                logger.info("-- New best --")
            else:
                epochs_since_best_score += 1

        ## log results
        # log best score
        performance_curves["score"] = best_score

        # log classification performance metrics
        performance_curves["epoch"].append(epoch)
        performance_curves["train_loss"].append(train_loss)
        performance_curves["lr"].append(optimizer.param_groups[0]["lr"])
        performance_curves["classification_class"].append(classif_metrics_class)

        for classif_metric, value in classif_metrics.items():
            performance_curves[classif_metric].append(value)

        # log localization performance metrics
        if localization_sweep is not None:
            # legacy one-variant entry: the primary variant's exact results in
            # the historical schema, so existing consumers keep finding one
            # localization entry per epoch
            primary_key = localization_sweep.primary_variant_key
            performance_curves["localization"].append(
                _legacy_localization_log_entry(sweep_results[primary_key])
            )

            # additive sweep entries: complete exact results for every
            # variant, plus the per-epoch selection summary
            performance_curves["localization_sweep"].append(
                _sweep_log_entry(localization_sweep, sweep_results)
            )
            performance_curves["localization_sweep_selection"].append(
                _sweep_selection_log_entry(
                    localization_sweep, artifact_manager.state, sweep_timings
                )
            )
        elif localization_loader is not None:
            # one combined helper for both loops: one calibration feeding both
            # families in the default mode, or the legacy per-metric sequence
            performance_curves["localization"].append(
                _legacy_localization_log_entry(family_results)
            )

        # save performance curves to disk as compressed pickle
        save_performance_log(performance_curves, log_path)

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # reset optimizer state if more than one loss function
        if num_criteria > 1:
            optimizer.state.clear()

        # early stop if no improvement (best-overall only in sweep mode)
        if epochs_since_best_score >= early_stop_num_epochs:
            logger.info(f">>> No improvement in {epochs_since_best_score} epochs, stopping <<<")
            break

        # terminate at exact epoch number
        if epoch >= terminate_epoch - 1:
            logger.info("Early termination")
            break
