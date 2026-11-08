"""Training loop orchestration (device/compile/train_and_evaluate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from clarity.training import loop as train_utils
from clarity.localization.sweep import LocalizationSweepSpec
from clarity.utils.runtime import set_primary_device, wrap_parallel


def run_training_loop(
    device: str,
    parallel: bool,
    jit_compile: bool,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    early_stop_num_epochs: int | float,
    terminate_epoch: int | float,
    train_loader: DataLoader,
    evaluate_loader: DataLoader,
    accumulation_factor: int,
    scheduler: LambdaLR,
    save_dir: Path,
    validate_localization_loader: DataLoader | None = None,
    mp_workers: int | None = None,
    localization_kwargs: dict[str, Any] = dict(),
    use_amp: bool = False,
    localization_sweep: LocalizationSweepSpec | None = None,
) -> None:
    # use DataParallel if parallel is enabled
    model, device = wrap_parallel(model, parallel, device)

    # Set the current CUDA device when applicable. CPU execution must not
    # call torch.cuda.set_device().
    device = set_primary_device(device)

    # send model to device
    model = model.to(device)

    # compile model
    if jit_compile:
        model.compile()
        # compile the loss module(s) too: the eager combined loss is a
        # significant share of the remaining step time.
        # The sweep/evaluation paths never call the training criterion, so
        # this cannot trigger recompilation from the multi-variant forward.
        # Graph breaks on stable per-configuration attributes (e.g. inactive
        # loss branches guarded by zero coefficients) specialize at first
        # call; shapes are constant across optimizer steps within an epoch.
        if isinstance(criterion, list):
            criterion = [th.compile(c) for c in criterion]
        else:
            criterion = th.compile(criterion)

    train_utils.train_and_evaluate(
        model=model,
        train_loader=train_loader,
        evaluation_loader=evaluate_loader,
        evaluation_split="validate",
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        early_stop_num_epochs=early_stop_num_epochs,
        terminate_epoch=terminate_epoch,
        accumulation_factor=accumulation_factor,
        scheduler=scheduler,
        save_dir=save_dir,
        localization_loader=validate_localization_loader,
        mp_workers=mp_workers,
        localization_kwargs=localization_kwargs,
        use_amp=use_amp,
        localization_sweep=localization_sweep,
    )
