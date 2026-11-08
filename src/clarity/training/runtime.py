"""Parallel wrapping and global-seed setup for training runs."""

from __future__ import annotations

import random

import numpy as np
import torch as th
import torch.nn as nn

from clarity.utils.runtime import order_device_ids


def wrap_parallel(model: nn.Module, device: th.DeviceObjType) -> nn.DataParallel:
    """Wrap model in DataParallel, send to device and return reference.

    Args:
        model (nn.Module): Model to wrap
        device (th.DeviceObjType): Compute device

    Returns:
        nn.DataParallel: Wrapped model
    """
    device_ids = order_device_ids(str(device))
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    return model


def unwrap_parallel(
    model: nn.DataParallel | nn.Module, device: th.DeviceObjType
) -> tuple[nn.Module, bool]:
    """Unwrap DataParallel model to simple nn.Module (unchanged if already nn.Module instance).

    Args:
        model (nn.DataParallel | nn.Module): Model to unwrap
        device (th.DeviceObjType): Compute device

    Returns:
        tuple[nn.Module, bool]: Unwrapped model and whether it was previously wrapped
    """
    was_wrapped = False
    if isinstance(model, nn.DataParallel):
        model = model.module
        model = model.to(device)
        was_wrapped = True

    return model, was_wrapped


def set_global_seed(seed: int = 0) -> None:
    """Set the global seed across random number generators for reproducibility.

    Args:
        seed (int): Seed value. Defaults to 0
    """
    # set python's random library seed
    random.seed(seed)
    # set numpy seed
    np.random.seed(seed)
    # set torch seed
    th.manual_seed(seed)
    # set torch cuda seed
    th.cuda.manual_seed(seed)
    # set torch cuda seed for all GPUs
    th.cuda.manual_seed_all(seed)
