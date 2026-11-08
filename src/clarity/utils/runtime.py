"""Device, autocast and model-wrapping helpers shared by the CLIs."""

from __future__ import annotations

from collections.abc import Iterable

import torch as th
from torch import nn

from clarity.utils.cli import parse_path


class InvalidDeviceError(Exception):
    """Raised when an invalid device is specified for parallel processing."""

    pass


def get_autocast_context(device: str | th.device, enabled: bool):
    """Return a bfloat16 autocast context for CPU or CUDA execution.

    Models and optimizers remain in their default (normally float32) dtype;
    callers should use this context only around model forward passes and loss
    computation. ``enabled=False`` preserves the ordinary float32 path while
    keeping call sites identical.
    """
    device_type = th.device(device).type
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(
            f"bfloat16 autocast is only supported here for cpu or cuda, got {device_type!r}"
        )
    return th.autocast(
        device_type=device_type,
        dtype=th.bfloat16,
        enabled=enabled,
    )


def set_primary_device(device: str | th.device) -> th.device:
    """Set the current CUDA device when applicable and return a torch device.

    CPU execution must not call ``torch.cuda.set_device``. The project only
    supports CPU and CUDA execution for training/evaluation.
    """
    device = th.device(device)
    if device.type == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        # ``torch.cuda.set_device(torch.device('cuda'))`` is rejected by some
        # PyTorch versions because the device index is implicit. Normalize it
        # to the current CUDA device before setting it.
        if device.index is None:
            device = th.device("cuda", th.cuda.current_device())
        th.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError(f"only cpu and cuda devices are supported, got {device.type!r}")
    return device


def parse_weights(s: str) -> tuple[str | dict | None, str]:
    """Parse a string into a weights label or weights dictionary. A string weights of information
    is returned for logging purposes.

    Args:
        s (str): Path-like string, or label

    Returns:
        tuple[str | dict | None, str]: Weights label or dictionary, and the weights information
    """
    # load weights from path
    if "/" in s:
        weights_path = parse_path(s)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"weights checkpoint not found: {weights_path} (from {s!r})"
            )
        checkpoint = th.load(weights_path, weights_only=True, map_location="cpu")
        weights = checkpoint["model_state_dict"]
        weights_info = str(weights_path)

    else:
        # set as 'imagenet' or other named weights
        weights = s.casefold()

        # set as random weights
        if weights == "none":
            weights = None

        weights_info = str(weights)

    return weights, weights_info


def wrap_parallel(
    model: nn.Module | Iterable[nn.Module], parallel: bool, device: str
) -> tuple[nn.Module | list[nn.Module], th.DeviceObjType]:
    """Wrap a model in DataParallel and normalize a CPU or CUDA device."""
    device_obj = th.device(device)
    if device_obj.type not in {"cpu", "cuda"}:
        raise InvalidDeviceError(
            f"only 'cpu' and 'cuda' devices are supported, not '{device_obj.type}'"
        )

    if parallel:
        if device_obj.type != "cuda":
            raise InvalidDeviceError(f"When parallel=True, device must be CUDA, not '{device}'")

        device_ids = order_device_ids(device_obj)
        if isinstance(model, Iterable):
            model = [nn.DataParallel(m, device_ids=device_ids) for m in model]
            if len(model) == 1:
                model = model[0]
        else:
            model = nn.DataParallel(model, device_ids=device_ids)

    return model, device_obj


def order_device_ids(device: str | th.device) -> list[int]:
    """Use the requested CUDA device as the DataParallel master."""
    device = th.device(device)
    if device.type != "cuda":
        raise InvalidDeviceError(f"DataParallel requires CUDA, not '{device}'")
    device_id = device.index
    if device_id is None:
        device_id = th.cuda.current_device()
    return [device_id] + [i for i in range(th.cuda.device_count()) if i != device_id]
