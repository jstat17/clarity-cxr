"""Loss registry: ``name`` → loss class, populated by decorators.

Loss modules are small and are imported together by :mod:`clarity.losses`, so decorator registration at import time is
both simple and sufficient — unlike models, no lazy module table is needed.

Usage::

    @register_loss("wbce")
    class WeightedBCEWithLogitsLoss(nn.Module): ...

    loss = get_loss("wbce", pred_background=False)

Names are case-insensitive; :func:`available_losses` lists everything that has
been registered in the current process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Type

from torch import nn

#: canonical name → loss class (preserved for introspection/debugging)
LOSS_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_loss(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Register a loss class under ``name``; duplicates are rejected."""
    canonical = str(name).casefold()

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        existing = LOSS_REGISTRY.get(canonical)
        if existing is not None:
            raise ValueError(
                f"loss {canonical!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        LOSS_REGISTRY[canonical] = cls
        return cls

    return decorator


def available_losses() -> list[str]:
    """Sorted canonical names of the registered losses."""
    return sorted(LOSS_REGISTRY)


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Instantiate a registered loss by name.

    Raises:
        ValueError: the name is unknown; the message lists the registered
            names (this replaces the historical bare ``KeyError``).
    """
    key = str(loss_name).casefold()
    try:
        loss_cls = LOSS_REGISTRY[key]
    except KeyError:
        available = ", ".join(available_losses()) or "<none>"
        raise ValueError(f"unknown loss {loss_name!r}; available: {available}") from None
    return loss_cls(**kwargs)
