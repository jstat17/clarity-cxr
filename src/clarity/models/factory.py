"""Public model factory: the subset the companion ships.

The public models are declared *here* (as lazy ``module:function`` targets) so
no other model catalogue is needed; importing this module never pulls an
unreleased model.
"""

from __future__ import annotations

from torch import nn

from clarity.models import registry
from clarity.models.registry import declare_model

# ---------------------------------------------------------------------------
# Public-bound models (shipped in the companion)
# ---------------------------------------------------------------------------
declare_model("clarity", "t-16", target="clarity.models.clarity:get_clarity_t_16", public=True)
declare_model("clarity", "s-8", target="clarity.models.clarity:get_clarity_s_8", public=True)
declare_model("clarity", "s-16", target="clarity.models.clarity:get_clarity_s_16", public=True)
declare_model("convnextv2", "b", target="clarity.models.convnext.convnextv2:get_convnextv2_b", public=True)


def get_public_model(
    model_name: str,
    model_size: str,
    num_classes: int,
    weights: dict | str | None,
    **kwargs,
) -> nn.Module | None:
    """Build a public model, or return ``None`` for an unreleased/unknown name."""
    try:
        return registry.get_model(
            model_name, model_size, num_classes, weights, public_only=True, **kwargs
        )
    except registry.ModelNotFoundError:
        return None


def get_model(
    model_name: str,
    model_size: str,
    num_classes: int,
    weights: dict | str | None,
    **kwargs,
) -> nn.Module:
    """Public factory used by the companion CLIs (raises for unreleased names)."""
    try:
        return registry.get_model(
            model_name, model_size, num_classes, weights, public_only=True, **kwargs
        )
    except registry.ModelNotFoundError as exc:
        raise ValueError(
            f"unknown public model {model_name!r}/{model_size!r}; the public "
            "factory ships CLARiTy and the ConvNeXtV2-B teacher only"
        ) from exc


def available_models() -> list[str]:
    """Public model labels, e.g. ``['clarity-s-16', 'convnextv2-b']``."""
    return registry.available_models(public_only=True)
