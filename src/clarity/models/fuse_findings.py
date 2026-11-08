"""Model-output adapter for the VinDr-CXR transfer arms.

The VinDr label map is not a rename of the NIH label space: ``Nodule/Mass`` is
one VinDr class but two NIH classes, and the A1 (NIH-calibrated) and A2 (Target-calibrated) arms keep both so
they can be fused. The evaluation code, however, must see the **target** label
space (6 localization findings / 11 classification findings) and nothing else.

This module applies a :class:`clarity.data.vindr.FindingOutputPlan` to whatever
a model returns, without the model knowing about VinDr:

* A1/A2 localization: 7 masked outputs -> fuse Mass/Nodule -> 6;
* A1/A2 classification: 12 masked outputs -> fuse -> 11;
* 11-class VinDr head, localization: 11 head outputs -> select the 6 box findings;
* 11-class VinDr head, classification: identity (no adapter is installed).

Both class-indexed layouts are handled: the flat ``[B, C, ...]`` logits /
heatmaps (class axis 1) and the sweep-mode ``[V, B, C, H, W]`` variant maps that
CLARiTy returns whenever a call-level attention-layer count is selected (class
axis 2 — see :func:`_class_axis`).

The plan is derived from the frozen mapping, so a wrong ``logit_mask`` fails
loudly instead of permuting findings. The wrapper delegates attribute access to
the model, so attention-layer introspection still works.
"""

from __future__ import annotations

from torch import nn

from clarity.data.vindr import FindingOutputPlan, apply_finding_output_plan


class FindingOutputAdapter(nn.Module):
    """Wrap ``model`` so its class axis matches a target finding space."""

    def __init__(self, model: nn.Module, plan: FindingOutputPlan) -> None:
        super().__init__()
        if plan.is_identity:
            raise ValueError(
                "the plan is the identity; do not install an adapter "
                f"for {plan.label_space!r}"
            )
        self.model = model
        self.plan = plan

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        applied = {"count": 0}
        adapted = _map_outputs(outputs, self.plan, applied)
        if applied["count"] == 0:
            raise RuntimeError(
                f"no model output tensor has {self.plan.emitted_width} class(es); "
                f"cannot apply the {self.plan.label_space!r} VinDr adapter"
            )
        return adapted

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = self.__dict__.get("_modules", {}).get("model")
            if model is None:
                raise
            return getattr(model, name)


def _class_axis(value, plan: FindingOutputPlan) -> int | None:
    """Axis of the class dimension in one model output tensor, if it has one.

    Two CLARiTy return layouts carry a class axis:

    * ``[B, C, ...]`` for logits and heatmaps (``C`` at axis 1);
    * ``[V, B, C, H, W]`` for the sweep-mode variant maps
      (``localization_layer_counts``; ``C`` at axis **2**).

    The sweep layout matters: it is exactly what ``get_model_attention`` asks a
    CLARiTy model for whenever a call-level attention-layer count is selected
    (``-alc``), i.e. on every localization evaluation. Checking only axis 1
    left those maps unfused, so the evaluator read rows ``0..n`` of a
    *wider* map tensor and silently scored the wrong findings.
    """
    if value.ndim >= 5 and value.shape[2] == plan.emitted_width:
        return 2
    if value.ndim >= 2 and value.shape[1] == plan.emitted_width:
        return 1
    return None


def _map_outputs(value, plan: FindingOutputPlan, applied: dict):
    """Recursively apply the plan to any class-indexed tensor in ``value``."""
    import torch as th

    if isinstance(value, th.Tensor):
        axis = _class_axis(value, plan)
        if axis is None:
            return value
        applied["count"] += 1
        if axis == 1:
            return apply_finding_output_plan(value, plan)
        # ``apply_finding_output_plan`` is the axis-1 contract: move the class
        # axis into place, apply, move it back. The sweep tensor stays
        # ``[V, B, C', H, W]`` so ``_single_count_sweep_map`` keeps working.
        moved = value.movedim(axis, 1)
        return apply_finding_output_plan(moved, plan).movedim(1, axis)
    if isinstance(value, tuple):
        return tuple(_map_outputs(item, plan, applied) for item in value)
    if isinstance(value, list):
        return [_map_outputs(item, plan, applied) for item in value]
    if isinstance(value, dict):
        return {key: _map_outputs(item, plan, applied) for key, item in value.items()}
    return value
