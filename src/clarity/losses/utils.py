"""Helpers shared by the paper loss subset (``wbce`` and ``dice``)."""

from __future__ import annotations

import torch as th


def get_binary_batch_weights(targets: th.Tensor, pred_background: bool = False) -> th.Tensor:
    """Per-label positive/negative weights for the weighted BCE.

    ``w_P = (|P| + |N|) / |P|`` and ``w_N = (|P| + |N|) / |N|`` over the
    batch, as defined in the paper's WBCE appendix.
    """

    def _get_weights(targets: th.Tensor) -> th.Tensor:
        targets = (targets > 0.5).to(targets.dtype)

        pos_count = targets.sum()
        neg_count = targets.numel() - pos_count

        beta_p = (pos_count + neg_count) / pos_count
        beta_n = (pos_count + neg_count) / neg_count

        return th.where(targets == 1, beta_p, beta_n)

    if pred_background:
        non_background_weights = _get_weights(targets[:, 1:])
        background_weights = th.ones(targets.size(0), 1, device=targets.device)
        return th.cat([background_weights, non_background_weights], dim=1)

    return _get_weights(targets)


def reduce(tensor: th.Tensor, reduction: str) -> th.Tensor:
    """Apply a registry-style reduction (``mean``/``sum``/``none``)."""
    match reduction.casefold():
        case "mean":
            return tensor.mean()
        case "sum":
            return tensor.sum()
        case "none":
            return tensor
        case _:
            raise ValueError(
                f"unsupported reduction {reduction!r}; expected 'mean', 'sum' or 'none'"
            )
