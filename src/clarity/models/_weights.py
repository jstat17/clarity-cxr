"""Shared state-dict helpers for the model families.

``linearize_state_dict`` strips ``DataParallel`` prefixes and
``load_compatible_weights`` loads the shape-compatible subset (used by the
weight loaders and the teacher/baseline factories).
"""

import torch as th
from torch import nn


def linearize_state_dict(state_dict: dict, rm_key: str = "module.") -> dict:
    n_rm = len(rm_key)
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(rm_key):
            new_state_dict[k[n_rm:]] = v  # Remove the prefix
        else:
            new_state_dict[k] = v

    return new_state_dict


def load_compatible_weights(model: nn.Module, weights: dict[str, th.Tensor]) -> nn.Module:
    """Load weights into a PyTorch model, ignoring keys that don't exist in the model.

    Args:
        model (nn.Module): PyTorch model
        weights (dict[str, th.Tensor]): Dictionary containing weight tensors

    Returns:
        nn.Module: PyTorch model with loaded weights
    """
    # get model's state dictionary
    model_state_dict: dict[str, th.Tensor] = model.state_dict()

    # create a new dictionary with only matching keys
    compatible_weights = {
        k: v
        for k, v in weights.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }

    # update model state dictionary with compatible weights
    model_state_dict.update(compatible_weights)

    # load the updated state dictionary into the model
    model.load_state_dict(model_state_dict)

    return model
