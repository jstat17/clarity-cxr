"""HiResCAM heatmaps (the ``cam`` localization method).

The transfer study reads the external classifiers (TorchXRayVision DenseNet121,
RAD-DINO + linear probe) and the ConvNeXtV2-B teacher through HiResCAM, then
hands the maps to the same OneBox calibration and evaluation as the CLARiTy
attention maps. This module is that map extractor.

Architectures it knows: the external wrappers of :mod:`clarity.external`
(``ExternalCam``, which carry their own CAM layer), ``ConvNeXtV2-B`` and
``CLARiTy-S-16``. Any other CAM method or architecture is delegated to the
extractor stub, which raises: other CAM methods are not part of this release.

Requires the ``grad-cam`` package, imported when the extractor is first used.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import numpy as np
import torch as th
from torch import nn


def reshape_transform_vit(
    tensor: th.Tensor, n_patches_each_dim: int = 14, num_class_tokens: int = 1
) -> th.Tensor:
    """``(N, tokens, C)`` -> ``(N, C, h, w)`` by dropping the prefix tokens."""
    height = width = n_patches_each_dim
    result = tensor[:, num_class_tokens:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def unwrap_for_cam(model: nn.Module) -> nn.Module:
    """Peel ``DataParallel`` and the VinDr output adapter off a model.

    Both shadow ``fullname`` and ``.model``; the CAM layer lookup must see the
    architecture itself or it would resolve on the wrong object.
    """
    from clarity.models.fuse_findings import FindingOutputAdapter

    while True:
        if isinstance(model, nn.DataParallel):
            model = model.module
        elif isinstance(model, FindingOutputAdapter):
            model = model.model
        else:
            return model


def _target_layers(model: nn.Module) -> list[nn.Module] | None:
    match model.fullname:
        case "ExternalCam":
            return list(model.cam_target_layers)
        case "ConvNeXtV2-B" | "ConvNeXtV2Segment-B":
            return [model.model.norm_pre]
        case "CLARiTy-S-16":
            return [model.blocks[-1].norm1]
    return None


def _reshape_transform(model: nn.Module, image_size: int):
    match model.fullname:
        case "ExternalCam":
            # None for a CNN wrapper (maps are already (N, C, H, W)); the ViT
            # wrapper carries a patch-token reshape closure
            return model.cam_reshape_transform
        case "CLARiTy-S-16":
            patch_size = int(model.fullname.split("-")[-1])
            return partial(
                reshape_transform_vit,
                n_patches_each_dim=image_size // patch_size,
                num_class_tokens=model.num_prefix_tokens,
            )
    return None


def get_cam_heatmaps(
    model: nn.Module,
    images: th.Tensor,
    labels: th.Tensor,
    filenames: tuple[str],
    method: str = "hirescam",
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
    only_dist_grad: bool = False,
    **kwargs,
) -> dict[str, dict[int, np.ndarray]]:
    """One HiResCAM map per (image, positive label) plus ``labels``/``probas``/``preds``.

    The return layout is the one the evaluator expects from every map extractor:
    ``{filename: {class_index: map, "labels": ..., "probas": ..., "preds": ...}}``.
    """
    architecture = unwrap_for_cam(model)
    layers = _target_layers(architecture)
    if method.casefold() != "hirescam" or layers is None:
        # other methods/architectures: not part of this release (the stub raises)
        from clarity.localization.legacy import get_cam_heatmaps as legacy_get_cam_heatmaps

        return legacy_get_cam_heatmaps(
            model,
            images,
            labels,
            filenames,
            method=method,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
            only_dist_grad=only_dist_grad,
            **kwargs,
        )

    from pytorch_grad_cam import HiResCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    filenames = np.array(filenames, dtype=str)
    cam_dict = defaultdict(dict)
    reshape_transform = _reshape_transform(architecture, images.size(-1))

    # pytorch-grad-cam's constructor puts the model in eval mode anyway; do it
    # explicitly so the contract does not depend on a third-party constructor.
    model.eval()
    model.only_classify = True
    model.only_dist_classify = False

    with HiResCAM(model=model, target_layers=layers, reshape_transform=reshape_transform) as cam:
        for col_idx in range(labels.shape[1]):
            image_idxs = [
                row for row in range(labels.shape[0]) if labels[row, col_idx].item() == 1
            ]
            if not image_idxs:
                continue
            maps = cam(
                input_tensor=images[image_idxs],
                targets=[ClassifierOutputTarget(col_idx)] * len(image_idxs),
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth,
            )
            for filename, cam_map in zip(filenames[image_idxs], maps):
                cam_dict[str(filename)][col_idx] = cam_map

    model.eval()
    with th.no_grad():
        logits = model(images)
        probas = th.sigmoid(logits)
        preds = (probas > 0.5).float()
    model.only_classify = False

    probas = probas.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for filename, img_labels, img_probas, img_preds in zip(filenames, labels, probas, preds):
        cam_dict[str(filename)]["labels"] = img_labels
        cam_dict[str(filename)]["probas"] = img_probas
        cam_dict[str(filename)]["preds"] = img_preds
    return cam_dict
