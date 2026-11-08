"""CAM localization for the external baseline rows — *indicative only*.

Neither the TXV checkpoints nor RAD-DINO was trained for localization: they carry
no box supervision anywhere, and RAD-DINO has no head at all. Running them through
the arms' own map extraction (HiResCAM) and the arms' own OneBox pipeline answers
one narrow question — *if you read this classifier the way we read our in-house
HiResCAM baseline, where does it look?* — and nothing more:

* the numbers are comparable to the **HiResCAM rows**, never to the CLARiTy
  attention rows (different map, different inductive bias);
* the map is taken at the last convolutional stage (TXV) or the last transformer
  block's attention-norm (RAD-DINO), with the probe's own logit as the CAM target;
* the label space is the 6 fused VinDr box findings, so `Nodule/Mass` is fused by
  max exactly as in every other arm.

Every artifact this produces must carry `supervision: image-level` and the
`indicative` caveat; nothing here is allowed to leak into a claim about the
CLARiTy arms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch as th
from torch import nn

from clarity.data.vindr import VINDR_BOX_FINDINGS, VINDR_DISEASES
from clarity.data.loader import PYTORCH_SECOND_NORMALIZATION_DICT
from clarity.external.labels import map_to_study_labels

#: the 11-finding space -> the 6 box findings' columns
BOX_COLUMNS = tuple(VINDR_DISEASES.index(name) for name in VINDR_BOX_FINDINGS)


def to_unit_range(images: th.Tensor) -> th.Tensor:
    """Undo the study loader's ImageNet normalisation: `(N,3,H,W)` -> `[0,1]`, 1 channel.

    Every arm's dataloader output has `(x - mean_c) / std_c` applied per channel
    with all three channels holding the same grayscale image, so undoing channel 0
    recovers the image exactly. The constants are imported from the loader rather
    than restated, so a change there cannot silently skew these maps.
    """
    mean = float(PYTORCH_SECOND_NORMALIZATION_DICT["mean"][0])
    std = float(PYTORCH_SECOND_NORMALIZATION_DICT["std"][0])
    return images[:, :1] * std + mean


class ExternalCamModel(nn.Module):
    """Common wrapper: the evaluator's CAM extractor sees a plain 6-logit model.

    `fullname` is registered in :mod:`clarity.localization.hirescam`, which is what
    lets `get_cam_heatmaps` drive a model it knows nothing about.
    """

    fullname = "ExternalCam"

    def __init__(
        self,
        model: nn.Module,
        cam_target_layers: Sequence[nn.Module],
        cam_reshape_transform=None,
        *,
        description: str = "",
    ) -> None:
        super().__init__()
        # a submodule, so `.eval()` / `.to(device)` on the wrapper reach the weights
        self.model = model
        self.cam_target_layers = list(cam_target_layers)
        self.cam_reshape_transform = cam_reshape_transform
        self.description = description
        # the CAM extractor sets these on every model it drives
        self.only_classify = False
        self.only_dist_classify = False

    def logits(self, images: th.Tensor) -> th.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, images: th.Tensor) -> th.Tensor:
        return self.logits(images)


class XrvCamModel(ExternalCamModel):
    """A TXV checkpoint read as a 6-finding CAM source.

    The loader hands out `(N, 3, H, W)` images in `[0, 1]` (the study's own
    preprocessing, shared with every other arm). TXV expects `[-1024, 1024]`, and
    the class axis is reduced to the 11 findings by name (`Nodule/Mass` = max)
    before the 6 box findings are selected.
    """

    #: CAM layers worth trying for each family; the grid resolution differs by
    #: 4x between them, and a coarse map caps the attainable IoU regardless of how
    #: good the model is (the arms' HiResCAM baseline reads a 56x56 map at 224)
    CAM_LAYERS = {
        "densenet": {
            "norm5": "features.norm5",  # 7x7 at 224
            "denseblock3": "features.denseblock3",  # 14x14
            "denseblock2": "features.denseblock2",  # 28x28
        },
        "resnet": {
            "layer4": "model.layer4[-1]",  # 16x16 at 512
            "layer3": "model.layer3[-1]",  # 32x32
        },
    }

    def __init__(
        self, model: nn.Module, checkpoint_labels: Sequence[str], cam_layer: str = "last"
    ) -> None:
        # TXV's ResNet *also* has a `.features` attribute - as a method, not a
        # module - so the check has to look at the object, not at hasattr
        features = getattr(model, "features", None)
        family = "densenet" if isinstance(features, nn.Module) else "resnet"
        self.family = family
        default_layer = "norm5" if family == "densenet" else "layer4"
        self.cam_layer = default_layer if cam_layer in ("last", "auto", "") else cam_layer
        if self.cam_layer not in self.CAM_LAYERS[family]:
            raise ValueError(
                f"unknown CAM layer {cam_layer!r} for a {family}; "
                f"choose from {sorted(self.CAM_LAYERS[family])}"
            )
        target = _resolve_attr(model, self.CAM_LAYERS[family][self.cam_layer])
        super().__init__(
            model, [target], None, description=f"TXV checkpoint + HiResCAM @ {self.cam_layer}"
        )
        self.checkpoint_labels = list(checkpoint_labels)

    def logits(self, images: th.Tensor) -> th.Tensor:
        # TXV models take one channel in [-1024, 1024] and pad to 3 internally
        x = to_unit_range(images.float()) * 2048.0 - 1024.0
        outputs = self.model(x)
        mapped = map_to_study_labels(outputs, self.checkpoint_labels)
        return mapped[:, list(BOX_COLUMNS)]


class XrvFindingCamModel(XrvCamModel):
    """A TXV checkpoint read as a single-finding CAM source (RSNA: ``Pneumonia``).

    The class axis stays the checkpoint's own; one named output is selected, so
    the CAM target (class index 0) is that finding and nothing else.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_labels: Sequence[str],
        finding: str = "Pneumonia",
        cam_layer: str = "last",
    ) -> None:
        super().__init__(model, checkpoint_labels, cam_layer=cam_layer)
        if finding not in self.checkpoint_labels:
            raise ValueError(f"the checkpoint has no {finding!r} output: {self.checkpoint_labels}")
        self.column = self.checkpoint_labels.index(finding)
        self.description = f"TXV checkpoint ({finding}) + HiResCAM @ {self.cam_layer}"

    def logits(self, images: th.Tensor) -> th.Tensor:
        x = to_unit_range(images.float()) * 2048.0 - 1024.0
        outputs = self.model(x)
        return outputs[:, self.column : self.column + 1]


class ConvNeXtControlCamModel(ExternalCamModel):
    """Control row: our own ConvNeXtV2-B, read exactly like the external models.

    Nothing about this row is new science — it exists to check the *pipeline*.
    Driven through the same driver/data/OneBox as the external rows it must
    reproduce the HiResCAM number of the in-house ConvNeXtV2-B model; if it
    does not, the external rows cannot be trusted.
    """

    def __init__(self, model: nn.Module, num_classes: int = len(VINDR_DISEASES)) -> None:
        super().__init__(
            model, [model.model.norm_pre], None, description="in-house ConvNeXtV2-B control"
        )
        self.num_classes = num_classes

    def logits(self, images: th.Tensor) -> th.Tensor:
        outputs = self.model(images)
        if outputs.shape[1] == len(VINDR_DISEASES):
            return outputs[:, list(BOX_COLUMNS)]
        raise ValueError(
            f"the control expects the 11-class VinDr head, got {outputs.shape[1]} logits"
        )


class RaddinoProbeCamModel(ExternalCamModel):
    """RAD-DINO encoder + fitted linear probes as a 6-finding CAM source.

    The CAM target is the probe's own decision value (a linear function of the
    CLS token), so the map explains *the thing that classifies*, not a
    hypothetical head. The encoder stays frozen — only the probe has weights.
    """

    def __init__(
        self,
        dinov2: nn.Module,
        processor,
        coefficients: dict[str, tuple[np.ndarray, float]],
        device: str | th.device = "cpu",
        n_patches_each_dim: int = 37,
        block_index: int = -1,
        findings_order: Sequence[str] | None = None,
    ) -> None:
        self.block_index = int(block_index)
        super().__init__(
            # the *nn.Module*, not the RaddinoEncoder wrapper: grad-cam takes
            # `next(model.parameters()).device` and hooks modules, so a plain
            # wrapper object would leave the model with no parameters at all
            dinov2,
            [dinov2.encoder.layer[self.block_index].norm1],
            _patch_reshape(n_patches_each_dim),
            description="RAD-DINO (frozen) + linear probe + HiResCAM",
        )
        self.processor = processor
        self.device = th.device(device)
        self.n_patches_each_dim = int(n_patches_each_dim)
        # default is the VinDr order; RSNA passes its own ("Pneumonia")
        order = tuple(findings_order) if findings_order else tuple(VINDR_BOX_FINDINGS)
        self.findings = [name for name in order if name in coefficients]
        if not self.findings:
            raise ValueError(
                f"none of {order} has a fitted probe (got {sorted(coefficients)})"
            )
        weights = th.stack(
            [th.as_tensor(np.ravel(coefficients[name][0]), dtype=th.float32) for name in self.findings]
        )
        biases = th.stack(
            [th.as_tensor(float(coefficients[name][1]), dtype=th.float32) for name in self.findings]
        )
        # buffers so they follow `.to(device)`
        self.register_buffer("probe_weights", weights)
        self.register_buffer("probe_biases", biases)
        self.to(self.device)  # the encoder is already there; the fresh buffers are not

    def logits(self, images: th.Tensor) -> th.Tensor:
        pixel_values = self._pixel_values(images)
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        features = hidden[:, 0]
        return features @ self.probe_weights.T + self.probe_biases

    def _pixel_values(self, images: th.Tensor) -> th.Tensor:
        """`(N, 3, H, W)` in [0, 1] -> the HF processor's `pixel_values`.

        The processor is the checkpoint's own (518² shortest-edge + 518 crop,
        RGB cast); going back through uint8 keeps it identical to the feature
        extraction that fitted the probes.
        """
        from PIL import Image

        arrays = (
            (to_unit_range(images.detach().float()).cpu().clamp(0, 1) * 255.0)
            .round()
            .to(th.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        pil = [Image.fromarray(np.asarray(array)[..., 0]) for array in arrays]
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        # the encoder is frozen, so without this no tensor in the graph requires a
        # gradient and grad-CAM gets `grads=None`. The input is the legitimate
        # differentiable root for a frozen-encoder CAM.
        pixel_values.requires_grad_(True)
        return pixel_values


def _resolve_attr(root, path: str):
    """`"model.layer4[-1]"` -> the object, so CAM layers are configurable."""
    import re

    current = root
    for name, index in re.findall(r"([^.\[]+)(?:\[(-?\d+)\])?", path):
        current = getattr(current, name)
        if index:
            current = current[int(index)]
    return current


def _patch_reshape(n_patches_each_dim: int):
    from functools import partial

    from clarity.localization.hirescam import reshape_transform_vit

    return partial(
        reshape_transform_vit,
        n_patches_each_dim=n_patches_each_dim,
        num_class_tokens=1,
    )
