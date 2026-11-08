"""RAD-DINO encoder (frozen DINOv2 ViT-B/14), used as an external feature source.

RAD-DINO ships **no classification head** (`microsoft/rad-dino`, arXiv 2401.10815),
so the row it supports is "frozen encoder + linear probe" — the protocol of the
source paper, not a checkpoint we can call like the TXV models. Two properties make
this module the encoder and nothing else:

* the preprocessing is the Hugging Face **processor's own** (`518²`, ImageNet
  statistics) — no 518/mean/std constant is re-implemented here, because a
  hand-written copy is how a comparison silently stops being "their model";
* `encode` is deterministic and per-sample (ViT, no batch norm): a row's features
  must not depend on which other images shared its batch, since the features are
  cached across runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch as th
from PIL import Image

def _as_processor_image(image):
    """2-D grayscale array -> PIL image; anything else passes through.

    The study's loaders hand out ``(H, W)`` uint8 arrays, while the HF image
    processors only accept single images when a channel axis is present (PIL or
    ``(C, H, W)`` / ``(H, W, C)``). Converting here keeps that quirk out of the
    callers: RAD-DINO's own processor then applies its `do_convert_rgb`, which is
    how the encoder was trained.
    """
    if isinstance(image, np.ndarray) and image.ndim == 2:
        return Image.fromarray(image)
    return image


#: the HF hub id of the published encoder
RADDINO_MODEL_ID = "microsoft/rad-dino"
#: RAD-DINO's native input (37 x 14 patches, grid-clean). Never rescaled: the
#: processor owns the number, this constant is for prose/artifact bookkeeping.
RADDINO_RESOLUTION = 518


class RaddinoEncoder:
    """Wrap a HF DINOv2 model + its processor into a feature extractor.

    ``model`` only has to accept ``pixel_values`` and return something carrying
    ``last_hidden_state`` or ``pooler_output``; the processor only has to accept
    ``processor(images=..., return_tensors="pt")`` and return ``pixel_values``.
    Keeping both injectable is what lets the contract tests run on a stub tower.
    """

    def __init__(self, model, processor, device: str | th.device = "cpu") -> None:
        self.device = th.device(device)
        # move here, not in the caller: an encoder that pins `device` for its
        # inputs but leaves the weights behind fails only at the first conv
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.processor = processor

    @property
    def hidden_size(self) -> int:
        for source in (getattr(self.model, "config", None), self.model):
            value = getattr(source, "hidden_size", None)
            if value:
                return int(value)
        raise AttributeError(
            "cannot determine the encoder's hidden size: the model has neither "
            "`config.hidden_size` nor `hidden_size`"
        )

    @property
    def input_resolution(self) -> tuple[int, int]:
        """`(height, width)` the processor actually produces.

        Read off the processor rather than assumed: RAD-DINO's checkpoint ships a
        `BitImageProcessor` whose `size` is `{"shortest_edge": 518}` plus a
        `crop_size` of 518 x 518, not the `{"height", "width"}` form a
        `ViTImageProcessor` uses. Both spellings end at the same square input, but
        the number must come from the object that applies it.
        """
        size = getattr(self.processor, "size", None) or {}
        crop = getattr(self.processor, "crop_size", None) or {}
        if size.get("height") and size.get("width"):
            return int(size["height"]), int(size["width"])
        if crop.get("height") and crop.get("width"):
            return int(crop["height"]), int(crop["width"])
        edge = size.get("shortest_edge") if isinstance(size, dict) else None
        if edge:
            return int(edge), int(edge)
        raise AttributeError(f"cannot read the input resolution from {type(self.processor)!r}")

    def encode(self, images: Sequence) -> np.ndarray:
        """`(N, hidden)` float32 features for a list of images.

        The HF processor does the resizing and normalisation; the CLS token of the
        last hidden state is the feature (the checkpoint's pooler is a freshly
        initialised linear layer, so it is not part of the pretrained encoder).
        """
        images = list(images)
        if not images:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        inputs = self.processor(
            images=[_as_processor_image(image) for image in images],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            with th.no_grad():
                output = self.model(pixel_values=pixel_values)
        finally:
            if was_training:
                self.model.train()

        hidden = getattr(output, "last_hidden_state", None)
        features = hidden[:, 0] if hidden is not None else output.pooler_output
        return features.detach().float().cpu().numpy()


def load_raddino(
    model_id: str = RADDINO_MODEL_ID,
    revision: str | None = None,
    device: str | th.device = "cpu",
) -> tuple[RaddinoEncoder, str]:
    """Load the encoder and the exact Hub commit it came from.

    The revision is resolved to a commit sha first and then used for **both** the
    processor and the weights, so a later `main` move cannot change a rerun; the
    sha is what the arm artifact records.
    """
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor, AutoModel

    if revision is None:
        revision = HfApi().model_info(model_id).sha
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).eval()
    return RaddinoEncoder(model=model, processor=processor, device=device), str(revision)
