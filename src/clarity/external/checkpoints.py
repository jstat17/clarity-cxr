"""Published TorchXRayVision checkpoints used as VinDr external baselines.

These are *comparators* for the transfer study, not models of this project.
Everything a row needs to justify itself lives here — the
native input resolution, the training corpus exactly as the checkpoint documents
it, whether that corpus contains the target dataset, and the citation — so the
caveat is data rather than prose in the results doc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torchxrayvision as xrv

#: cache the library itself uses for downloaded weights
XRV_CACHE_DIR = Path.home() / ".torchxrayvision" / "models_data"

_COHEN_2020 = (
    "Cohen, Hashir, Brooks & Bertrand, 'On the limits of cross-domain "
    "generalization in automated X-ray prediction', MIDL 2020 (arXiv:2002.02497)"
)
_TXV_2022 = (
    "Cohen et al., 'TorchXRayVision: A library of chest X-ray datasets and models', "
    "MIDL 2022 (arXiv:2111.00595)"
)


@dataclass(frozen=True)
class XrvCheckpoint:
    """One published checkpoint and the provenance a comparison row must carry."""

    weight_name: str
    #: label used in the generated tables
    arm_label: str
    resolution: int
    #: the corpus as the checkpoint documents it
    training_corpus: str
    #: True when the target dataset (VinDr-CXR) is part of that corpus
    training_corpus_includes_target: bool
    citation: str
    supervision: str
    supervision_note: str = ""

    @property
    def weights_url(self) -> str:
        return xrv.models.model_urls[self.weight_name]["weights_url"]

    @property
    def checkpoint_path(self) -> Path:
        return XRV_CACHE_DIR / self.weights_url.rsplit("/", 1)[-1]

    def checkpoint_sha256(self) -> str | None:
        """sha256 of the downloaded checkpoint, or None when it is not cached."""
        path = self.checkpoint_path
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


XRV_CHECKPOINTS: dict[str, XrvCheckpoint] = {
    "densenet121-res224-nih": XrvCheckpoint(
        weight_name="densenet121-res224-nih",
        arm_label="TXV DenseNet121 (NIH, zero-shot)",
        resolution=224,
        training_corpus="NIH ChestX-ray14",
        training_corpus_includes_target=False,
        citation=f"{_COHEN_2020}; {_TXV_2022}",
        supervision="zero-shot",
    ),
    "densenet121-res224-all": XrvCheckpoint(
        weight_name="densenet121-res224-all",
        arm_label="TXV DenseNet121 (union, zero-shot)",
        resolution=224,
        training_corpus=(
            "nih-pc-chex-mimic_ch-google-openi-rsna (as documented by the checkpoint)"
        ),
        training_corpus_includes_target=False,
        citation=f"{_COHEN_2020}; {_TXV_2022}",
        supervision="zero-shot",
    ),
    "resnet50-res512-all": XrvCheckpoint(
        weight_name="resnet50-res512-all",
        arm_label="TXV ResNet50 (union incl. VinDr)",
        resolution=512,
        training_corpus="pc-nih-rsna-siim-vin (as documented by the checkpoint)",
        training_corpus_includes_target=True,
        citation=f"{_TXV_2022} (no model-specific paper for this checkpoint)",
        supervision="supervised",
        supervision_note=(
            "the corpus includes VinDr-CXR, but the checkpoint does not document "
            "which split or label policy it used; treat this row as a supervised "
            "reference, not as a zero-shot comparison"
        ),
    ),
}


def xrv_checkpoint(weight_name: str) -> XrvCheckpoint:
    """Look up a checkpoint, listing the known names on a typo."""
    try:
        return XRV_CHECKPOINTS[weight_name]
    except KeyError:
        known = ", ".join(sorted(XRV_CHECKPOINTS))
        raise ValueError(
            f"unknown TorchXRayVision checkpoint {weight_name!r}; known: {known}"
        ) from None


def load_xrv_model(weight_name: str):
    """Load a published checkpoint in eval mode, emitting probabilities.

    `apply_sigmoid=True` is set explicitly so the output is a probability whatever
    the checkpoint defaults to; the library's per-class `op_norm` rescaling keeps
    each class monotone, so AUC (our metric) is unaffected while F1/sens/spec at
    the study's 0.5 threshold stay meaningful.
    """
    info = xrv_checkpoint(weight_name)
    if weight_name.startswith("densenet"):
        model = xrv.models.DenseNet(weights=weight_name)
    elif weight_name.startswith("resnet"):
        model = xrv.models.ResNet(weights=weight_name)
    else:  # pragma: no cover - the registry only holds the two families
        raise ValueError(f"no loader for checkpoint family {info.weight_name!r}")
    model.apply_sigmoid = True
    model.eval()
    return model
