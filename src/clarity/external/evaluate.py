"""Score a published TXV checkpoint on the VinDr split with the study's metric.

The metric is not re-implemented: `classification_metrics_from_arrays` is the same
function the CLARiTy arms use, so an external row cannot quietly average over a
different denominator. `Edema` has no positive image in the official test split,
so its AUC is undefined and the macro spans the other ten findings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from PIL import Image

from clarity.data.vindr import VINDR_DISEASES, VINDR_BOX_FINDINGS  # noqa: F401  (documented label space)
from clarity.evaluation.classification import classification_metrics_from_arrays
from clarity.external.checkpoints import xrv_checkpoint
from clarity.external.labels import map_to_study_labels
from clarity.external.preprocess import XRVPreprocessor

#: the metric keys the tables read, in the study's naming
_MACRO_KEYS = (
    "auc_roc_macro",
    "ap_macro",
    "f1_macro",
    "sensitivity_macro",
    "specificity_macro",
)
#: per-finding metric names expected from `classification_metrics_from_arrays`
PER_FINDING_KEYS = ("auc", "ap", "f1", "sensitivity", "specificity")


def _finite_mean(values: Iterable[float]) -> float | None:
    kept = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(kept)) if kept else None


def load_split_filenames(
    splits_dir: Path | str, images_path: Path | str, split: str
) -> list[str]:
    """Read a committed split list and require every name to exist on disk.

    Unlike the study's own readers this does not filter: a silently smaller test
    set would move the numbers with no trace.
    """
    from clarity.data.vindr import SPLIT_FILES
    from clarity.data.utils import read_file_lines

    path = Path(splits_dir) / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"split list not found: {path}")
    filenames = read_file_lines(path)
    images_path = Path(images_path)
    missing = [name for name in filenames if not (images_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) from {path.name} are missing under {images_path}, "
            f"e.g. {missing[:3]}"
        )
    return filenames


def label_matrix(df: pd.DataFrame, filenames: Sequence[str]) -> np.ndarray:
    """(N, 11) multi-hot matrix in `VINDR_DISEASES` order."""
    indexed = df.set_index("Filename")
    rows = []
    for name in filenames:
        labels = set(indexed.loc[name, "Labels"])
        rows.append([1 if finding in labels else 0 for finding in VINDR_DISEASES])
    return np.asarray(rows, dtype=int)


def study_metrics(
    labels: np.ndarray,
    probas: np.ndarray,
    threshold: float = 0.5,
    findings: Sequence[str] | None = None,
) -> dict:
    """Per-finding AUC/AP/F1/sens/spec plus the macro over the evaluable findings.

    `findings` names the columns and is only needed when a caller scores a subset of
    the 11 findings (the RAD-DINO probe has no `Edema` model, so it passes the 10
    it can score). A full-width matrix without `findings` is `VINDR_DISEASES` in
    order; a narrower one without names is labelled `column_<i>` (naming a prefix
    would be a lie for the probe's 10 columns). The metric itself — and therefore
    every macro — is the same function either way.
    """
    labels = np.asarray(labels, dtype=int)
    probas = np.asarray(probas, dtype=float)
    if labels.shape != probas.shape:
        raise ValueError(f"labels {labels.shape} vs probas {probas.shape}")
    if findings is None:
        if labels.shape[1] > len(VINDR_DISEASES):
            raise ValueError(
                f"expected at most {len(VINDR_DISEASES)} findings, got "
                f"{labels.shape[1]} columns"
            )
        # a full-width matrix is the 11 findings; a narrower one without names is
        # named positionally, because assuming a prefix would silently mislabel the
        # columns (the RAD-DINO path always passes findings=)
        findings = (
            list(VINDR_DISEASES)
            if labels.shape[1] == len(VINDR_DISEASES)
            else [f"column_{index}" for index in range(labels.shape[1])]
        )
    else:
        findings = list(findings)
        if len(findings) != labels.shape[1]:
            raise ValueError(
                f"{len(findings)} finding names for {labels.shape[1]} columns"
            )
        unknown = [finding for finding in findings if finding not in VINDR_DISEASES]
        if unknown:
            raise ValueError(f"unknown VinDr finding(s): {unknown}")

    preds = (probas > threshold).astype(int)
    _macro_all, per_class = classification_metrics_from_arrays(labels, preds, probas)

    per_finding: dict[str, dict[str, float | None]] = {}
    source_keys = {
        "auc": "auc_roc",
        "ap": "ap",
        "f1": "f1",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
    }
    for index, finding in enumerate(findings):
        row = {}
        for key in PER_FINDING_KEYS:
            value = float(per_class[source_keys[key]][index])
            row[key] = None if np.isnan(value) else value
        per_finding[finding] = row

    evaluable = [
        index
        for index in range(len(findings))
        if np.isfinite(float(per_class["auc_roc"][index]))
    ]
    macro_sources = {
        "auc_roc_macro": "auc_roc",
        "ap_macro": "ap",
        "f1_macro": "f1",
        "sensitivity_macro": "sensitivity",
        "specificity_macro": "specificity",
    }
    macro = {
        key: _finite_mean(float(per_class[source][index]) for index in evaluable)
        for key, source in macro_sources.items()
    }
    return {
        "labels": list(findings),
        "per_finding": per_finding,
        "n_evaluable": len(evaluable),
        "evaluable_labels": [findings[index] for index in evaluable],
        "macro": macro,
        "threshold": float(threshold),
        "n_images": int(labels.shape[0]),
    }


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L") if image.mode not in ("L", "I;16") else image)


def evaluate_xrv(
    model: th.nn.Module,
    *,
    filenames: Sequence[str],
    labels: np.ndarray,
    images_path: Path | str,
    resolution: int,
    batch_size: int = 64,
    device: str = "cpu",
    threshold: float = 0.5,
    checkpoint_labels: Sequence[str] | None = None,
) -> dict:
    """Run one checkpoint over `filenames` and score it like the study's arms.

    The model is called as `model(images)` — the contract every TXV checkpoint has —
    and its output must be the checkpoint's own class axis (18 slots), which is then
    reduced to the 11 findings by name.
    """
    images_path = Path(images_path)
    preprocess = XRVPreprocessor(resolution)
    if checkpoint_labels is None:
        checkpoint_labels = getattr(model, "pathologies", None)
    if checkpoint_labels is None:
        raise ValueError(
            "cannot determine the checkpoint's label list: pass checkpoint_labels= "
            "or use a loader that sets `model.pathologies`"
        )
    checkpoint_labels = list(checkpoint_labels)

    model = model.to(device).eval()
    probas: list[np.ndarray] = []
    for start in range(0, len(filenames), batch_size):
        chunk = filenames[start : start + batch_size]
        batch = np.stack([preprocess(_load_image(images_path / name)) for name in chunk])
        tensor = th.from_numpy(batch).to(device)
        with th.no_grad():
            outputs = model(tensor)
        if not th.is_tensor(outputs):
            raise TypeError(f"model returned {type(outputs).__name__}, expected a tensor")
        mapped = map_to_study_labels(outputs.float(), checkpoint_labels)
        probas.append(mapped.detach().cpu().numpy())

    stacked = np.concatenate(probas, axis=0)
    document = study_metrics(labels, stacked, threshold=threshold)
    document["probas"] = stacked
    document["filenames"] = list(filenames)
    return document


def build_arm_document(
    *,
    model_name: str,
    resolution: int,
    evaluation: dict,
    split_sha256: str,
    arm_label: str,
    val_evaluation: dict | None = None,
) -> dict:
    """Shape one external row like every other arm result document."""
    info = xrv_checkpoint(model_name)
    macro = dict(evaluation["macro"])
    val_macro = {key: None for key in _MACRO_KEYS}
    val_macro.update((val_evaluation or {}).get("macro") or {})
    document = {
        "schema_version": 1,
        "arm": arm_label,
        "resolution": int(resolution),
        "run": f"xrv:{model_name}",
        "split_coverage": (
            "test {} images".format(evaluation.get("n_images", "?"))
            if val_evaluation is None
            else "val {} images, test {} images".format(
                val_evaluation.get("n_images", "?"), evaluation.get("n_images", "?")
            )
        ),
        "classification": {
            "pickle": f"xrv:{model_name}",
            "sha256": info.checkpoint_sha256() or "unknown",
            "labels": list(evaluation["labels"]),
            "test": {
                "macro": macro,
                "per_finding": {
                    key: [row[key] for row in evaluation["per_finding"].values()]
                    for key in ("auc", "ap")
                },
                "per_finding_by_name": evaluation["per_finding"],
                "n_evaluable": evaluation["n_evaluable"],
                "evaluable_labels": evaluation.get("evaluable_labels"),
                "n_images": evaluation.get("n_images"),
            },
            "val": {"macro": val_macro},
        },
        "provenance": {
            "model": model_name,
            "weights_url": info.weights_url,
            "checkpoint_sha256": info.checkpoint_sha256(),
            "training_corpus": info.training_corpus,
            "training_corpus_includes_target": info.training_corpus_includes_target,
            "supervision": info.supervision,
            "supervision_note": info.supervision_note,
            "citation": info.citation,
            "input_resolution": int(resolution),
            "recipe": (
                "checkpoint's own preprocessing: grayscale, normalize to "
                "[-1024,1024], xrv.datasets.XRayResizer to the native resolution; "
                "classes reduced to the 11 VinDr findings by name "
                "(Nodule+Mass = max); scored with the study's "
                "classification_metrics_from_arrays at threshold 0.5"
            ),
            "split": "test",
            "split_sha256": split_sha256,
        },
    }
    return document


def split_sha256(splits_dir: Path | str, split: str = "test") -> str:
    from clarity.data.vindr import SPLIT_FILES

    path = Path(splits_dir) / SPLIT_FILES[split]
    return hashlib.sha256(path.read_bytes()).hexdigest()
