"""Run the external baseline arms on RSNA Pneumonia or VinDr-CXR.

Two published comparators are scored on the same split, label space and metric
as the CLARiTy arms:

* ``txv`` -- the TorchXRayVision DenseNet121-NIH checkpoint, loaded unchanged
  (zero-shot head);
* ``raddino`` -- the frozen RAD-DINO encoder behind a logistic-regression probe
  fitted on the target *train* list (C selected on *val*).

``classify_*`` return the classification metrics; ``localize`` reads either
baseline through HiResCAM and runs the same OneBox calibration (fitted on the
target validation split, test scored once) as the CLARiTy arms. The dataset is
a :class:`TransferDataset`; image and split locations are always explicit.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from clarity.data import loader as loader_utils
from clarity.data import rsna, vindr

DATASETS = ("rsna", "vindr")
#: the TorchXRayVision checkpoint of the transfer study
TXV_NIH = "densenet121-res224-nih"


@dataclass(frozen=True)
class TransferDataset:
    """What differs between the two target datasets."""

    name: str
    #: label names of the classification space
    labels: tuple[str, ...]
    #: label names of the localization space
    box_labels: tuple[str, ...]
    regions: dict
    load_dataframe: Callable[[Path], pd.DataFrame]
    split_filenames: Callable[..., tuple[list[str], list[str], list[str]]]
    bbox_dataframe: Callable[..., pd.DataFrame]


def get_dataset(name: str) -> TransferDataset:
    match name:
        case "rsna":
            return TransferDataset(
                name="rsna",
                labels=tuple(rsna.RSNA_DISEASES),
                box_labels=tuple(rsna.RSNA_DISEASES),
                regions=dict(rsna.RSNA_DISEASE_REGIONS),
                load_dataframe=rsna.get_rsna_dataframe,
                split_filenames=rsna.get_split_filenames,
                bbox_dataframe=lambda root, size, **_: rsna.get_rsna_bbox_dataframe(
                    rsna_path=root, image_size=size
                ),
            )
        case "vindr":
            return TransferDataset(
                name="vindr",
                labels=tuple(vindr.VINDR_DISEASES),
                box_labels=tuple(vindr.VINDR_BOX_FINDINGS),
                regions={n: vindr.VINDR_DISEASE_REGIONS[n] for n in vindr.VINDR_BOX_FINDINGS},
                load_dataframe=lambda root: vindr.get_vindr_dataframe(root, "all"),
                split_filenames=vindr.get_split_filenames,
                bbox_dataframe=lambda root, size, image_ids=None, images_path=None: (
                    vindr.get_vindr_bbox_dataframe(
                        root, "all", image_size=size, image_ids=image_ids, images_path=images_path
                    )
                ),
            )
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


def label_matrix(dataset: TransferDataset, df: pd.DataFrame, filenames: Sequence[str]) -> np.ndarray:
    """``(N, len(dataset.labels))`` multi-hot matrix in the dataset's label order.

    ``Labels`` is a list of names (negatives carry ``['No Finding']``), so
    membership is tested by name, never by truthiness.
    """
    indexed = df.set_index("Filename")
    rows = []
    for name in filenames:
        present = set(indexed.loc[name, "Labels"])
        rows.append([1 if label in present else 0 for label in dataset.labels])
    return np.asarray(rows, dtype=int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L") if image.mode not in ("L", "I;16") else image)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _split_lists(
    dataset: TransferDataset, splits_dir: Path, images: Path, limit: int | None
) -> dict[str, list[str]]:
    train, val, test = dataset.split_filenames(splits_dir=splits_dir, available_images_path=images)
    if limit is not None:
        val, test = val[:limit], test[:limit]
    return {"train": train, "val": val, "test": test}


def classify_txv(
    dataset: TransferDataset,
    *,
    root: Path,
    splits_dir: Path,
    images: Path,
    device: str = "cpu",
    batch_size: int = 64,
    limit: int | None = None,
    weight_name: str = TXV_NIH,
) -> dict:
    """Score the TXV checkpoint on the validation and test lists."""
    from clarity.external.checkpoints import load_xrv_model, xrv_checkpoint
    from clarity.external.evaluate import evaluate_xrv
    from clarity.external.preprocess import XRVPreprocessor

    info = xrv_checkpoint(weight_name)
    model = load_xrv_model(weight_name)
    df = dataset.load_dataframe(root)
    lists = _split_lists(dataset, splits_dir, images, limit)
    result: dict = {"arm": info.arm_label, "dataset": dataset.name, "resolution": info.resolution}
    for split in ("val", "test"):
        names = lists[split]
        labels = label_matrix(dataset, df, names)
        if dataset.name == "vindr":
            document = evaluate_xrv(
                model,
                filenames=names,
                labels=labels,
                images_path=images,
                resolution=info.resolution,
                batch_size=batch_size,
                device=device,
            )
            document.pop("probas", None)
            document.pop("filenames", None)
        else:
            # RSNA is one finding: take the checkpoint's own Pneumonia output
            preprocess = XRVPreprocessor(info.resolution)
            column = list(model.pathologies).index("Pneumonia")
            model = model.to(device).eval()
            scores = []
            for start in range(0, len(names), batch_size):
                batch = np.stack(
                    [preprocess(_load_gray(images / n)) for n in names[start : start + batch_size]]
                )
                with th.no_grad():
                    scores.append(model(th.from_numpy(batch).to(device))[:, column].float().cpu().numpy())
            score = np.concatenate(scores)
            document = {
                "n_images": len(names),
                "macro": {"auc_roc_macro": _auc(labels[:, 0], score)},
                "n_positives": int(labels.sum()),
            }
        result[split] = document
        auc = document["macro"]["auc_roc_macro"]
        print(f"{info.arm_label} [{split}] {len(names)} images | AUC {'n/a' if auc is None else f'{auc:.4f}'}")
    if limit is not None:
        result["limit"] = int(limit)
    return result


def extract_raddino_features(
    encoder, filenames: Sequence[str], images: Path, batch_size: int, tag: str
) -> np.ndarray:
    """``(N, hidden)`` float32 features in list order (deterministic, per sample)."""
    features = np.zeros((len(filenames), encoder.hidden_size), dtype=np.float32)
    started = time.time()
    for start in range(0, len(filenames), batch_size):
        chunk = filenames[start : start + batch_size]
        features[start : start + len(chunk)] = encoder.encode(
            [_load_gray(images / name) for name in chunk]
        ).astype(np.float32)
        done = start + len(chunk)
        if done % (batch_size * 10) == 0 or done == len(filenames):
            rate = done / max(time.time() - started, 1e-9)
            print(f"[{tag}] {done}/{len(filenames)} ({rate:.1f}/s)", flush=True)
    return features


def classify_raddino(
    dataset: TransferDataset,
    *,
    root: Path,
    splits_dir: Path,
    images: Path,
    features_cache: Path,
    device: str = "cpu",
    batch_size: int = 32,
    limit: int | None = None,
    encoder_factory: Callable[..., tuple] | None = None,
) -> dict:
    """Fit one probe per finding on train (C on val); report val/test metrics.

    Returns the metrics plus a ``probes`` entry (``{finding: (coef, intercept)}``)
    that :func:`localize` takes to put the same linear head on the encoder.
    """
    from clarity.external.evaluate import study_metrics
    from clarity.external.probe import RaddinoProbe
    from clarity.external.raddino import RADDINO_MODEL_ID, RADDINO_RESOLUTION, load_raddino

    df = dataset.load_dataframe(root)
    lists = _split_lists(dataset, splits_dir, images, limit)
    labels = {s: label_matrix(dataset, df, lists[s]) for s in lists}

    if features_cache.is_file():
        payload = np.load(features_cache, allow_pickle=False)
        for split in lists:
            if int(payload[f"n_{split}"]) != len(lists[split]):
                raise SystemExit(
                    f"{features_cache} holds {int(payload[f'n_{split}'])} {split} rows but "
                    f"the split has {len(lists[split])}; use another --features-cache"
                )
        features = {s: payload[s] for s in lists}
        revision = str(payload["revision"])
        print(f"reusing feature cache {features_cache}")
    else:
        encoder, revision = (encoder_factory or load_raddino)(device=device)
        features = {
            s: extract_raddino_features(encoder, lists[s], images, batch_size, s) for s in lists
        }
        features_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            features_cache,
            revision=np.array(revision),
            **{f"n_{s}": np.array(len(lists[s])) for s in lists},
            **features,
        )

    probes: dict[str, RaddinoProbe] = {}
    coefficients: dict[str, tuple[np.ndarray, float]] = {}
    for index, finding in enumerate(dataset.labels):
        if labels["train"][:, index].sum() == 0 or labels["val"][:, index].sum() == 0:
            print(f"skipping {finding}: no positive in train and/or val")
            continue
        probe = RaddinoProbe(train_filenames=lists["train"]).fit(
            features["train"], labels["train"][:, index], lists["train"],
            features["val"], labels["val"][:, index],
        )
        probes[finding] = probe
        classifier = probe._classifier  # noqa: SLF001 - the object the row is scored with
        coefficients[finding] = (classifier.coef_.ravel(), float(classifier.intercept_[0]))
        print(f"{finding:20s} C={probe.chosen_c:g}")

    names = [f for f in dataset.labels if f in probes]
    columns = [dataset.labels.index(f) for f in names]
    result: dict = {
        "arm": f"RAD-DINO + linear probe ({RADDINO_RESOLUTION})",
        "dataset": dataset.name,
        "resolution": RADDINO_RESOLUTION,
        "model": RADDINO_MODEL_ID,
        "revision": revision,
        "chosen_c": {f: probes[f].chosen_c for f in names},
        "probes": coefficients,
    }
    for split in ("val", "test"):
        probas = np.stack([probes[f].predict_proba(features[split]) for f in names], axis=1)
        if dataset.name == "vindr":
            document = study_metrics(labels[split][:, columns], probas, findings=names)
        else:
            document = {
                "n_images": len(lists[split]),
                "n_positives": int(labels[split].sum()),
                "macro": {"auc_roc_macro": _auc(labels[split][:, 0], probas[:, 0])},
            }
        result[split] = document
        auc = document["macro"]["auc_roc_macro"]
        print(f"RAD-DINO [{split}] {len(lists[split])} images | AUC {'n/a' if auc is None else f'{auc:.4f}'}")
    if limit is not None:
        result["limit"] = int(limit)
    return result


# ---------------------------------------------------------------------------
# localization
# ---------------------------------------------------------------------------


def _loaders(
    dataset: TransferDataset,
    *,
    root: Path,
    splits_dir: Path,
    images: Path,
    segmentations: Path | None,
    resolution: int,
    batch_size: int,
    workers: int,
    limit: int | None,
):
    _train, val, test = dataset.split_filenames(splits_dir=splits_dir, available_images_path=images)
    if limit is not None:
        val, test = val[:limit], test[:limit]
    df = dataset.load_dataframe(root)
    df_bbox = dataset.bbox_dataframe(
        root,
        resolution,
        image_ids={Path(n).stem for n in val + test},
        images_path=images,
    )
    loaders = []
    for split, names in (("validate", val), ("test", test)):
        data = loader_utils.MultilabelDataset(
            df=df,
            images_path=images,
            img_shape=resolution,
            split=split,
            split_pct1=0.8,
            split_pct2=1.0,
            possible_labels=list(dataset.box_labels),
            filenames=names,
            df_bbox=df_bbox,
            multi_box_gt=True,
            return_filename=True,
            segmentation_path=segmentations,
            label_anatomical_regions=dataset.regions,
        )
        loaders.append(
            th.utils.data.DataLoader(
                dataset=data,
                batch_size=batch_size,
                num_workers=workers,
                shuffle=False,
                collate_fn=loader_utils.custom_collate,
                worker_init_fn=loader_utils.dataloader_worker_init_fn,
            )
        )
    return loaders[0], loaders[1], val, test


def build_cam_model(
    baseline: str,
    dataset: TransferDataset,
    *,
    device: str,
    probes: dict[str, tuple[np.ndarray, float]] | None = None,
    cam_layer: str = "last",
    weight_name: str = TXV_NIH,
):
    """The baseline as a CAM source whose outputs are the dataset's box findings."""
    from clarity.external.checkpoints import load_xrv_model
    from clarity.external.localization import (
        RaddinoProbeCamModel,
        XrvCamModel,
        XrvFindingCamModel,
    )
    from clarity.external.raddino import RADDINO_RESOLUTION, load_raddino

    if baseline == "txv":
        model = load_xrv_model(weight_name)
        if dataset.name == "vindr":
            return XrvCamModel(model, model.pathologies, cam_layer=cam_layer)
        return XrvFindingCamModel(model, model.pathologies, "Pneumonia", cam_layer=cam_layer)
    if baseline == "raddino":
        if not probes:
            raise SystemExit("raddino localization needs the fitted probes (run classify first)")
        encoder, revision = load_raddino(device=device)
        print(f"RAD-DINO revision {revision}")
        block = -1 if cam_layer in ("last", "auto", "") else int(cam_layer)
        return RaddinoProbeCamModel(
            encoder.model,
            encoder.processor,
            probes,
            device=device,
            n_patches_each_dim=RADDINO_RESOLUTION // 14,
            block_index=block,
            findings_order=dataset.box_labels,
        )
    raise ValueError(f"unknown baseline {baseline!r}; expected 'txv' or 'raddino'")


def localize(
    baseline: str,
    dataset: TransferDataset,
    *,
    root: Path,
    splits_dir: Path,
    images: Path,
    segmentations: Path,
    device: str = "cuda:0",
    batch_size: int = 8,
    workers: int = 4,
    fit_workers: int = 1,
    limit: int | None = None,
    cam_layer: str = "last",
    probes: dict[str, tuple[np.ndarray, float]] | None = None,
    model=None,
) -> dict:
    """OneBox fitted on the target validation split through HiResCAM, test scored once."""
    from clarity.evaluation.loop import LOCALIZATION_TEST_SCORE_THRESHOLDS
    from clarity.localization.artifacts import localization_method_block
    from clarity.localization.evaluation import run_localization_evaluation_both

    if segmentations is None:
        raise SystemExit(
            "localization needs the anatomical segmentation directory "
            "(--segmentations; create it with clarity-segment / clarity-prepare)"
        )
    if model is None:
        model = build_cam_model(baseline, dataset, device=device, probes=probes, cam_layer=cam_layer)
    resolution = 224 if baseline == "txv" else 512  # RAD-DINO's processor resizes 512 -> 518
    model = model.to(th.device(device)).eval()
    val_loader, test_loader, val_names, test_names = _loaders(
        dataset,
        root=root,
        splits_dir=splits_dir,
        images=images,
        segmentations=segmentations,
        resolution=resolution,
        batch_size=batch_size,
        workers=workers,
        limit=limit,
    )
    print(f"{model.description}: val {len(val_names)}, test {len(test_names)} images at {resolution}px")
    kwargs = {"localization_method": "cam", "method": "hirescam"}
    val_family, calibration = run_localization_evaluation_both(
        model=model,
        dataloader=val_loader,
        device=device,
        max_workers=fit_workers,
        localization_calibration=None,
        bbox_method="onebox",
        selection="joint_argmax",
        use_amp=False,
        **kwargs,
    )
    test_output = run_localization_evaluation_both(
        model=model,
        dataloader=test_loader,
        device=device,
        max_workers=fit_workers,
        localization_calibration=calibration,
        bbox_method="onebox",
        selection=calibration.selection,
        score_thresholds=LOCALIZATION_TEST_SCORE_THRESHOLDS,
        return_per_instance=True,
        use_amp=False,
        **kwargs,
    )
    family, instances = test_output[0], test_output[2]
    iou, iobb = family["IoU"][0], family["IoBB"][0]
    keys = ("IoU_accuracy_macro", "IoU_afp_macro", "IoBB_accuracy_macro", "IoBB_afp_macro")
    print(f"  [test] IoU {iou['IoU_accuracy_macro']:.4f} | IoBB {iobb['IoBB_accuracy_macro']:.4f}")
    return {
        "arm": f"{model.description} (Target-calibrated OneBox)",
        "dataset": dataset.name,
        "resolution": resolution,
        "labels": list(dataset.box_labels),
        "split_coverage": f"val {len(val_names)} images, test {len(test_names)} images",
        "macro": {k: float(iou[k] if k.startswith("IoU") else iobb[k]) for k in keys},
        "macro_val": {k: float(val_family["IoU"][0][k]) for k in ("IoU_accuracy_macro", "IoBB_accuracy_macro")},
        "localization_method": localization_method_block(calibration),
        "test_instances": instances,
        "test_filenames": list(test_names),
        "limit": limit,
    }
