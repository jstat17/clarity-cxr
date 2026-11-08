import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    jaccard_score,
    hamming_loss,
    confusion_matrix,
)
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle as pk
import lzma
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from clarity.utils.runtime import get_autocast_context
from clarity.utils.types import contiguity_order
from clarity.utils.typing import Array1DLike, Matrix2DLike


#: per-abnormality metric names, in the order the report appends them
_PER_ABNORMALITY_KEYS = (
    "accuracy",
    "sensitivity",
    "precision",
    "specificity",
    "f1",
    "auc_roc",
    "ap",
    "cohen_kappa",
    "mcc",
    "jaccard_similarity",
    "hamming_loss",
)


def _get_first_tensor(output: tuple[Any] | list[Any] | th.Tensor) -> th.Tensor:
    """Retrieve the first tensor after unpacking nested lists/tuples."""
    if isinstance(output, (tuple, list)):
        return _get_first_tensor(output[0])
    return output


def classification_metrics_from_arrays(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probas: np.ndarray,
    split: str = "test",
    mean_loss: float | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Compute classification metrics from stacked prediction arrays.

    Shared by ``evaluate_model_classification`` and bootstrap/extraction paths
    so point estimates stay identical.
    """
    metrics: dict[str, float] = dict()

    # accuracy
    metrics["accuracy_sample"] = accuracy_score(all_labels, all_preds)
    metrics["accuracy_label"] = get_metric_macro(all_labels, all_preds, accuracy_score)

    # sensitivity
    metrics["sensitivity_macro"] = recall_score(all_labels, all_preds, average="macro")
    metrics["sensitivity_micro"] = recall_score(all_labels, all_preds, average="micro")

    # precision
    metrics["precision_macro"] = precision_score(all_labels, all_preds, average="macro")
    metrics["precision_micro"] = precision_score(all_labels, all_preds, average="micro")

    # specificity
    metrics["specificity_macro"] = get_metric_macro(all_labels, all_preds, specificity_score)
    metrics["specificity_micro"] = get_metric_micro(all_labels, all_preds, specificity_score)

    # F1
    metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
    metrics["f1_micro"] = f1_score(all_labels, all_preds, average="micro")

    # AUC ROC
    metrics["auc_roc_macro"] = get_metric_macro(all_labels, all_probas, roc_auc_score)
    metrics["auc_roc_micro"] = get_metric_micro(all_labels, all_probas, roc_auc_score)

    # AP / AUC Precision-Recall
    metrics["ap_macro"] = get_metric_macro(all_labels, all_probas, average_precision_score)
    metrics["ap_micro"] = get_metric_micro(all_labels, all_probas, average_precision_score)

    # Cohen's Kappa
    metrics["cohen_k_macro"] = get_metric_macro(all_labels, all_preds, cohen_kappa_score)
    metrics["cohen_k_micro"] = get_metric_micro(all_labels, all_preds, cohen_kappa_score)

    # Matthew's Correlation Coefficient
    metrics["mcc_macro"] = get_metric_macro(all_labels, all_preds, matthews_corrcoef)
    metrics["mcc_micro"] = get_metric_micro(all_labels, all_preds, matthews_corrcoef)

    # Jaccard similarity
    metrics["jacc_sim_macro"] = get_metric_macro(
        all_labels, all_preds, jaccard_score, average="binary"
    )
    metrics["jacc_sim_micro"] = get_metric_micro(
        all_labels, all_preds, jaccard_score, average="binary"
    )

    # hamming loss
    metrics["hamming_loss"] = get_metric_macro(all_labels, all_preds, hamming_loss)

    # evaluation loss (optional; omitted when criterion was not used)
    if mean_loss is not None:
        metrics[f"{split}_loss"] = mean_loss

    metrics_abnorm: dict[str, list[float]] = defaultdict(list)
    _nrows, ncols = all_preds.shape

    for j in range(ncols):
        y_proba = all_probas[:, j]
        y_pred = all_preds[:, j]
        y_label = all_labels[:, j]

        if np.unique(y_label).size < 2:
            # A class with a single ground-truth value makes AUC/specificity/
            # kappa undefined (e.g. VinDr Edema has no positive image in the
            # test or val split). Record NaN rather than a fake 0, and skip the
            # sklearn calls that would raise. The macro metrics above already
            # exclude zero-positive classes via ``get_metric_macro``.
            for key in _PER_ABNORMALITY_KEYS:
                metrics_abnorm[key].append(float("nan"))
            continue

        acc = accuracy_score(y_label, y_pred)
        sen = recall_score(y_label, y_pred)
        pre = precision_score(y_label, y_pred)
        spe = specificity_score(y_label, y_pred)
        f1 = f1_score(y_label, y_pred)

        auc = roc_auc_score(y_label, y_proba)
        ap = average_precision_score(y_label, y_proba)

        coh = cohen_kappa_score(y_label, y_pred)
        mcc = matthews_corrcoef(y_label, y_pred)
        jcc = jaccard_score(y_label, y_pred)
        hmm = hamming_loss(y_label, y_pred)

        metrics_abnorm["accuracy"].append(acc)
        metrics_abnorm["sensitivity"].append(sen)
        metrics_abnorm["precision"].append(pre)
        metrics_abnorm["specificity"].append(spe)
        metrics_abnorm["f1"].append(f1)

        metrics_abnorm["auc_roc"].append(auc)
        metrics_abnorm["ap"].append(ap)

        metrics_abnorm["cohen_kappa"].append(coh)
        metrics_abnorm["mcc"].append(mcc)
        metrics_abnorm["jaccard_similarity"].append(jcc)
        metrics_abnorm["hamming_loss"].append(hmm)

    return metrics, metrics_abnorm


def _is_filename_batch(item: Any) -> bool:
    """True if collated batch slot is a sequence of filename strings."""
    if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
        return True
    return False


def collect_classification_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    split: str = "test",
    pred_threshold: float = 0.5,
    criterion: nn.Module | None = None,
    collect_filenames: bool = False,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Run classification inference and return stacked arrays (no metric aggregation).

    Batch layouts (after ``custom_collate``):
      - ``(image, labels[, teacher_or_other tensors...])``
      - with filenames: string filenames may appear anywhere after the image
        (e.g. ``MultilabelDataset`` → index 2; a distillation dataset
        → last slot). Filenames are skipped when building criterion targets.

    When ``criterion`` is provided, loss is averaged over batches (same contract
    as the historical eval loop: all non-filename tensor slots after the image
    are passed as ``Y``, enabling student-teacher / distillation criteria).
    """
    all_probas = []
    all_preds = []
    all_labels = []
    all_losses: list[float] = []
    all_filenames: list[str] = []

    pbar = tqdm(
        iterable=loader,
        desc=f"Evaluating on {split} set",
        unit="batch",
    )

    model.eval()
    with th.no_grad():
        for batch in pbar:
            X = batch[0].to(device, non_blocking=True)

            # Separate criterion targets (tensors) from filename slots (str sequences).
            # Mirrors historical ``Y = batch[1:]; Y = [y.to(device) for y in Y]`` while
            # remaining robust when ``return_filename=True`` inserts non-tensors.
            tensor_targets: list[th.Tensor] = []
            filenames_batch = None
            for item in batch[1:]:
                if _is_filename_batch(item):
                    if collect_filenames:
                        if filenames_batch is not None:
                            raise RuntimeError(
                                "multiple filename slots found in classification batch"
                            )
                        filenames_batch = item
                    continue
                if th.is_tensor(item):
                    tensor_targets.append(item.to(device, non_blocking=True))

            if not tensor_targets:
                raise RuntimeError("classification batch has no tensor targets after the image")

            Y: th.Tensor | list[th.Tensor]
            if len(tensor_targets) == 1:
                Y = tensor_targets[0]
            else:
                Y = tensor_targets

            # Evaluation autocast covers the model forward and optional loss;
            # metric aggregation and host-side conversion remain outside it.
            with get_autocast_context(device, enabled=use_amp):
                Y_pred = model(X)

                if criterion is not None:
                    loss = criterion(Y_pred, Y)
                    all_losses.append(loss.item())

                labels: th.Tensor = _get_first_tensor(Y)
                logits_pred: th.Tensor = _get_first_tensor(Y_pred)
                probas = th.sigmoid(logits_pred)
                preds = (probas > pred_threshold).float()

            # NumPy does not support bfloat16 directly.
            probas = probas.float()
            preds = preds.float()

            all_probas.append(probas.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if collect_filenames:
                if filenames_batch is None:
                    raise RuntimeError(
                        "collect_filenames=True but no filename slot found in batch; "
                        "ensure dataset return_filename=True"
                    )
                all_filenames.extend([str(f) for f in filenames_batch])

    # Cast every tensor to float32 before crossing the NumPy boundary. This is
    # required for bfloat16 autocast outputs and is harmless for the normal
    # float32 path.
    all_probas_np = np.asfortranarray(th.cat(all_probas).float().numpy().astype(float))
    all_preds_np = np.asfortranarray(th.cat(all_preds).float().numpy().astype(int))
    all_labels_np = np.asfortranarray(th.cat(all_labels).float().numpy().astype(int))

    if getattr(loader.dataset, "use_background_label", False):
        all_probas_np = all_probas_np[:, 1:]
        all_preds_np = all_preds_np[:, 1:]
        all_labels_np = all_labels_np[:, 1:]

    mean_loss = (sum(all_losses) / len(all_losses)) if all_losses else None

    out: dict[str, Any] = {
        "labels": all_labels_np,
        "preds": all_preds_np,
        "probas": all_probas_np,
        "mean_loss": mean_loss,
    }
    if collect_filenames:
        if len(all_filenames) != all_labels_np.shape[0]:
            raise RuntimeError(
                f"filename count {len(all_filenames)} != n_images {all_labels_np.shape[0]}; "
                "ensure dataset return_filename=True and custom_collate"
            )
        out["filenames"] = all_filenames
    return out


def evaluate_model_classification(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    device: str,
    criterion: nn.Module | None,
    return_per_image: bool = False,
    pred_threshold: float = 0.5,
    use_amp: bool = False,
) -> (
    tuple[dict[str, float], dict[str, list[float]]]
    | tuple[dict[str, float], dict[str, list[float]], dict[str, Any]]
):
    """Evaluate classification metrics on a loader.

    Default return is the historical 2-tuple ``(metrics, metrics_abnorm)`` used
    by training and ``clarity-evaluate``. When ``return_per_image=True``,
    also returns a third dict with ``labels``, ``preds``, ``probas``, ``mean_loss``,
    and (if the dataset provided them) ``filenames``.

    ``criterion`` is always applied so ``{split}_loss`` and student-teacher /
    distillation losses stay available on both return paths.
    """
    collected = collect_classification_predictions(
        model=model,
        loader=loader,
        device=device,
        split=split,
        pred_threshold=pred_threshold,
        criterion=criterion,
        collect_filenames=return_per_image,
        use_amp=use_amp,
    )
    metrics, metrics_abnorm = classification_metrics_from_arrays(
        collected["labels"],
        collected["preds"],
        collected["probas"],
        split=split,
        mean_loss=collected["mean_loss"],
    )
    if not return_per_image:
        return metrics, metrics_abnorm

    per_image = {
        "labels": collected["labels"],
        "preds": collected["preds"],
        "probas": collected["probas"],
        "filenames": collected.get("filenames"),
        "mean_loss": collected["mean_loss"],
        "pred_threshold": pred_threshold,
    }
    return metrics, metrics_abnorm, per_image


def print_metrics(metrics: dict[str, float], cols: int = 3) -> None:
    keys = list(metrics.keys())
    values = list(metrics.values())

    for i in range(0, len(keys), cols):
        row_keys = keys[i : i + cols]
        row_values = values[i : i + cols]
        row = []
        for key, value in zip(row_keys, row_values):
            row.append(f"{key:<20} {value:.4f}")
        print(" | ".join(row))


def get_metric_macro(
    Y_true: Matrix2DLike, Y_pred: Matrix2DLike, metric: Callable, **kwargs
) -> float:
    """Get macro-averaged metric of multi-label predictions.

    Args:
        Y_true (Matrix2DLike): Ground truth binary labels, shape (num_samples, num_classes)
        Y_pred (Matrix2DLike): Predicted binary labels or probabilities,
            shape (num_samples, num_classes)
        metric (Callable): Single-class metric function

    Returns:
        float: Macro-averaged metric
    """
    values = []
    _nrows, ncols = Y_true.shape

    # loop over different classes (columns)
    for j in range(ncols):
        # only calculate metric if ground truth labels have 1s
        if np.sum(Y_true[:, j]) > 0:
            value = metric(Y_true[:, j], Y_pred[:, j], **kwargs)
            values.append(value)

    # calculate macro average
    value = np.mean(values)

    return value


def get_metric_micro(
    Y_true: Matrix2DLike, Y_pred: Matrix2DLike, metric: Callable, **kwargs
) -> float:
    """Get micro-averaged metric of multi-label predictions.

    Args:
        Y_true (Matrix2DLike): Ground truth binary labels, shape (num_samples, num_classes)
        Y_pred (Matrix2DLike): Predicted binary labels or probabilities,
            shape (num_samples, num_classes)
        metric (Callable): Single-class metric function

    Returns:
        float: Micro-averaged metric
    """
    # use contiguity order of labels to flatten
    order = contiguity_order(Y_true)

    # flatten matrices using same order
    Y_true_flat = Y_true.flatten(order=order)
    Y_pred_flat = Y_pred.flatten(order=order)

    # calculate metric value
    value = metric(Y_true_flat, Y_pred_flat, **kwargs)

    return value


def specificity_score(y_true: Array1DLike, y_pred: Array1DLike) -> float:
    """Compute specificity from true and predicted binary labels.

    Args:
        y_true (Array1DLike): True binary labels.
        y_pred (Array1DLike): Predicted binary labels.

    Returns:
        float: Specificity value.

    ``labels=[0, 1]`` is passed explicitly: ``confusion_matrix`` only emits the
    labels it observes, so a class whose truth *and* prediction are constant —
    VinDr ``Edema`` has no positive test image, and an arm that also predicts no
    positive returns a 1x1 matrix — would make the four-way unpack raise
    ``ValueError``. The explicit label set keeps the matrix 2x2 (and makes
    ``tn / (tn + fp)`` the same value it always was for non-degenerate input).
    """
    # calculate confusion matrix
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # avoid division by zero

    return specificity


def get_run_score(run_folder: Path) -> float:
    # get log path
    run_log_path = run_folder / "log.pk.xz"

    # load run log
    with lzma.open(run_log_path, "rb") as f:
        log_file = pk.load(f)

    # obtain score
    score = log_file["score"]

    return score
