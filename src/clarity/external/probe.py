"""Per-finding linear probe on frozen RAD-DINO features.

This is the only *fitted* part of the RAD-DINO row, so it owns the leakage surface:

* it may be fitted on the **train** list only (``fit`` raises when a fitted
  filename is outside ``train_filenames``);
* its regularisation is selected on **val**, never on test (``chosen_c`` is the
  arg-max of the val AUC over the grid, and nothing here touches test arrays);
* the metric is the study's own — ``study_metrics`` over the evaluable findings —
  so the row cannot quietly average over a different denominator than the CLARiTy
  arms and the TXV rows (AUROC only, no AUPRC column).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from clarity.data.vindr import VINDR_DISEASES
from clarity.external.evaluate import study_metrics
from clarity.external.raddino import RADDINO_MODEL_ID, RADDINO_RESOLUTION

#: the regularisation grid; selected on val, one value per finding
RADDINO_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
#: the study's own probe recipe, recorded in the arm artifact
RADDINO_PROBE_MAX_ITER = 1000

_METRIC_KEYS = ("auc", "ap")


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUC, or None when the labels have a single class (undefined, not zero)."""
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


class RaddinoProbe:
    """One finding's L2 logistic regression with C selected on the val split."""

    def __init__(
        self,
        train_filenames: Sequence[str],
        c_grid: Sequence[float] = RADDINO_C_GRID,
        max_iter: int = RADDINO_PROBE_MAX_ITER,
    ) -> None:
        self.train_filenames = [str(name) for name in train_filenames]
        self.c_grid = tuple(float(c) for c in c_grid)
        self.max_iter = int(max_iter)
        self.chosen_c: float | None = None
        self.val_auc_by_c: dict[float, float | None] = {}
        self._classifier: LogisticRegression | None = None

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        filenames: Sequence[str],
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> RaddinoProbe:
        """Fit on train, select C on val. Any filename outside the train list fails."""
        names = [str(name) for name in filenames]
        allowed = set(self.train_filenames)
        outside = [name for name in names if name not in allowed]
        if outside:
            raise ValueError(
                f"{len(outside)} fitted filename(s) are outside the probe's train list "
                f"(e.g. {outside[0]!r}): the probe may only ever be fitted on the "
                f"train split"
            )

        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y).astype(int).ravel()
        val_x = np.asarray(val_x, dtype=float)
        val_y = np.asarray(val_y).astype(int).ravel()
        if len(names) != len(train_x):
            raise ValueError(
                f"{len(names)} filenames for {len(train_x)} feature rows"
            )

        best_c: float | None = None
        best_auc: float | None = None
        best_classifier: LogisticRegression | None = None
        for c in self.c_grid:
            classifier = LogisticRegression(C=c, max_iter=self.max_iter).fit(
                train_x, train_y
            )
            auc = _safe_auc(val_y, classifier.predict_proba(val_x)[:, 1])
            self.val_auc_by_c[c] = auc
            # strict '>' keeps the first C of the grid on a tie, the same rule as
            # `max(val_aucs, key=val_aucs.get)` recomputed by the contract test
            if auc is not None and (best_auc is None or auc > best_auc):
                best_c, best_auc, best_classifier = c, auc, classifier

        if best_classifier is None:
            # a degenerate val split (one class) carries no selection signal:
            # fall back to the weakest regularisation rather than inventing one
            best_c = self.c_grid[0]
            best_classifier = LogisticRegression(
                C=best_c, max_iter=self.max_iter
            ).fit(train_x, train_y)

        self.chosen_c = float(best_c)
        self._classifier = best_classifier
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """P(positive) for every row, from the val-selected classifier."""
        if self._classifier is None:
            raise RuntimeError("the probe is not fitted yet; call fit() first")
        return self._classifier.predict_proba(np.asarray(features, dtype=float))[:, 1]

    def test_auc(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> float | None:
        """Test AUC at the val-selected C (refit on train; never re-selects)."""
        if self.chosen_c is None:
            raise RuntimeError("the probe is not fitted yet; call fit() first")
        classifier = LogisticRegression(C=self.chosen_c, max_iter=self.max_iter).fit(
            np.asarray(train_x, dtype=float), np.asarray(train_y).astype(int).ravel()
        )
        return _safe_auc(
            test_y, classifier.predict_proba(np.asarray(test_x, dtype=float))[:, 1]
        )


def _finding_order(probes: dict[str, RaddinoProbe]) -> list[str]:
    unknown = sorted(set(probes) - set(VINDR_DISEASES))
    if unknown:
        raise ValueError(f"unknown VinDr finding(s) in the probe set: {unknown}")
    return [finding for finding in VINDR_DISEASES if finding in probes]


def evaluate_raddino_probe(
    probes: dict[str, RaddinoProbe],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Score the fitted probes on one split with the study's metric.

    `labels` is the usual ``(N, 11)`` matrix in ``VINDR_DISEASES`` order, or an
    ``(N, len(probes))`` matrix whose columns follow the probe order for a single
    finding. Only the findings that have a probe (10 of 11 — ``Edema`` is skipped)
    enter the metric, so the macro is over the same denominator as the other rows.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    findings = _finding_order(probes)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2-D label matrix, got {labels.shape}")
    if len(labels) != len(features):
        raise ValueError(f"{len(labels)} label rows for {len(features)} feature rows")
    if labels.shape[1] == len(VINDR_DISEASES):
        selected = np.stack(
            [labels[:, VINDR_DISEASES.index(finding)] for finding in findings], axis=1
        )
    elif labels.shape[1] == len(findings):
        selected = labels
    else:
        raise ValueError(
            f"label matrix has {labels.shape[1]} columns; expected "
            f"{len(VINDR_DISEASES)} (all findings) or {len(findings)} "
            f"(the probed findings, in probe order)"
        )

    probas = np.stack(
        [probes[finding].predict_proba(features) for finding in findings], axis=1
    )

    document = study_metrics(selected, probas, threshold=threshold, findings=findings)
    document["probas"] = {
        finding: probas[:, index] for index, finding in enumerate(findings)
    }
    document["probe_selection"] = {
        finding: {
            "chosen_c": probes[finding].chosen_c,
            "c_grid": list(probes[finding].c_grid),
            "val_auc_by_c": {
                f"{c:g}": value for c, value in probes[finding].val_auc_by_c.items()
            },
        }
        for finding in findings
    }
    return document


def _feature_cache_sha256(document: dict) -> str | None:
    cache = (document.get("provenance") or {}).get("feature_cache_sha256")
    return str(cache) if cache else None


def _empty_metric_row() -> dict[str, float | None]:
    return {key: None for key in _METRIC_KEYS}


def build_raddino_arm_document(
    *,
    evaluation: dict,
    evaluation_val: dict,
    split_sha256: str,
    arm_label: str,
    revision: str,
) -> dict:
    """Shape the probe row like every other arm result document.

    The 10 probed findings are expanded back to the 11-finding layout the table
    renderer indexes by name, with ``Edema`` explicitly ``n/a`` — the same shape the
    TXV rows emit, so one renderer prints it unchanged.
    """
    probed = dict(evaluation.get("per_finding") or {})
    per_finding_by_name: dict[str, dict[str, float | None]] = {}
    for finding in VINDR_DISEASES:
        row = probed.get(finding)
        per_finding_by_name[finding] = (
            {key: row.get(key) for key in _METRIC_KEYS} if row else _empty_metric_row()
        )

    selection = evaluation.get("probe_selection") or {}
    c_grid = sorted(
        {float(c) for entry in selection.values() for c in entry.get("c_grid") or []}
    )
    document = {
        "schema_version": 1,
        "arm": arm_label,
        "resolution": int(RADDINO_RESOLUTION),
        "run": f"raddino:{RADDINO_RESOLUTION}",
        "split_coverage": "val {} images, test {} images".format(
            evaluation_val.get("n_images", "?"), evaluation.get("n_images", "?")
        ),
        "classification": {
            "pickle": f"hf:{RADDINO_MODEL_ID}#{revision}",
            "sha256": _feature_cache_sha256(evaluation) or revision,
            "labels": list(VINDR_DISEASES),
            "test": {
                "macro": dict(evaluation["macro"]),
                "per_finding": {
                    key: [per_finding_by_name[finding][key] for finding in VINDR_DISEASES]
                    for key in _METRIC_KEYS
                },
                "per_finding_by_name": per_finding_by_name,
                "n_evaluable": evaluation["n_evaluable"],
                "evaluable_labels": evaluation.get("evaluable_labels"),
                "n_images": evaluation.get("n_images"),
            },
            "val": {"macro": dict(evaluation_val["macro"])},
        },
        "provenance": {
            "model": RADDINO_MODEL_ID,
            "revision": str(revision),
            "weights_url": f"https://huggingface.co/{RADDINO_MODEL_ID}/tree/{revision}",
            "training_corpus": (
                "MIMIC-CXR + (per the paper) additional public chest X-ray corpora, "
                "self-supervised; no VinDr-CXR"
            ),
            "training_corpus_includes_target": False,
            "supervision": (
                "frozen encoder + per-finding linear probe fitted on the VinDr train list"
            ),
            "supervision_note": (
                "the encoder is pretrained without VinDr; only the logistic probe sees "
                "VinDr labels, and it is fitted on the committed train list with C "
                "selected on val, so this is a *transfer* row, not a zero-shot one"
            ),
            "citation": (
                "Pérez-García et al., 'Rad-DINO: Exploring Scalable Medical Image "
                "Encoders Beyond Text Supervision', 2024 (arXiv 2401.10815)"
            ),
            "input_resolution": int(RADDINO_RESOLUTION),
            "probe": {
                "c_grid": c_grid,
                "chosen_c_by_finding": {
                    finding: entry.get("chosen_c") for finding, entry in selection.items()
                },
                "max_iter": RADDINO_PROBE_MAX_ITER,
                "n_train": evaluation.get("n_train_filenames"),
                "n_val": evaluation_val.get("n_images"),
            },
            "recipe": (
                "frozen RAD-DINO encoder at 518^2 through the checkpoint's own HF "
                "processor (no hand-written resize/mean/std), CLS token of the last "
                "hidden state; per-finding LogisticRegression fitted on the VinDr "
                "train list with C selected on val; scored with the study's "
                "classification_metrics_from_arrays at threshold 0.5, AUROC only"
            ),
            "split": "test",
            "split_sha256": split_sha256,
            "feature_cache_sha256": _feature_cache_sha256(evaluation),
        },
    }
    return document


def feature_cache_digest(path) -> str:
    """sha256 of a feature cache, so a cached run is distinguishable from a rerun."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
