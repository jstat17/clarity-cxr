"""Persistence for the frozen localization calibration artifact.

Schema version 1 is the ``localization_method`` block carried by every new
evaluation pickle.  Archived pickles have no block and are inferred as the
legacy SingleCut + ``mean_argmax`` method by the readers.  The on-disk key
names are frozen; do not rename them.
"""

from __future__ import annotations

from typing import Any, Mapping

import json
from pathlib import Path

import numpy as np

from clarity.localization.fitting import LocalizationCalibration
from clarity.localization.onebox import OneBoxParams, SingleCutParams


def _calibration_thresholds(calibration) -> dict[int, float] | None:
    """Legacy ``dict[int, float]`` thresholds from a polymorphic artifact.

    The 4th evaluator tuple slot is either a legacy ``heatmap_thr`` dict or a
    :class:`~localization.fitting.LocalizationCalibration`.  OneBox
    calibrations have no scalar thresholds, so the returned dict is empty.
    """
    if calibration is None:
        return None
    if isinstance(calibration, LocalizationCalibration):
        out: dict[int, float] = {}
        for ab, params in calibration.params.items():
            if isinstance(params, SingleCutParams):
                out[int(ab)] = float(params.threshold)
        return out
    return {int(ab): float(t) for ab, t in calibration.items()}


def _params_representative_threshold(params) -> float:
    """Scalar heatmap threshold stored in the historical summary table.

    SingleCut uses its absolute ``threshold``; OneBox uses ``t_hi`` (the
    documented representative level; the full OneBox vector lives in the
    calibration artifact).
    """
    if isinstance(params, OneBoxParams):
        return float(params.t_hi)
    return float(params.threshold)


def _params_to_dict(params) -> dict:
    """JSON/pickle-friendly parameter block for per-instance records.

    OneBox persists the pixel sigma as ``sigma_px`` plus the relative
    sigma; SingleCut persists its absolute ``threshold``.
    """
    if isinstance(params, OneBoxParams):
        return {
            "sigma_px": float(params.sigma),
            "t_lo": float(params.t_lo),
            "t_hi": float(params.t_hi),
            "s": float(params.s),
            "alpha": float(params.alpha),
            "min_area": int(params.min_area),
            "sigma_rel": (None if params.sigma_rel is None else float(params.sigma_rel)),
        }
    return {
        "threshold": float(params.threshold),
        "sigma": float(params.sigma),
        "alpha": float(params.alpha),
        "min_area": int(params.min_area),
    }


def _params_from_dict(vector: Mapping[str, Any], bbox_method: str):
    """Inverse of :func:`_params_to_dict` for persisted artifacts."""
    if bbox_method == "onebox":
        sigma = vector.get("sigma_px", vector.get("sigma"))
        return OneBoxParams(
            sigma=float(sigma if sigma is not None else 0.0),
            t_lo=float(vector["t_lo"]),
            t_hi=float(vector["t_hi"]),
            s=float(vector["s"]),
            alpha=float(vector.get("alpha", 1.0)),
            min_area=int(vector.get("min_area", 4)),
            sigma_rel=(None if vector.get("sigma_rel") is None else float(vector["sigma_rel"])),
        )
    return SingleCutParams(
        threshold=float(vector["threshold"]),
        sigma=float(vector.get("sigma", 0.0)),
        alpha=float(vector.get("alpha", 1.0)),
        min_area=int(vector.get("min_area", 0)),
    )


LOCALIZATION_METHOD_SCHEMA_VERSION = 1


def _jsonable(value):
    """Recursively convert numpy scalars/arrays into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def calibration_thresholds(calibration):
    """Public alias for the polymorphic 4th-slot threshold dispatch.

    Returns ``dict[int, float]`` for a legacy ``heatmap_thr`` dict or a
    SingleCut :class:`~localization.fitting.LocalizationCalibration`,
    ``None`` for ``None``, and ``{}`` for a OneBox calibration (no scalar
    threshold exists).
    """
    return _calibration_thresholds(calibration)


def localization_method_block(calibration) -> dict:
    """Normalize a 4th-slot artifact into the persisted schema-1 block.

    ``calibration`` is a
    :class:`~localization.fitting.LocalizationCalibration` for new runs
    (both methods) or a legacy ``dict[int, float]``/``None``.  Legacy inputs
    are recorded as an inferred SingleCut + ``mean_argmax`` artifact so every
    new pickle carries a self-describing block; archived pickles without a
    block are inferred by the readers instead.
    """
    if isinstance(calibration, LocalizationCalibration):
        params = {str(int(ab)): _params_to_dict(p) for ab, p in calibration.params.items()}
        params_iobb = None
        if calibration.params_iobb is not None:
            params_iobb = {
                str(int(ab)): _params_to_dict(p) for ab, p in calibration.params_iobb.items()
            }
        return {
            "schema_version": LOCALIZATION_METHOD_SCHEMA_VERSION,
            "bbox_method": calibration.bbox_method,
            "selection": calibration.selection,
            "selection_metric": calibration.selection_metric,
            "optimize_metrics_separately": bool(calibration.optimize_metrics_separately),
            "image_size": int(calibration.image_size),
            "search_space": _jsonable(calibration.search_space),
            "params": params,
            "params_iobb": params_iobb,
            "val5": {str(k): float(v) for k, v in (calibration.val5 or {}).items()},
            "fit_seconds": float(calibration.fit_seconds),
            "fit_workers": int(calibration.fit_workers),
        }

    # Legacy dict / None: record an explicit inferred SingleCut artifact.
    block: dict[str, Any] = {
        "schema_version": 0,
        "bbox_method": "singlecut",
        "selection": "mean_argmax",
        "selection_metric": "IoU",
        "optimize_metrics_separately": False,
        "image_size": 0,
        "search_space": {},
        "params": {},
        "params_iobb": None,
        "val5": {},
        "fit_seconds": 0.0,
        "fit_workers": 0,
        "inferred_legacy": True,
    }
    if calibration is not None:
        block["params"] = {
            str(int(ab)): _params_to_dict(SingleCutParams(float(t)))
            for ab, t in calibration.items()
        }
    return block


def localization_calibration_from_block(block: Mapping[str, Any]) -> LocalizationCalibration:
    """Rebuild a :class:`LocalizationCalibration` from a schema block.

    Used by artifact readers/extractors that must hand a frozen calibration
    back to the evaluator without re-fitting.
    """
    bbox_method = str(block.get("bbox_method") or "singlecut")
    if bbox_method not in ("singlecut", "onebox"):
        raise ValueError(f"unknown bbox_method {bbox_method!r}")
    params = {
        int(ab): _params_from_dict(vector, bbox_method)
        for ab, vector in (block.get("params") or {}).items()
    }
    raw_iobb = block.get("params_iobb")
    params_iobb = None
    if raw_iobb:
        params_iobb = {
            int(ab): _params_from_dict(vector, bbox_method) for ab, vector in raw_iobb.items()
        }
    selection = block.get("selection") or (
        "joint_argmax" if bbox_method == "onebox" else "mean_argmax"
    )
    return LocalizationCalibration(
        bbox_method=bbox_method,
        selection=selection,
        selection_metric=str(block.get("selection_metric") or "IoU"),
        params=params,
        params_iobb=params_iobb,
        image_size=int(block.get("image_size") or 0),
        search_space=dict(block.get("search_space") or {}),
        val5=dict(block.get("val5") or {}),
        fit_seconds=float(block.get("fit_seconds") or 0.0),
        fit_workers=int(block.get("fit_workers") or 0),
        optimize_metrics_separately=bool(block.get("optimize_metrics_separately", False)),
    )


def load_frozen_calibration(path) -> dict | None:
    """Read a ported schema-1 OneBox block (arm A1, NIH-calibrated).

    The transfer-study calibration files record the parameters twice: under
    ``params`` keyed by the *source* (NIH) class index, which is the provenance
    record, and under ``target_label_map`` keyed by the *evaluation* label index.
    A model instantiated with ``--num_classes 14 -mkw 'logit_mask=[12]'`` emits
    maps indexed by masked position, so pneumonia comes back as index 0 while the
    RSNA dataset's own label space also uses 0 — the two agree here, but the
    translation is explicit so it cannot be guessed wrong for a multi-finding
    target such as VinDr.

    A bare ``localization_method`` block (no ``target_label_map``) is passed
    through unchanged: those blocks are already keyed by the label space they were
    fitted in.
    """
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    if "bbox_method" not in payload and "localization_method" in payload:
        payload = dict(payload["localization_method"])

    target_map = payload.pop("target_label_map", None)
    if "target" in payload:
        payload.pop("target")
    if target_map is not None:
        payload["params"] = {str(index): dict(vec) for index, vec in target_map.items()}
        payload["params_iobb"] = None

    if payload.get("bbox_method") != "onebox":
        raise ValueError(
            f"{path}: expected bbox_method 'onebox', got {payload.get('bbox_method')!r}"
        )
    return payload
