"""Localization method/selection abstraction, val fitting and calibration.

This module is pure CPU (numpy + the ported
:mod:`localization.onebox`); it never imports torch, a model or a dataloader.  It owns:

* :class:`LocalizationMethodSpec` — the two-axis method/selection configuration
  (``singlecut`` / ``onebox`` × ``joint_argmax`` / ``mean_argmax``), the
  default-resolution rules and the legacy ``optimize_metrics_separately``
  guard;
* :class:`OneBoxSearchSpace` — the exact 0.05 grid (5 sigma × 171 pairs ×
  6 s × 3 alpha = 15 390 configs) with the relative sigma grid;
* :func:`singlecut_valstats_from_raw_tables` — adapt per-instance 91-threshold detection tables into the ``ValStats``
  shape so the same selectors work for both methods;
* :class:`OneBoxFitter` / :class:`OneBoxFitPool` — streamed val5 fitting over
  chunks of instances using the batched accumulator, with a fork process
  pool and deterministic, chunk-ordered merging;
* :func:`select_singlecut` / :func:`select_onebox` — the selection rules;
  the OneBox objective is always val5 IoU, IoBB is never selected;
* :func:`score_fixed_params` / :func:`boxes_at_fixed_params` — fixed-parameter
  scoring for the test pass (no threshold scan, no fitting);
* :class:`LocalizationCalibration` — the frozen artifact persisted next to the
  historical ``heatmap_thr`` block.

The OneBox grids, gate and selection bodies come from
:mod:`localization.onebox`; the SingleCut adapter reproduces the legacy
best-threshold reduction (including its first-max tie rule and 2-dp rounding).  See docs/onebox.md.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from clarity.localization.onebox import (
    GATE_AREA,
    ONEBOX_ALPHA_GRID,
    ONEBOX_S_GRID,
    SCORES,
    SIGMA_REL_MAX,
    SIGMA_REL_REF,
    THR01,
    THR05,
    Instance,
    InstanceCache,
    JointArgmax,
    MeanArgmax,
    OneBoxParams,
    Params,
    SingleCutParams,
    ValStats,
    boxes_for_with_scores,
    boxes_for,
    build_batch_plan,
    build_onebox_plan,
    det_metrics,
    det_metrics_iobb,
    onebox_configs,
    onebox_partial_job,
    onebox_worker_init,
    relative_sigma_grid,
    threshold_pairs,
    warmup_onebox,
)

BBoxMethod = Literal["singlecut", "onebox"]
SelectionStrategy = Literal["joint_argmax", "mean_argmax"]

#: default chunk size for the streamed fit (16-32 instances per finding)
DEFAULT_FIT_CHUNK = 32


# ---------------------------------------------------------------------------
# 1. method spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizationMethodSpec:
    """The two-axis localization method/selection configuration.

    ``selection=None`` resolves to the method default: ``joint_argmax`` for
    OneBox, ``mean_argmax`` for the legacy SingleCut path.

    ``optimize_metrics_separately=True`` restores the legacy per-metric
    two-call sequence and is only valid for ``singlecut`` + ``mean_argmax``: there is no reference IoBB objective for OneBox or for
    ``joint_argmax``.  :meth:`resolved` raises ``NotImplementedError`` for any
    other combination.
    """

    bbox_method: BBoxMethod = "onebox"
    selection: SelectionStrategy | None = None
    image_size: int | None = None
    optimize_metrics_separately: bool = False

    def __post_init__(self) -> None:
        if self.bbox_method not in ("singlecut", "onebox"):
            raise ValueError(
                f"unknown bbox_method {self.bbox_method!r}; " "expected 'singlecut' or 'onebox'"
            )
        if self.selection not in (None, "joint_argmax", "mean_argmax"):
            raise ValueError(
                f"unknown selection {self.selection!r}; expected None, "
                "'joint_argmax' or 'mean_argmax'"
            )

    def resolved(self) -> "LocalizationMethodSpec":
        """Fill in the method-default selection and validate the legacy flag."""
        if self.bbox_method == "onebox":
            selection: SelectionStrategy = self.selection or "joint_argmax"
        else:  # singlecut: legacy default
            selection = self.selection or "mean_argmax"
        if self.optimize_metrics_separately and not (
            self.bbox_method == "singlecut" and selection == "mean_argmax"
        ):
            raise NotImplementedError(
                "optimize_metrics_separately=True is only valid for "
                "singlecut + mean_argmax (the default "
                "evaluation optimizes val5 IoU once and there is no reference "
                f"IoBB objective); got {self.bbox_method} + {selection}"
            )
        return replace(self, selection=selection)


def resolve_method_spec(
    bbox_method: BBoxMethod | None = "onebox",
    selection: SelectionStrategy | None = None,
    image_size: int | None = None,
    optimize_metrics_separately: bool = False,
) -> LocalizationMethodSpec:
    """Build a resolved spec from evaluator-style kwargs.

    The default is OneBox + ``joint_argmax``.  An explicit
    ``bbox_method=None`` is the legacy alias (``singlecut`` + ``mean_argmax``);
    it is never a default.
    """
    if bbox_method is None:
        bbox_method = "singlecut"
        if selection is None:
            selection = "mean_argmax"
    return LocalizationMethodSpec(
        bbox_method=bbox_method,
        selection=selection,
        image_size=image_size,
        optimize_metrics_separately=optimize_metrics_separately,
    ).resolved()


# ---------------------------------------------------------------------------
# 2. OneBox config-space builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OneBoxSearchSpace:
    """The frozen OneBox search space.

    Defaults produce 15 390 configs at any image size:
    ``5 sigma × 171 pairs × 6 s × 3 alpha``.  ``thresholds`` is the 19-level
    0.05 grid (floor 0.05); ``sigmas(size)`` returns
    ``k / 224 * size`` for ``k = 0..sigma_rel_max``.
    """

    thresholds: np.ndarray = field(default_factory=lambda: THR05)
    sigma_rel_max: float = SIGMA_REL_MAX
    s_grid: tuple[float, ...] = ONEBOX_S_GRID
    alpha_grid: tuple[float, ...] = ONEBOX_ALPHA_GRID
    min_area: int = GATE_AREA

    def __post_init__(self) -> None:
        thresholds = np.ascontiguousarray(self.thresholds, dtype=np.float64)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "s_grid", tuple(float(v) for v in self.s_grid))
        object.__setattr__(self, "alpha_grid", tuple(float(v) for v in self.alpha_grid))

    @property
    def pairs(self) -> tuple[tuple[float, float], ...]:
        """All ``(t_lo, t_hi)`` levels with ``t_lo < t_hi`` (171 by default)."""
        return tuple(threshold_pairs(self.thresholds))

    def sigmas(self, size: int) -> list[float]:
        return relative_sigma_grid(int(size), self.sigma_rel_max, SIGMA_REL_REF)

    def configs(self, size: int) -> list[OneBoxParams]:
        return onebox_configs(
            self.sigmas(size),
            self.thresholds,
            s_grid=self.s_grid,
            alpha_grid=self.alpha_grid,
            min_area=self.min_area,
            pairs=self.pairs,
        )

    def summary(self, size: int) -> dict[str, Any]:
        """Machine-readable description for the calibration artifact."""
        return {
            "thresholds": [float(v) for v in self.thresholds],
            "sigma_parameterization": "sigma_rel",
            "sigma_rel_ref": float(SIGMA_REL_REF),
            "sigma_rel_max": float(self.sigma_rel_max),
            "sigmas": [float(v) for v in self.sigmas(size)],
            "s_grid": [float(v) for v in self.s_grid],
            "alpha_grid": [float(v) for v in self.alpha_grid],
            "min_area": int(self.min_area),
            "n_pairs": len(self.pairs),
            "n_configs": len(self.configs(size)),
        }


# ---------------------------------------------------------------------------
# 3. SingleCut ValStats from the standard raw tables
# ---------------------------------------------------------------------------
def singlecut_valstats_from_raw_tables(
    raw_results: Sequence[Mapping[str, Any]],
    all_abnorm_idxs: Iterable[int],
    score_thresholds: Iterable[float],
    metric: Literal["IoU", "IoBB"] = "IoU",
) -> ValStats:
    """Build ``ValStats`` from per-instance raw detection tables.

    ``raw_results`` is the list of per-instance detection-table dicts
    (``{'abnorm_idx': int, score_thr: DataFrame}``); each DataFrame has the
    91 ``THRESHOLDS_HEATMAP`` rows and the four count columns.  Per
    ``(abnorm, heatmap threshold t)``::

        sum_det[cfg(t), ab, j]    = sum_i 1[num_detected_i[t, j] > 0]
        sum_missed[cfg(t), ab, j] = sum_i num_missed_i[t, j]
        sum_boxes[cfg(t), ab]     = sum_i (num_detected_i[t, 0] + num_missed_i[t, 0])

    ``metric`` selects the IoU (default) or IoBB count columns; the standard
    ``sum_boxes`` identity is metric-independent, so ``j = 0`` is used once per
    instance.  Fingerprint caveat: an empty map is stored as
    ``detected=0, missed=1`` (one box), while the reference
    ``_accumulate_singlecut`` adds nothing to ``sum_boxes`` for empty maps;
    ``sum_det``/``sum_missed`` are identical in both.  This only affects the
    fewer-boxes tie-break of ``joint_argmax``.

    ``all_abnorm_idxs`` is the caller's evaluated-finding set; findings with
    no raw results are excluded (they cannot be scored), and raw results for
    findings outside the set raise ``ValueError``.
    """
    score_thresholds = list(score_thresholds)
    scores = np.asarray(score_thresholds, dtype=np.float64)
    if metric == "IoU":
        det_col, missed_col = "IoU_NumDetected", "IoU_NumMissed"
    elif metric == "IoBB":
        det_col, missed_col = "IoBB_NumDetected", "IoBB_NumMissed"
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'IoU' or 'IoBB'")

    findings_present = sorted({int(r["abnorm_idx"]) for r in raw_results})
    allowed = {int(a) for a in all_abnorm_idxs}
    missing = [ab for ab in findings_present if ab not in allowed]
    if missing:
        raise ValueError(
            f"raw_results contain abnormality indices absent from " f"all_abnorm_idxs: {missing}"
        )

    configs = [SingleCutParams(float(t), 0.0, 1.0, 0) for t in THR01]
    findings = findings_present
    pos = {ab: i for i, ab in enumerate(findings)}
    n_cfg = len(configs)
    n_score = len(score_thresholds)
    sum_det = np.zeros((n_cfg, len(findings), n_score), np.float64)
    sum_missed = np.zeros_like(sum_det)
    sum_boxes = np.zeros((n_cfg, len(findings)), np.float64)
    n_inst = {ab: 0 for ab in findings}

    for result in raw_results:
        ab = int(result["abnorm_idx"])
        fi = pos[ab]
        n_inst[ab] += 1
        for j, score_thr in enumerate(score_thresholds):
            df = result[score_thr]
            thr = np.asarray(df["HeatmapThreshold"], dtype=np.float64)
            det = np.asarray(df[det_col], dtype=np.int64)
            missed = np.asarray(df[missed_col], dtype=np.int64)
            # map each row onto its index in the standard 91-level grid
            ks = np.argmin(np.abs(THR01[None, :] - thr[:, None]), axis=1)
            np.add.at(sum_det[:, fi, j], ks, (det > 0).astype(np.float64))
            np.add.at(sum_missed[:, fi, j], ks, missed.astype(np.float64))
            if j == 0:
                # one box count per instance, independent of the score level
                np.add.at(sum_boxes[:, fi], ks, (det + missed).astype(np.float64))

    return ValStats(
        method="singlecut",
        configs=configs,
        findings=findings,
        scores=scores,
        sum_det=sum_det,
        sum_missed=sum_missed,
        sum_boxes=sum_boxes,
        n_inst=n_inst,
    )


def assert_singlecut_mean_matches_legacy(
    stats: ValStats,
    legacy_heatmap_thr: Mapping[int, float],
) -> dict[int, SingleCutParams]:
    """Assert ``MeanArgmax(digits=2)`` reproduces the legacy thresholds.

    Raises ``AssertionError`` listing the mismatching findings; returns the
    selected params so a later pass can reuse the selection.  ``legacy_heatmap_thr``
    is the legacy best-threshold average (first-max ties, averaged,
    rounded to 2 dp).
    """
    selected = MeanArgmax(thresholds=THR01, digits=2).select(stats)
    mismatches = []
    for ab, expected in legacy_heatmap_thr.items():
        got = selected.get(int(ab))
        expected_f = float(expected)
        if got is None or not np.isclose(got.threshold, expected_f, rtol=0.0, atol=1e-9):
            mismatches.append((int(ab), None if got is None else got.threshold, expected_f))
    if mismatches:
        raise AssertionError(
            "MeanArgmax(digits=2) does not reproduce the legacy "
            f"average_best_heatmap_thr; (abnorm, got, expected) = {mismatches}"
        )
    return selected


# ---------------------------------------------------------------------------
# 4. selection
# ---------------------------------------------------------------------------
def select_singlecut(
    raw_results: Sequence[Mapping[str, Any]],
    all_abnorm_idxs: Iterable[int],
    score_thresholds: Iterable[float],
    spec: LocalizationMethodSpec,
    metric: Literal["IoU", "IoBB"] = "IoU",
) -> dict[int, SingleCutParams]:
    """Select SingleCut params from the standard raw tables.

    * ``mean_argmax`` (legacy) supports the metric-specific reduction: the
      per-score argmax config is chosen from the requested family's
      per-instance detection bits (``metric='IoBB'`` is the historical
      standard objective).
    * ``joint_argmax`` is always IoU-based; requesting
      ``metric='IoBB'`` raises ``NotImplementedError``.
    * ``optimize_metrics_separately=True`` is validated by
      :meth:`LocalizationMethodSpec.resolved` (only
      ``singlecut`` + ``mean_argmax``).
    """
    resolved = spec.resolved()
    if resolved.bbox_method != "singlecut":
        raise ValueError(
            f"select_singlecut requires bbox_method='singlecut'; " f"got {resolved.bbox_method!r}"
        )
    if resolved.selection == "joint_argmax" and metric != "IoU":
        raise NotImplementedError(
            "singlecut + joint_argmax selects on val5 IoU only "
            "(there is no reference IoBB objective); "
            f"got metric={metric!r}"
        )
    stats = singlecut_valstats_from_raw_tables(
        raw_results, all_abnorm_idxs, score_thresholds, metric=metric
    )
    if resolved.selection == "joint_argmax":
        return JointArgmax().select(stats)
    return MeanArgmax(thresholds=THR01, digits=2).select(stats)


def select_onebox(
    stats: ValStats,
    stats_iobb: ValStats,
    spec: LocalizationMethodSpec,
) -> dict[int, OneBoxParams]:
    """Select OneBox params from the fitted val5 sums.

    The objective is always the val5 IoU criterion: ``stats`` is the
    IoU family and ``stats_iobb`` the parallel IoBB family used later for the
    reported tables.  ``stats_iobb`` is validated for shape consistency but
    never selects.  ``mean_argmax`` is kept for reference parity and snaps the
    averaged parameters back to the search grids.
    """
    resolved = spec.resolved()
    if resolved.bbox_method != "onebox":
        raise ValueError(
            f"select_onebox requires bbox_method='onebox'; " f"got {resolved.bbox_method!r}"
        )
    if stats_iobb is not None and (
        stats_iobb.n_cfg != stats.n_cfg
        or stats_iobb.findings != stats.findings
        or stats_iobb.sum_det.shape != stats.sum_det.shape
    ):
        raise ValueError("IoU and IoBB ValStats must share the config space " "and finding axis")
    if resolved.selection == "joint_argmax":
        return JointArgmax().select(stats)
    sigma_grid = sorted({c.sigma for c in stats.configs})
    return MeanArgmax(
        sigma_grid=sigma_grid,
        thresholds=THR05,
        s_grid=ONEBOX_S_GRID,
        alpha_grid=ONEBOX_ALPHA_GRID,
        digits=None,
        box_tie_break=False,
    ).select(stats)


def val5_macros(stats: ValStats, selected: Mapping[int, Params]) -> dict[str, float]:
    """Macro acc/afp/boxes at the selected per-finding configs.

    ``acc`` and ``afp`` are the mean over findings of the val5 mean detection
    and missed counts (the reference ``ValStats`` criterion); ``boxes`` is the
    mean predicted box count.  Used for the calibration ``val5`` block.
    """
    index = {cfg.key: i for i, cfg in enumerate(stats.configs)}
    mean_det = stats.mean_det()
    mean_missed = stats.mean_missed()
    mean_boxes = stats.mean_boxes()
    accs, afps, boxes = [], [], []
    for ab in stats.findings:
        params = selected[ab]
        ci = index[params.key]
        fi = stats.finding_pos[ab]
        accs.append(float(mean_det[ci, fi]))
        afps.append(float(mean_missed[ci, fi]))
        boxes.append(float(mean_boxes[ci, fi]))
    if not accs:
        return {"acc": 0.0, "afp": 0.0, "boxes": 0.0}
    return {
        "acc": float(np.mean(accs)),
        "afp": float(np.mean(afps)),
        "boxes": float(np.mean(boxes)),
    }


# ---------------------------------------------------------------------------
# 5. fixed-parameter scoring (test pass; no scan, no fit)
# ---------------------------------------------------------------------------
def boxes_at_fixed_params(
    heatmap: np.ndarray,
    params: Params,
    *,
    bbox_method: BBoxMethod = "onebox",
    thresholds: np.ndarray = THR05,
) -> np.ndarray:
    """Extract the bboxes of one instance at frozen params.

    Reuses :class:`localization.onebox.InstanceCache` at the params' ``sigma``
    (``level(0)`` for the legacy SingleCut default).  The heatmap is copied
    into a bare :class:`~localization.onebox.Instance`; no GT box is needed.
    """
    inst = Instance("", -1, np.ascontiguousarray(heatmap, dtype=np.float64), {})
    cache = InstanceCache(inst, thresholds).level(float(params.sigma))
    return boxes_for(bbox_method, cache, params)


def score_fixed_params(
    heatmap: np.ndarray,
    bbox,
    params: Params,
    score_thresholds: Iterable[float] | None = None,
    return_boxes: bool = False,
) -> dict[str, Any]:
    """IoU and IoBB detected/missed/box counts for one instance at frozen params.

    Returns::

        {"num_boxes": int,
         "scores": {score_thr: {"iou_is_detected": int,
                                "iou_num_detected": int,
                                "iou_num_missed": int,
                                "iobb_is_detected": int,
                                "iobb_num_detected": int,
                                "iobb_num_missed": int}}}

    The per-score keys mirror the standard detection frame
    so existing consumers can aggregate them the same way.  Empty box sets
    follow the standard rule: one missed box in both families.  No
    threshold scan and no fitting is performed.
    """
    if score_thresholds is None:
        score_thresholds = list(SCORES)
    score_thresholds = list(score_thresholds)
    score_arr = np.asarray(score_thresholds, dtype=np.float64)
    bbox_method: BBoxMethod = "onebox" if isinstance(params, OneBoxParams) else "singlecut"
    cache = None
    if return_boxes:
        # one cache, so boxes and their ordering scores come from the same
        # component set (rebuilding it could in principle differ). The cache
        # must use the same 0.05 heatmap-level grid as ``boxes_at_fixed_params``:
        # THRESHOLDS_SCORE is the five *val score* cut-points, not a heatmap
        # level grid, and using it here snapped t_hi (e.g. 0.55) onto a coarser
        # level and silently changed the extracted boxes.
        inst = Instance("", -1, np.ascontiguousarray(heatmap, dtype=np.float64), {})
        cache = InstanceCache(inst, np.asarray(THR05, dtype=np.float64)).level(
            float(params.sigma)
        )
        boxes, box_scores = boxes_for_with_scores(bbox_method, cache, params)
    else:
        boxes = boxes_at_fixed_params(heatmap, params, bbox_method=bbox_method)

    # ``bbox`` is a single box dict, or a list of them under
    # ``multi_box_gt=True`` (RSNA). ``det_metrics*`` reduce over the GT boxes
    # with a best match via ``iou_vs_gt``/``iobb_vs_gt``, so pass it through
    # rather than coercing with ``dict()`` (which raises on a list).
    det_iou, nd_iou, missed_iou = det_metrics(boxes, bbox, score_arr)
    det_iobb, nd_iobb, missed_iobb = det_metrics_iobb(boxes, bbox, score_arr)
    scores: dict[float, dict[str, int]] = {}
    for j, score_thr in enumerate(score_thresholds):
        scores[float(score_thr)] = {
            "iou_is_detected": int(det_iou[j]),
            "iou_num_detected": int(nd_iou[j]),
            "iou_num_missed": int(missed_iou[j]),
            "iobb_is_detected": int(det_iobb[j]),
            "iobb_num_detected": int(nd_iobb[j]),
            "iobb_num_missed": int(missed_iobb[j]),
        }
    out: dict[str, Any] = {"num_boxes": int(boxes.shape[0]), "scores": scores}
    if return_boxes:
        # Ranked detections for AP-style metrics. ``box_scores`` is OneBox's
        # pseudo-confidence (the component peak), NOT a calibrated probability;
        # it is only meaningful for ordering within one method.
        out["boxes"] = boxes
        out["box_scores"] = box_scores
    return out


# ---------------------------------------------------------------------------
# 6. streamed OneBox val fitting
# ---------------------------------------------------------------------------
def _normalize_n_inst(findings: Sequence[int], n_inst: Mapping[int, int] | int) -> dict[int, int]:
    if isinstance(n_inst, Mapping):
        out = {int(ab): int(n_inst[ab]) for ab in findings}
    else:
        out = {int(ab): int(n_inst) for ab in findings}
    for ab, n in out.items():
        if n <= 0:
            raise ValueError(
                f"n_inst[{ab}] = {n}: every fitted finding needs at least " "one instance"
            )
    return out


class OneBoxFitter:
    """Accumulates per-config val5 sums over streamed (instance, finding) jobs.

    Five partial families per job: IoU det/missed and IoBB
    det/missed, both ``(n_cfg, 5)`` float64, plus the shared ``(n_cfg,)`` box
    counts.  Partials come from
    :func:`localization.onebox.accumulate_onebox_instance_batched` (directly or via
    :class:`OneBoxFitPool` / ``onebox_partial_job``).  Every partial value is
    an exact integer count, so chunking and merge order cannot change the
    result; :meth:`finalize` returns the IoU and IoBB ``ValStats``.
    """

    def __init__(
        self,
        spec: LocalizationMethodSpec,
        image_size: int,
        findings: Sequence[int],
        n_inst: Mapping[int, int] | int,
        search_space: OneBoxSearchSpace | None = None,
        configs: Sequence[OneBoxParams] | None = None,
    ):
        self.spec = spec.resolved()
        if self.spec.bbox_method != "onebox":
            raise ValueError(
                f"OneBoxFitter requires bbox_method='onebox'; " f"got {self.spec.bbox_method!r}"
            )
        self.image_size = int(image_size)
        self.search_space = search_space or OneBoxSearchSpace()
        self.configs = (
            list(configs) if configs is not None else self.search_space.configs(self.image_size)
        )
        self.findings = sorted(int(ab) for ab in findings)
        self.finding_pos = {ab: i for i, ab in enumerate(self.findings)}
        self.n_inst = _normalize_n_inst(self.findings, n_inst)

        n_cfg = len(self.configs)
        n_findings = len(self.findings)
        n_score = SCORES.shape[0]
        self.sum_det = np.zeros((n_cfg, n_findings, n_score), np.float64)
        self.sum_missed = np.zeros((n_cfg, n_findings, n_score), np.float64)
        self.sum_boxes = np.zeros((n_cfg, n_findings), np.float64)
        self.sum_det_iobb = np.zeros((n_cfg, n_findings, n_score), np.float64)
        self.sum_missed_iobb = np.zeros((n_cfg, n_findings, n_score), np.float64)
        self.n_partials: dict[int, int] = {ab: 0 for ab in self.findings}
        self._finalized = False

    # -- accumulation --------------------------------------------------------
    def add_partial(self, ab: int, partial: Sequence[np.ndarray]) -> None:
        """Merge one summed partial (five arrays) for finding ``ab``."""
        if self._finalized:
            raise RuntimeError("add_partial called after finalize()")
        fi = self.finding_pos.get(int(ab))
        if fi is None:
            raise KeyError(
                f"abnorm_idx {ab} is not part of this fit's finding axis " f"{self.findings}"
            )
        if len(partial) != 5:
            raise ValueError(f"expected 5 partial arrays; got {len(partial)}")
        sd, sm, sb, sdi, smi = partial
        expected_det = (len(self.configs), SCORES.shape[0])
        for name, arr in (
            ("sum_det", sd),
            ("sum_missed", sm),
            ("sum_det_iobb", sdi),
            ("sum_missed_iobb", smi),
        ):
            if np.shape(arr) != expected_det:
                raise ValueError(f"{name} partial shape {np.shape(arr)} != {expected_det}")
        if np.shape(sb) != (len(self.configs),):
            raise ValueError(
                f"sum_boxes partial shape {np.shape(sb)} != " f"{(len(self.configs),)}"
            )
        self.sum_det[:, fi, :] += sd
        self.sum_missed[:, fi, :] += sm
        self.sum_boxes[:, fi] += sb
        self.sum_det_iobb[:, fi, :] += sdi
        self.sum_missed_iobb[:, fi, :] += smi
        self.n_partials[int(ab)] += 1

    # -- finalisation --------------------------------------------------------
    def finalize(self) -> tuple[ValStats, ValStats]:
        """Return ``(IoU ValStats, IoBB ValStats)``; call once.

        The returned stats alias the fitter's accumulation buffers (the fitter
        must not be reused afterwards).  Both share the same ``configs``,
        ``findings``, ``n_inst`` and box counts; only the detection/missed
        families differ.
        """
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True
        common = dict(
            method="onebox",
            configs=self.configs,
            findings=list(self.findings),
            scores=SCORES.copy(),
            n_inst=dict(self.n_inst),
        )
        stats_iou = ValStats(
            sum_det=self.sum_det,
            sum_missed=self.sum_missed,
            sum_boxes=self.sum_boxes,
            **common,
        )
        stats_iobb = ValStats(
            sum_det=self.sum_det_iobb,
            sum_missed=self.sum_missed_iobb,
            sum_boxes=self.sum_boxes,
            **common,
        )
        return stats_iou, stats_iobb


class OneBoxWorkerError(RuntimeError):
    """A fork-pool fit chunk failed; carries its playback context.

    Raised by :class:`OneBoxFitPool` when a worker exception is re-raised in
    the parent.  The context is the chunk's submission ``tag`` (the sweep
    variant key, or ``None`` for the ordinary evaluator), the finding index
    and the first instance filename of the chunk.  ``localization.evaluation`` wraps this in
    :class:`localization.errors.DetectionWorkerError` so the failing
    (variant, filename, abnormality) triple survives to the caller.
    """

    def __init__(
        self,
        message: str,
        *,
        tag: Any = None,
        abnorm_idx: int | None = None,
        filename: str | None = None,
    ):
        self.tag = tag
        self.abnorm_idx = abnorm_idx
        self.filename = filename
        super().__init__(message)

    @classmethod
    def from_worker(
        cls, tag: Any, abnorm_idx: int, filename: str | None, exc: BaseException
    ) -> "OneBoxWorkerError":
        where = [f"finding={abnorm_idx}"]
        if filename is not None:
            where.append(f"filename={filename!r}")
        if tag is not None:
            where.append(f"variant={tag!r}")
        return cls(
            "OneBox fit worker failed for " + ", ".join(where) + ": "
            f"{type(exc).__name__}: {exc}",
            tag=tag,
            abnorm_idx=abnorm_idx,
            filename=filename,
        )


def _chunk_filename(chunk: Sequence[Any]) -> str | None:
    """First instance filename of a job chunk (informational only)."""
    for item in chunk:
        filename = getattr(item, "filename", None)
        if filename:
            return str(filename)
    return None


class OneBoxFitPool:
    """Fork process pool for exact OneBox val fits.

    The pool is created with ``multiprocessing.get_context("fork")`` and an
    initializer that ships the prebuilt :class:`~localization.onebox.BatchPlan` to
    every worker once.  Jobs are chunks of instances of one finding; the
    worker accumulates them with the batched accumulator and returns one
    summed partial per chunk (all values are exact integer counts).

    Use as a context manager.  ``submit`` buffers instances per finding and
    dispatches a job whenever ``chunk_size`` accumulate; ``drain`` returns
    completed ``(abnorm_idx, partial)`` pairs in submission order and blocks
    only to respect the in-flight window; ``flush`` dispatches the remaining
    partial chunks and waits for everything.  ``fit_chunks`` is the one-shot
    convenience wrapper.

    Callers must create the pool **before** starting other threads
    (fork-after-threads is a mutex hazard; the OneBox path needs no thread
    pool) and must fetch all results (``flush``/``flush_into``/``fit_chunks``)
    before leaving the context: an unfetched close raises ``RuntimeError``
    instead of silently dropping sums.
    """

    def __init__(
        self,
        search_space: OneBoxSearchSpace | None = None,
        image_size: int | None = None,
        workers: int | None = None,
        chunk_size: int = DEFAULT_FIT_CHUNK,
        max_in_flight: int | None = None,
        warmup: bool = True,
    ):
        if image_size is None:
            raise ValueError("OneBoxFitPool requires image_size (short side)")
        self.search_space = search_space or OneBoxSearchSpace()
        self.image_size = int(image_size)
        self.configs = self.search_space.configs(self.image_size)
        self.batch_plan = build_batch_plan(
            build_onebox_plan(
                self.configs,
                self.search_space.thresholds,
                s_grid=self.search_space.s_grid,
                alpha_grid=self.search_space.alpha_grid,
            ),
            len(self.configs),
            s_grid=self.search_space.s_grid,
            alpha_grid=self.search_space.alpha_grid,
            thresholds=self.search_space.thresholds,
        )
        self.chunk_size = max(1, int(chunk_size))
        self.workers = None if workers is None else max(1, int(workers))
        if max_in_flight is not None:
            self.max_in_flight = max(1, int(max_in_flight))
        else:
            n_workers = self.workers or (os.cpu_count() or 1)
            self.max_in_flight = max(2 * n_workers, 2)
        self._warmup = bool(warmup)
        self._pool: Any | None = None
        # buffers are keyed by (tag, finding) so a shared sweep pool can keep
        # the variants' instance streams separate; ``tag`` is ``None``
        # for the ordinary evaluator.
        self._buffers: dict[tuple[Any, int], list] = defaultdict(list)
        self._pending: deque[tuple[Any, int, str | None, Any]] = deque()

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "OneBoxFitPool":
        if self._pool is not None:
            raise RuntimeError("OneBoxFitPool is already open")
        if self._warmup:
            warmup_onebox()
        ctx = mp.get_context("fork")
        n_workers = self.workers if self.workers is not None else (os.cpu_count() or 1)
        self._pool = ctx.Pool(
            processes=max(1, int(n_workers)),
            initializer=onebox_worker_init,
            initargs=(self.batch_plan,),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._pool is None:
            return False
        leftover = bool(self._pending) or any(self._buffers.values())
        if exc_type is not None or leftover:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()
        self._pool = None
        if exc_type is None and leftover:
            raise RuntimeError(
                "OneBoxFitPool exited with unfetched work; call flush()/"
                "flush_into() (or fit_chunks) before closing the context"
            )
        return False

    def _ensure_open(self) -> None:
        if self._pool is None:
            raise RuntimeError(
                "OneBoxFitPool must be used inside its context "
                "manager (with OneBoxFitPool(...) as pool:)"
            )

    # -- fitter construction -------------------------------------------------
    def make_fitter(
        self, spec: LocalizationMethodSpec, findings: Sequence[int], n_inst: Mapping[int, int] | int
    ) -> OneBoxFitter:
        """A fitter wired to this pool's config space (no BatchPlan rebuild)."""
        return OneBoxFitter(
            spec,
            self.image_size,
            findings,
            n_inst,
            search_space=self.search_space,
            configs=self.configs,
        )

    # -- streaming submission ------------------------------------------------
    def submit(self, ab: int, items: Sequence[Any], tag: Any = None) -> None:
        """Buffer instances of finding ``ab`` and dispatch full chunks.

        ``tag`` is an optional opaque label (used by the sweep evaluator to
        keep variants apart while sharing one pool).  Buffers are keyed by
        ``(tag, finding)`` so a chunk never mixes variants.  Chunk payloads
        may be :class:`~localization.onebox.Instance` objects: the first filename is
        recorded so worker failures can name the failing file.
        """
        self._ensure_open()
        ab = int(ab)
        buf = self._buffers[(tag, ab)]
        buf.extend(items)
        while len(buf) >= self.chunk_size:
            chunk = buf[: self.chunk_size]
            del buf[: self.chunk_size]
            self._dispatch(ab, chunk, tag=tag)

    def _dispatch(self, ab: int, chunk: Sequence[Any], tag: Any = None) -> None:
        ar = self._pool.apply_async(onebox_partial_job, (list(chunk),))
        self._pending.append((tag, int(ab), _chunk_filename(chunk), ar))

    @staticmethod
    def _resolve(tag: Any, ab: int, filename: str | None, ar: Any) -> tuple:
        try:
            return ar.get()
        except BaseException as exc:
            raise OneBoxWorkerError.from_worker(tag, int(ab), filename, exc) from exc

    def _pop_ready(self) -> list[tuple[Any, int, tuple]]:
        results: list[tuple[Any, int, tuple]] = []
        while self._pending and self._pending[0][3].ready():
            tag, ab, filename, ar = self._pending.popleft()
            results.append((tag, ab, self._resolve(tag, ab, filename, ar)))
        return results

    def _enforce_window(self) -> list[tuple[Any, int, tuple]]:
        results: list[tuple[Any, int, tuple]] = []
        while len(self._pending) > self.max_in_flight:
            tag, ab, filename, ar = self._pending.popleft()
            results.append((tag, ab, self._resolve(tag, ab, filename, ar)))
        return results

    def drain_tagged(self) -> list[tuple[Any, int, tuple]]:
        """Like :meth:`drain` but preserves the submission ``tag``."""
        self._ensure_open()
        results = self._enforce_window()
        results.extend(self._pop_ready())
        return results

    def drain(self) -> list[tuple[int, tuple]]:
        """Return completed chunks in submission order (blocks on the window)."""
        return [(ab, partial) for _tag, ab, partial in self.drain_tagged()]

    def drain_into(self, fitter: OneBoxFitter) -> int:
        """``drain()`` straight into a fitter; returns the number merged."""
        results = self.drain()
        for ab, partial in results:
            fitter.add_partial(ab, partial)
        return len(results)

    def flush_tagged(self) -> list[tuple[Any, int, tuple]]:
        """Like :meth:`flush` but preserves the submission ``tag``."""
        self._ensure_open()
        for (tag, ab), buf in list(self._buffers.items()):
            while buf:
                chunk = buf[: self.chunk_size]
                del buf[: self.chunk_size]
                self._dispatch(ab, chunk, tag=tag)
        results: list[tuple[Any, int, tuple]] = []
        while self._pending:
            tag, ab, filename, ar = self._pending.popleft()
            results.append((tag, ab, self._resolve(tag, ab, filename, ar)))
        return results

    def flush(self) -> list[tuple[int, tuple]]:
        """Dispatch all buffered chunks and wait for every outstanding job."""
        return [(ab, partial) for _tag, ab, partial in self.flush_tagged()]

    def flush_into(self, fitter: OneBoxFitter) -> int:
        """``flush()`` straight into a fitter; returns the number merged."""
        results = self.flush()
        for ab, partial in results:
            fitter.add_partial(ab, partial)
        return len(results)

    # -- one-shot convenience ------------------------------------------------
    def fit_chunks(
        self,
        jobs: Iterable[tuple[int, Sequence[Any]]],
        spec: LocalizationMethodSpec,
        findings: Sequence[int],
        n_inst: Mapping[int, int] | int,
    ) -> tuple[ValStats, ValStats]:
        """Stream ``(abnorm_idx, instances)`` jobs through the pool and fit.

        The jobs are chunked by ``chunk_size`` per finding; results are merged
        into a :class:`OneBoxFitter` in submission order and
        ``(stats_iou, stats_iobb)`` is returned.
        """
        fitter = self.make_fitter(spec, findings, n_inst)
        owns_pool = self._pool is None
        if owns_pool:
            with self:
                self._fit_chunks_inner(jobs, fitter)
        else:
            self._fit_chunks_inner(jobs, fitter)
        return fitter.finalize()

    def _fit_chunks_inner(
        self, jobs: Iterable[tuple[int, Sequence[Any]]], fitter: OneBoxFitter
    ) -> None:
        for ab, items in jobs:
            self.submit(ab, items)
            self.drain_into(fitter)
        self.flush_into(fitter)


# ---------------------------------------------------------------------------
# 7. calibration artifact
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizationCalibration:
    """The frozen method artifact persisted next to ``heatmap_thr``.

    ``params`` holds one ``SingleCutParams`` or ``OneBoxParams`` per finding,
    selected on val5 IoU (``selection_metric == "IoU"``).
    ``params_iobb`` is populated **only** in the legacy
    ``optimize_metrics_separately=True`` mode (the historical IoBB-selected
    thresholds); the default path never fits on IoBB.  The test pass consumes
    this artifact verbatim and never re-fits.
    """

    bbox_method: BBoxMethod
    selection: SelectionStrategy
    selection_metric: str
    params: dict[int, Params]
    params_iobb: dict[int, Params] | None
    image_size: int
    search_space: dict
    val5: dict
    fit_seconds: float
    fit_workers: int
    optimize_metrics_separately: bool = False

    def __post_init__(self) -> None:
        if self.bbox_method not in ("singlecut", "onebox"):
            raise ValueError(f"unknown bbox_method {self.bbox_method!r}")
        if self.selection not in ("joint_argmax", "mean_argmax"):
            raise ValueError(f"unknown selection {self.selection!r}")
        if self.selection_metric != "IoU":
            raise ValueError(
                "selection_metric must be 'IoU' (IoBB is never "
                f"selected on the default path); got {self.selection_metric!r}"
            )
        if self.params_iobb is not None and not self.optimize_metrics_separately:
            raise ValueError("params_iobb is only valid with optimize_metrics_separately=True")


def _with_sigma_rel(params: Params, image_size: int) -> Params:
    """Fill ``OneBoxParams.sigma_rel = sigma_px / short_side``.

    The selectors return grid configs whose pixel sigma is the identity; the
    short side is only known to the calibration builder, so the persisted
    ``sigma_rel`` (schema) is attached here.  ``SingleCutParams`` pass
    through untouched.
    """
    if isinstance(params, OneBoxParams) and image_size:
        return replace(params, sigma_rel=float(params.sigma) / float(image_size))
    return params


def build_calibration(
    spec: LocalizationMethodSpec,
    params: Mapping[int, Params],
    *,
    image_size: int,
    params_iobb: Mapping[int, Params] | None = None,
    search_space: OneBoxSearchSpace | Mapping[str, Any] | None = None,
    val5: Mapping[str, float] | None = None,
    fit_seconds: float = 0.0,
    fit_workers: int = 1,
) -> LocalizationCalibration:
    """Build the frozen calibration artifact from a selection.

    ``search_space`` may be an :class:`OneBoxSearchSpace` (summarised for the
    given ``image_size``) or a pre-built mapping; ``val5`` is the output of
    :func:`val5_macros`.  OneBox params are annotated with their
    ``sigma_rel`` (``sigma_px / short_side``).
    ``selection_metric`` is always ``"IoU"``.
    """
    resolved = spec.resolved()
    if isinstance(search_space, OneBoxSearchSpace):
        search_space_dict: dict[str, Any] = search_space.summary(image_size)
    elif search_space is None:
        search_space_dict = {}
    else:
        search_space_dict = dict(search_space)
    return LocalizationCalibration(
        bbox_method=resolved.bbox_method,
        selection=resolved.selection,
        selection_metric="IoU",
        params={int(ab): _with_sigma_rel(p, image_size) for ab, p in params.items()},
        params_iobb=(
            None
            if params_iobb is None
            else {int(ab): _with_sigma_rel(p, image_size) for ab, p in params_iobb.items()}
        ),
        image_size=int(image_size),
        search_space=search_space_dict,
        val5=dict(val5) if val5 is not None else {},
        fit_seconds=float(fit_seconds),
        fit_workers=int(fit_workers),
        optimize_metrics_separately=bool(resolved.optimize_metrics_separately),
    )
