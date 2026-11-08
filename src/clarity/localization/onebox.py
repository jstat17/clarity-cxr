"""Fast OneBox bbox extraction.

Contents: the constants, level/instance caches, geometry and gate kernels,
the parameter grids, selection strategies and the OneBox accumulators.

Two accumulators are provided and MUST produce identical results:

* :func:`accumulate_onebox_instance` — literal per-config body (reference and
  generic fallback);
* :func:`accumulate_onebox_instance_batched` — the hot path (one geometry
  pass per sigma over the components of *all* threshold pairs).

The IoU arithmetic is gated against the literal accumulator and the IoBB
partials against a naive per-box oracle.  Do not change semantics without a
parity check.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numba
import numpy as np
from scipy.ndimage import gaussian_filter

from clarity.localization.constants import THRESHOLDS_HEATMAP, THRESHOLDS_SCORE

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: the 91-level 0.01 heatmap-threshold array (legacy SingleCut grid)
THR01: np.ndarray = np.asarray(THRESHOLDS_HEATMAP, dtype=np.float64)
#: the 0.05 sub-grid used by the OneBox search (19 levels)
THR05: np.ndarray = np.ascontiguousarray(THR01[::5])
#: canonical val score thresholds (val5)
SCORES: np.ndarray = np.asarray(THRESHOLDS_SCORE, dtype=np.float64)

#: frozen gate
GATE_AREA = 4
GATE_PEAK_RATIO = 0.0
#: output capacity per threshold in the component kernel.  Real CLARiTy maps
#: produce <= ~5 4-connected components per level; 64 keeps the per-instance
#: cache tiny (a 4096 default costs ~3 MB per (instance, sigma)).
#: NEVER raise this to the kernel default.
MAX_BOXES = 64

#: resolution-adaptive sigma parameterisation.  At the 224
#: reference size the relative grid equals the pixel grid {0..4}.
SIGMA_REL_REF = 224.0
SIGMA_REL_MAX = 4.0
#: frozen OneBox parameter grids 
ONEBOX_S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
ONEBOX_ALPHA_GRID = (0.9, 1.0, 1.1)

# The 19-level 0.05 grid is fixed; fail loudly if it changes.
assert len(THR05) == 19 and THR05[0] == 0.05 and THR05[-1] == 0.95


# ---------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instance:
    """One (image, finding) problem: an attention map and a GT bbox dict.

    ``filename``/``abnorm_idx`` are informational only
    and never touched by the numeric code.
    """

    filename: str
    abnorm_idx: int
    heatmap: np.ndarray
    bbox: dict

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Instance({self.filename}, abn={self.abnorm_idx})"


# ---------------------------------------------------------------------------
# numba kernels — verbatim ports
# ---------------------------------------------------------------------------
@numba.njit(cache=True, nogil=True, inline="always")
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@numba.njit(cache=True, nogil=True)
def boxes_attrs_at_thresholds(hm_flat, H, W, thr, max_boxes=MAX_BOXES):
    """Like ``boxes_at_thresholds`` but also returns
    per-component pixel attributes at every threshold:
    ``(boxes, counts, areas, peaks, means)``.

    ``boxes[k, :counts[k]]`` are the 4-connected component bboxes of
    ``hm > thr[k]`` (row_start, row_end, col_start, col_end);
    ``areas[k, i]`` = component pixel count, ``peaks[k, i]`` = max heatmap
    value in the component, ``means[k, i]`` = mean heatmap value.  Component
    ordering matches ``boxes_at_thresholds`` (linked-list order).
    """
    n_thr = thr.shape[0]
    boxes = np.zeros((n_thr, max_boxes, 4), np.int32)
    counts = np.zeros(n_thr, np.int32)
    areas = np.zeros((n_thr, max_boxes), np.int64)
    peaks = np.zeros((n_thr, max_boxes), np.float64)
    means = np.zeros((n_thr, max_boxes), np.float64)
    HW = H * W
    bucket = np.zeros(n_thr + 1, np.int64)
    for p in range(HW):
        v = hm_flat[p]
        b = 0
        for j in range(n_thr):
            if thr[j] < v:
                b += 1
            else:
                break
        bucket[b] += 1
    offs = np.zeros(n_thr + 2, np.int64)
    for b in range(n_thr + 1):
        offs[b + 1] = offs[b] + bucket[b]
    fill = offs.copy()
    pixels_by_bucket = np.empty(HW, np.int64)
    for p in range(HW):
        v = hm_flat[p]
        b = 0
        for j in range(n_thr):
            if thr[j] < v:
                b += 1
            else:
                break
        pixels_by_bucket[fill[b]] = p
        fill[b] += 1

    parent = np.arange(HW)
    added = np.zeros(HW, numba.boolean)
    nrs = np.empty(HW, np.int64)
    nre = np.empty(HW, np.int64)
    ncs = np.empty(HW, np.int64)
    nce = np.empty(HW, np.int64)
    nnx = np.empty(HW, np.int64)
    npr = np.empty(HW, np.int64)
    n_ar = np.empty(HW, np.int64)
    n_pk = np.empty(HW, np.float64)
    n_sm = np.empty(HW, np.float64)
    head = -1

    for k in range(n_thr - 1, -1, -1):
        for ii in range(offs[k + 1], offs[k + 1] + bucket[k + 1]):
            p = pixels_by_bucket[ii]
            v = hm_flat[p]
            added[p] = True
            r = p // W
            c = p - r * W
            parent[p] = p
            nrs[p] = r
            nre[p] = r + 1
            ncs[p] = c
            nce[p] = c + 1
            n_ar[p] = 1
            n_pk[p] = v
            n_sm[p] = v
            nnx[p] = head
            npr[p] = -1
            if head >= 0:
                npr[head] = p
            head = p
            for q in (
                p - 1 if c > 0 else -1,
                p + 1 if c < W - 1 else -1,
                p - W if r > 0 else -1,
                p + W if r < H - 1 else -1,
            ):
                if q < 0 or not added[q]:
                    continue
                ra = _uf_find(parent, p)
                rb = _uf_find(parent, q)
                if ra == rb:
                    continue
                parent[rb] = ra
                if nrs[rb] < nrs[ra]:
                    nrs[ra] = nrs[rb]
                if nre[rb] > nre[ra]:
                    nre[ra] = nre[rb]
                if ncs[rb] < ncs[ra]:
                    ncs[ra] = ncs[rb]
                if nce[rb] > nce[ra]:
                    nce[ra] = nce[rb]
                n_ar[ra] += n_ar[rb]
                n_sm[ra] += n_sm[rb]
                if n_pk[rb] > n_pk[ra]:
                    n_pk[ra] = n_pk[rb]
                if npr[rb] >= 0:
                    nnx[npr[rb]] = nnx[rb]
                else:
                    head = nnx[rb]
                if nnx[rb] >= 0:
                    npr[nnx[rb]] = npr[rb]
        n = 0
        x = head
        while x >= 0 and n < max_boxes:
            boxes[k, n, 0] = nrs[x]
            boxes[k, n, 1] = nre[x]
            boxes[k, n, 2] = ncs[x]
            boxes[k, n, 3] = nce[x]
            areas[k, n] = n_ar[x]
            peaks[k, n] = n_pk[x]
            means[k, n] = n_sm[x] / n_ar[x]
            n += 1
            x = nnx[x]
        counts[k] = n
    return boxes, counts, areas, peaks, means


@numba.njit(cache=True, nogil=True)
def gated_parts_r0(boxes, counts, areas, peaks, k_lo, k_hi, a):
    """Return ``(gidx, core, cpeak, lpeak)`` for the ``r = 0`` gate.

    ``gidx`` indexes the low-level components (ascending, matching
    ``np.unique(par[ok])``), ``core`` is the bbox union of the gated high
    components of each surviving low component, and ``cpeak`` / ``lpeak`` are
    the corresponding max blurred-map peaks.
    """
    n_lo = counts[k_lo]
    n_hi = counts[k_hi]
    if n_lo == 0 or n_hi == 0:
        return (
            np.zeros(0, np.int64),
            np.zeros((0, 4), np.int32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
        )

    par = np.full(n_hi, -1, np.int64)

    # np-style parent map: use bbox containment for *all* high boxes when at
    # least one containment exists; otherwise fall back to centre containment
    # for all.  Parent = smallest-area containing low box.
    any_inside = False
    for j in range(n_hi):
        h0 = boxes[k_hi, j, 0]
        h1 = boxes[k_hi, j, 1]
        h2 = boxes[k_hi, j, 2]
        h3 = boxes[k_hi, j, 3]
        for i in range(n_lo):
            if (
                boxes[k_lo, i, 0] <= h0
                and boxes[k_lo, i, 1] >= h1
                and boxes[k_lo, i, 2] <= h2
                and boxes[k_lo, i, 3] >= h3
            ):
                any_inside = True
                break
        if any_inside:
            break

    if any_inside:
        for j in range(n_hi):
            h0 = boxes[k_hi, j, 0]
            h1 = boxes[k_hi, j, 1]
            h2 = boxes[k_hi, j, 2]
            h3 = boxes[k_hi, j, 3]
            best_area = np.iinfo(np.int64).max
            best_i = -1
            for i in range(n_lo):
                l0 = boxes[k_lo, i, 0]
                l1 = boxes[k_lo, i, 1]
                l2 = boxes[k_lo, i, 2]
                l3 = boxes[k_lo, i, 3]
                if l0 <= h0 and l1 >= h1 and l2 <= h2 and l3 >= h3:
                    ar = np.int64(l1 - l0) * np.int64(l3 - l2)
                    if ar < best_area:
                        best_area = ar
                        best_i = i
            par[j] = best_i
    else:
        for j in range(n_hi):
            rc = (boxes[k_hi, j, 0] + boxes[k_hi, j, 1]) / 2.0
            cc = (boxes[k_hi, j, 2] + boxes[k_hi, j, 3]) / 2.0
            best_area = np.iinfo(np.int64).max
            best_i = -1
            for i in range(n_lo):
                l0 = boxes[k_lo, i, 0]
                l1 = boxes[k_lo, i, 1]
                l2 = boxes[k_lo, i, 2]
                l3 = boxes[k_lo, i, 3]
                if l0 <= rc and rc <= l1 and l2 <= cc and cc <= l3:
                    ar = np.int64(l1 - l0) * np.int64(l3 - l2)
                    if ar < best_area:
                        best_area = ar
                        best_i = i
            par[j] = best_i

    gidx = np.empty(n_lo, np.int64)
    core = np.empty((n_lo, 4), np.int32)
    cpeak = np.empty(n_lo, np.float32)
    ng = 0
    for i in range(n_lo):
        r0 = np.int64(1 << 30)
        r1 = np.int64(-1)
        c0 = np.int64(1 << 30)
        c1 = np.int64(-1)
        pk = np.float32(-1.0)
        found = False
        for j in range(n_hi):
            if par[j] != i or areas[k_hi, j] < a:
                continue
            found = True
            if boxes[k_hi, j, 0] < r0:
                r0 = boxes[k_hi, j, 0]
            if boxes[k_hi, j, 1] > r1:
                r1 = boxes[k_hi, j, 1]
            if boxes[k_hi, j, 2] < c0:
                c0 = boxes[k_hi, j, 2]
            if boxes[k_hi, j, 3] > c1:
                c1 = boxes[k_hi, j, 3]
            if peaks[k_hi, j] > pk:
                pk = peaks[k_hi, j]
        if found:
            gidx[ng] = i
            core[ng, 0] = r0
            core[ng, 1] = r1
            core[ng, 2] = c0
            core[ng, 3] = c1
            cpeak[ng] = pk
            ng += 1

    return gidx[:ng], core[:ng], cpeak[:ng], peaks[k_lo, gidx[:ng]]


# ---------------------------------------------------------------------------
# component / level caches (verbatim reference semantics)
# ---------------------------------------------------------------------------
@dataclass
class LevelCache:
    """Connected-component attributes at every level of ``thresholds``.

    ``boxes[k, :counts[k]]`` are the 4-connected component bboxes of
    ``blurred_map > thresholds[k]`` (row_start, row_end, col_start, col_end);
    ``areas`` / ``peaks`` are per component.  ``means`` is always ``None`` in
    the standard caches (the extractors never read it).
    """

    sigma: float
    thresholds: np.ndarray
    blurred: np.ndarray | None
    boxes: np.ndarray
    counts: np.ndarray
    areas: np.ndarray
    peaks: np.ndarray
    means: np.ndarray | None
    #: memoized ``extract_gated_parts`` results keyed by
    #: ``(k_lo, k_hi, min_area, GATE_PEAK_RATIO)``
    gated: dict = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return int(self.thresholds.shape[0])

    def as_tuple(self):
        return (self.blurred, self.boxes, self.counts, self.areas, self.peaks, self.means)

    def boxes_at(self, k: int) -> np.ndarray:
        return self.boxes[k, : self.counts[k]]


def _blur(hm: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ascontiguousarray(hm, dtype=np.float64)
    return np.ascontiguousarray(gaussian_filter(hm, sigma=sigma, mode="nearest"), dtype=np.float64)


class InstanceCache:
    """Lazily-built per-``sigma`` level caches for one (image, finding).

    All cached levels share the same threshold array (passed at construction),
    so configuration indices are comparable across sigmas.
    """

    __slots__ = (
        "inst",
        "thresholds",
        "_by_sigma",
        "n_builds",
        "build_time",
        "max_boxes",
        "store_blurred",
    )

    def __init__(
        self,
        inst: Instance,
        thresholds: Sequence[float],
        max_boxes: int = MAX_BOXES,
        store_blurred: bool = False,
    ):
        self.inst = inst
        self.thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
        self._by_sigma: dict[float, LevelCache] = {}
        self.n_builds = 0
        self.build_time = 0.0
        self.max_boxes = int(max_boxes)
        # the blurred map is not needed for box extraction; keep it only for
        # parity/debugging (it is the largest per-instance allocation)
        self.store_blurred = bool(store_blurred)

    def level(self, sigma: float) -> LevelCache:
        sigma = float(sigma)
        cache = self._by_sigma.get(sigma)
        if cache is None:
            t0 = time.perf_counter()
            hm = _blur(self.inst.heatmap, sigma)
            boxes, counts, areas, peaks, means = boxes_attrs_at_thresholds(
                hm.reshape(-1),
                hm.shape[0],
                hm.shape[1],
                self.thresholds,
                self.max_boxes,
            )
            cache = LevelCache(
                sigma=sigma,
                thresholds=self.thresholds,
                blurred=hm if self.store_blurred else None,
                boxes=boxes,
                counts=counts,
                # compact attrs: component areas/peaks are small ints / map
                # values already float32; the kernel returns int64/float64
                areas=np.asarray(areas, dtype=np.int32),
                peaks=np.asarray(peaks, dtype=np.float32),
                # means are unused by the extractors; drop the allocation
                means=None,
            )
            self._by_sigma[sigma] = cache
            self.n_builds += 1
            self.build_time += time.perf_counter() - t0
        return cache


def index_of(thresholds: np.ndarray, value: float) -> int:
    """Exact (or nearest, if slightly off-grid) index of ``value``."""
    return int(np.argmin(np.abs(thresholds - value)))


# ---------------------------------------------------------------------------
# box-geometry primitives (verbatim reference semantics)
# ---------------------------------------------------------------------------
def scale_boxes(boxes: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 1.0 or boxes.shape[0] == 0:
        return boxes
    rs, re, cs, ce = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    rc, cc = (rs + re) / 2.0, (cs + ce) / 2.0
    hh, hw = (re - rs) / 2.0 * alpha, (ce - cs) / 2.0 * alpha
    out = np.empty_like(boxes)
    out[:, 0] = np.clip(np.round(rc - hh), 0, None)
    out[:, 1] = np.round(rc + hh)
    out[:, 2] = np.clip(np.round(cc - hw), 0, None)
    out[:, 3] = np.round(cc + hw)
    return out


def mid_box(gL: np.ndarray, core: np.ndarray, s: float) -> np.ndarray:
    """Edgewise interpolation ``core + s * (gL - core)``, rounded+clipped."""
    out = core.astype(np.float64) + s * (gL.astype(np.float64) - core.astype(np.float64))
    b = np.round(out).astype(np.int32)
    b[:, 0] = np.clip(b[:, 0], 0, None)
    b[:, 2] = np.clip(b[:, 2], 0, None)
    return b


def mid_boxes_batch(gL: np.ndarray, core: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """All ``s`` values at once: ``(n_s, n_comp, 4)`` int32 mid boxes."""
    out = core[None, :, :].astype(np.float64) + s_grid[:, None, None] * (
        gL[None, :, :].astype(np.float64) - core[None, :, :].astype(np.float64)
    )
    b = np.round(out).astype(np.int32)
    b[..., 0] = np.clip(b[..., 0], 0, None)
    b[..., 2] = np.clip(b[..., 2], 0, None)
    return b


def box_array(gt) -> np.ndarray:
    """Normalise a ground-truth value into an ``(n, 4)`` float64 box array.

    Accepts the historical single box dict, a ``multi_box=True`` list of box
    dicts, or an already-numeric ``(n, 4)`` array. Keeping this in one place is
    what makes the multi-box path additive: a single dict yields a ``(1, 4)``
    array, so the numeric kernels see exactly the same geometry as before.
    """
    if isinstance(gt, np.ndarray):
        arr = np.asarray(gt, dtype=np.float64)
        return arr.reshape(-1, 4) if arr.size else np.zeros((0, 4), dtype=np.float64)
    if isinstance(gt, dict):
        gt = [gt]
    boxes = [
        (box["row_start"], box["row_end"], box["col_start"], box["col_end"])
        for box in gt
        if box is not None
    ]
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _iou_pairs(rs, re, cs, ce, grs, gre, gcs, gce) -> np.ndarray:
    """IoU of each predicted box against each GT box; broadcasts to (n_pred, n_gt)."""
    ih = np.maximum(0.0, np.minimum(re[..., None], gre) - np.maximum(rs[..., None], grs))
    iw = np.maximum(0.0, np.minimum(ce[..., None], gce) - np.maximum(cs[..., None], gcs))
    inter = ih * iw
    gt_area = np.maximum(0.0, gre - grs) * np.maximum(0.0, gce - gcs)
    pred_area = (re - rs) * (ce - cs)
    union = gt_area[None, :] + pred_area[..., None] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def iou_vs_gt(boxes: np.ndarray, gt) -> np.ndarray:
    """IoU of every predicted box against the ground-truth box; ``(n,)`` float64.

    With several GT boxes (``multi_box=True``) each predicted box scores its
    **best** match, which is the standard detection convention and degenerates
    to the historical single-box value when there is one GT box.
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    gt_arr = box_array(gt)
    if gt_arr.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=np.float64)
    rs, re, cs, ce = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ious = _iou_pairs(rs, re, cs, ce, gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3])
    return ious.max(axis=1)


def iou_matrix(boxes: np.ndarray, gt) -> np.ndarray:
    """Full ``(n_pred, n_gt)`` IoU matrix (used by AP matching)."""
    gt_arr = box_array(gt)
    if boxes.shape[0] == 0 or gt_arr.shape[0] == 0:
        return np.zeros((boxes.shape[0], gt_arr.shape[0]), dtype=np.float64)
    rs, re, cs, ce = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return _iou_pairs(rs, re, cs, ce, gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3])


def iobb_vs_gt(boxes: np.ndarray, gt) -> np.ndarray:
    """IoBB (precision) of every box: intersection / predicted area.

    Verbatim standard definition from ``localization.metrics.calculate_iobb``:
    the denominator is the *predicted* box area and the value is ``0`` when
    that area is ``0``.  Boxes are ``(n, 4)`` int/float row/col starts/ends.

    With several GT boxes each predicted box scores its **best** match, matching
    :func:`iou_vs_gt`; one GT box reproduces the historical value exactly.
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    gt_arr = box_array(gt)
    if gt_arr.shape[0] == 0:
        return np.zeros(boxes.shape[0], dtype=np.float64)
    rs, re, cs, ce = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ih = np.maximum(0.0, np.minimum(re[..., None], gt_arr[None, :, 1]) - np.maximum(rs[..., None], gt_arr[None, :, 0]))
    iw = np.maximum(0.0, np.minimum(ce[..., None], gt_arr[None, :, 3]) - np.maximum(cs[..., None], gt_arr[None, :, 2]))
    inter = ih * iw
    pred_area = ((re - rs) * (ce - cs))[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_gt = np.where(pred_area != 0, inter / pred_area, 0.0)
    return per_gt.max(axis=1)


def det_metrics(
    boxes: np.ndarray, gt, scores: np.ndarray = SCORES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(is_detected, n_detected, n_missed)`` per score threshold (IoU).

    Empty box sets follow standard semantics: one missed detection.
    ``gt`` may be one box dict or a ``multi_box_gt`` list; :func:`iou_vs_gt`
    reduces each prediction to its best match.
    """
    if boxes.shape[0] == 0:
        n = scores.shape[0]
        return (
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        )
    ious = iou_vs_gt(boxes, gt)
    nd = (ious[None, :] >= scores[:, None]).sum(axis=1)
    return (
        (nd > 0).astype(np.uint8),
        nd.astype(np.float32),
        (boxes.shape[0] - nd).astype(np.float32),
    )


def det_metrics_iobb(
    boxes: np.ndarray, gt, scores: np.ndarray = SCORES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(is_detected, n_detected, n_missed)`` per score threshold (IoBB).

    Same empty-map rule as :func:`det_metrics`; the numbers are the standard
    ``calculate_iobb`` counts.  Added for the default evaluation (both families from
    one IoU-optimised calibration).
    """
    if boxes.shape[0] == 0:
        n = scores.shape[0]
        return (
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        )
    iobb = iobb_vs_gt(boxes, gt)
    nd = (iobb[None, :] >= scores[:, None]).sum(axis=1)
    return (
        (nd > 0).astype(np.uint8),
        nd.astype(np.float32),
        (boxes.shape[0] - nd).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# gated-parts numpy reference (test reference only, never on the hot path)
# ---------------------------------------------------------------------------
def parent_map(lo_b: np.ndarray, hi_b: np.ndarray) -> np.ndarray:
    """Smallest low box containing each high box (-1 if none)."""
    n_lo, n_hi = lo_b.shape[0], hi_b.shape[0]
    par = np.full(n_hi, -1, np.int64)
    if n_lo == 0 or n_hi == 0:
        return par
    inside = (
        (lo_b[:, 0, None] <= hi_b[None, :, 0])
        & (lo_b[:, 1, None] >= hi_b[None, :, 1])
        & (lo_b[:, 2, None] <= hi_b[None, :, 2])
        & (lo_b[:, 3, None] >= hi_b[None, :, 3])
    )
    if not inside.any():
        rc = (hi_b[:, 0] + hi_b[:, 1]) / 2.0
        cc = (hi_b[:, 2] + hi_b[:, 3]) / 2.0
        inside = (
            (lo_b[:, 0, None] <= rc[None, :])
            & (lo_b[:, 1, None] >= rc[None, :])
            & (lo_b[:, 2, None] <= cc[None, :])
            & (lo_b[:, 3, None] >= cc[None, :])
        )
    area = (lo_b[:, 1] - lo_b[:, 0]).astype(np.int64) * (lo_b[:, 3] - lo_b[:, 2]).astype(np.int64)
    sentinel = np.iinfo(np.int64).max
    masked = np.where(inside, area[:, None], sentinel)
    best = masked.argmin(0)
    has = inside.any(0)
    par[has] = best[has]
    return par


def extract_gated_parts(
    cache: LevelCache, k_lo: int, k_hi: int, a: int = GATE_AREA, r: float = GATE_PEAK_RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core-gated low components at one threshold pair.

    Returns ``(low_boxes, core_boxes, core_peaks, low_peaks)`` for every low
    component containing at least one core with ``area >= a`` (and
    ``peak >= r * peak(low)`` when ``r > 0``).  Numpy port of the reference
    reference; tests only.
    """
    boxes, counts, areas, peaks = (cache.boxes, cache.counts, cache.areas, cache.peaks)
    n_hi = int(counts[k_hi])
    L = boxes[k_lo, : counts[k_lo]].astype(np.int32)
    H = boxes[k_hi, :n_hi].astype(np.int32)
    empty = (
        np.zeros((0, 4), np.int32),
        np.zeros((0, 4), np.int32),
        np.zeros(0, np.float64),
        np.zeros(0, np.float64),
    )
    if n_hi == 0 or L.shape[0] == 0:
        return empty
    par = parent_map(L, H)
    ok = par >= 0
    if a > 1:
        ok &= areas[k_hi, :n_hi] >= a
    if r > 0:
        ratio = np.zeros(n_hi, np.float64)
        valid = par >= 0
        ratio[valid] = peaks[k_hi, :n_hi][valid] / np.maximum(
            peaks[k_lo, : counts[k_lo]][par[valid]], 1e-12
        )
        ok &= ratio >= r
    if not ok.any():
        return empty
    gidx = np.unique(par[ok])
    gL = L[gidx]
    core = np.empty((len(gidx), 4), np.int32)
    cpeak = np.empty(len(gidx), np.float64)
    for j, gi in enumerate(gidx):
        sel = ok & (par == gi)
        hh = H[sel]
        core[j] = (hh[:, 0].min(), hh[:, 1].max(), hh[:, 2].min(), hh[:, 3].max())
        cpeak[j] = peaks[k_hi, :n_hi][sel].max()
    lpeak = peaks[k_lo, : counts[k_lo]][gidx]
    return gL, core, cpeak, lpeak


# ---------------------------------------------------------------------------
# method parameter containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SingleCutParams:
    """Legacy SingleCut extractor: one absolute cut, component bboxes."""

    threshold: float
    sigma: float = 0.0
    alpha: float = 1.0
    min_area: int = 0

    @property
    def key(self) -> tuple:
        return (
            "single",
            float(self.threshold),
            float(self.sigma),
            float(self.alpha),
            int(self.min_area),
        )

    def describe(self) -> str:
        return (
            f"SingleCut(t={self.threshold:.4f}, s={self.sigma:g}, "
            f"a={self.alpha:g}, A={self.min_area})"
        )


@dataclass(frozen=True)
class OneBoxParams:
    """OneBox: gated mid box per confirmed component.

    ``key`` is the exact reference tuple (with the anisotropic slot pinned to
    ``-1.0``; anisotropic scaling is out of scope.
    ``sigma_rel = sigma / short_side`` is persisted for reproducibility and is not part of the identity key.
    """

    sigma: float
    t_lo: float
    t_hi: float
    s: float
    alpha: float = 1.0
    min_area: int = GATE_AREA
    sigma_rel: float | None = None

    @property
    def key(self) -> tuple:
        return (
            "onebox",
            float(self.t_lo),
            float(self.t_hi),
            float(self.sigma),
            float(self.s),
            float(self.alpha),
            -1.0,
            int(self.min_area),
        )

    def describe(self) -> str:
        return (
            f"OneBox(sg={self.sigma:g}, lo={self.t_lo:.4f}, "
            f"hi={self.t_hi:.4f}, s={self.s:g}, a={self.alpha:g})"
        )


Params = SingleCutParams | OneBoxParams


# ---------------------------------------------------------------------------
# method extractors
# ---------------------------------------------------------------------------
def singlecut_boxes(cache: LevelCache, params: SingleCutParams) -> np.ndarray:
    k = index_of(cache.thresholds, params.threshold)
    n = int(cache.counts[k])
    if n == 0:
        return np.zeros((0, 4), np.int32)
    b = cache.boxes[k, :n].astype(np.int32).copy()
    if params.min_area > 0:
        keep = cache.areas[k, :n] >= params.min_area
        b = b[keep]
    if params.alpha != 1.0:
        b = scale_boxes(b, params.alpha)
    return b


def onebox_boxes_with_scores(
    cache: LevelCache, params: OneBoxParams, score_kind: str = "peak"
) -> tuple[np.ndarray, np.ndarray]:
    """OneBox boxes **and** a per-box pseudo-confidence, aligned index-wise.

    OneBox is a gating rule, not a detector: it emits every component that
    survives the hysteresis gate at a *fixed* operating point, so it has no
    trained objectness score. Any AP-style metric nevertheless needs a ranked
    detection list, so one is derived from the quantity the gate already
    thresholds:

    * ``peak``  (default) — ``cpeak``, the max blurred-map value inside each
      surviving component's high-threshold core. It is *not* compared against
      ``t_hi`` to decide admission: the gate is **area**-based (the high-level
      core must exceed ``min_area``), and the thresholds snap to the shared
      grid, so a component whose peak sits between the two snapped levels still
      passes. ``cpeak`` is therefore a ranking key, not an admission test;
    * ``lpeak`` — the same at the low cut (the component's overall maximum);
    * ``mean``  — per-component mean map value, which is always ``None`` in the
      standard caches, so it falls back to ``peak`` with a clear error
      rather than silently ranking by something else.

    The score is **not calibrated** and must be described as a pseudo-confidence
    in any write-up: it ranks components within the chosen operating point and
    does not express a detection probability. It is not comparable to a trained
    detector's objectness score, only useful for ordering within one method.
    """
    if score_kind != "peak":
        raise ValueError(
            f"OneBox only exposes a 'peak' pseudo-confidence (got {score_kind!r}); "
            "'lpeak' and 'mean' are not available in the standard caches"
        )
    k_lo = index_of(cache.thresholds, params.t_lo)
    k_hi = index_of(cache.thresholds, params.t_hi)
    if k_lo == k_hi:
        # degenerate: no gate, so fall back to the single low cut and score the
        # surviving components by their peaks at that level
        k = k_lo
        n = int(cache.counts[k])
        boxes = singlecut_boxes(
            cache,
            SingleCutParams(
                float(cache.thresholds[k]), cache.sigma, params.alpha, params.min_area
            ),
        )
        if n == 0:
            return boxes, np.zeros(0, np.float64)
        keep = (
            cache.areas[k, :n] >= params.min_area
            if params.min_area > 0
            else np.ones(n, dtype=bool)
        )
        return boxes, cache.peaks[k, :n][keep].astype(np.float64)
    if k_lo > k_hi:
        k_lo, k_hi = k_hi, k_lo
    key = (k_lo, k_hi, params.min_area, GATE_PEAK_RATIO)
    parts = cache.gated.get(key)
    if parts is None:
        gidx, core, cpeak, lpeak = gated_parts_r0(
            cache.boxes, cache.counts, cache.areas, cache.peaks, k_lo, k_hi, params.min_area
        )
        gL = (
            cache.boxes[k_lo, gidx].astype(np.int32)
            if gidx.shape[0]
            else np.zeros((0, 4), np.int32)
        )
        parts = (gL, core, cpeak, lpeak)
        cache.gated[key] = parts
    gL, core, cpeak = parts[0], parts[1], parts[2]
    if gL.shape[0] == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, np.float64)
    b = mid_box(gL, core, params.s)
    if params.alpha != 1.0:
        b = scale_boxes(b, params.alpha)
    return b, np.asarray(cpeak, dtype=np.float64)


def onebox_boxes(cache: LevelCache, params: OneBoxParams) -> np.ndarray:
    k_lo = index_of(cache.thresholds, params.t_lo)
    k_hi = index_of(cache.thresholds, params.t_hi)
    if k_lo == k_hi:
        # degenerate: no gate; fall back to the single low cut
        return singlecut_boxes(
            cache,
            SingleCutParams(
                float(cache.thresholds[k_lo]), cache.sigma, params.alpha, params.min_area
            ),
        )
    if k_lo > k_hi:
        k_lo, k_hi = k_hi, k_lo
    key = (k_lo, k_hi, params.min_area, GATE_PEAK_RATIO)
    parts = cache.gated.get(key)
    if parts is None:
        gidx, core, cpeak, lpeak = gated_parts_r0(
            cache.boxes, cache.counts, cache.areas, cache.peaks, k_lo, k_hi, params.min_area
        )
        gL = (
            cache.boxes[k_lo, gidx].astype(np.int32)
            if gidx.shape[0]
            else np.zeros((0, 4), np.int32)
        )
        parts = (gL, core, cpeak, lpeak)
        cache.gated[key] = parts
    gL, core = parts[0], parts[1]
    if gL.shape[0] == 0:
        return np.zeros((0, 4), np.int32)
    b = mid_box(gL, core, params.s)
    if params.alpha != 1.0:
        b = scale_boxes(b, params.alpha)
    return b


def boxes_for_with_scores(
    method: Literal["singlecut", "onebox"],
    cache: LevelCache,
    params: Params,
    score_kind: str = "peak",
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes plus their pseudo-confidence for either method.

    SingleCut degrades to the per-component peak at its single threshold, so the
    same AP machinery can consume a legacy artifact without special-casing.
    """
    if method == "singlecut":
        assert isinstance(params, SingleCutParams)
        boxes = singlecut_boxes(cache, params)
        k = index_of(cache.thresholds, params.threshold)
        n = int(cache.counts[k])
        if n == 0 or boxes.shape[0] == 0:
            return boxes, np.zeros(boxes.shape[0], np.float64)
        keep = (
            cache.areas[k, :n] >= params.min_area
            if params.min_area > 0
            else np.ones(n, dtype=bool)
        )
        return boxes, cache.peaks[k, :n][keep].astype(np.float64)
    if method == "onebox":
        assert isinstance(params, OneBoxParams)
        return onebox_boxes_with_scores(cache, params, score_kind=score_kind)
    raise ValueError(f"unknown method {method!r}")


def boxes_for(
    method: Literal["singlecut", "onebox"], cache: LevelCache, params: Params
) -> np.ndarray:
    if method == "singlecut":
        assert isinstance(params, SingleCutParams)
        return singlecut_boxes(cache, params)
    if method == "onebox":
        assert isinstance(params, OneBoxParams)
        return onebox_boxes(cache, params)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# val statistics container
# ---------------------------------------------------------------------------
@dataclass
class ValStats:
    """Per-config per-finding val sums.

    ``sum_det[c, ab, j]``   sum over instances of the detection indicator at
                            val score ``j`` (float64)
    ``sum_missed[c, ab, j]`` sum over instances of missed boxes at score ``j``
    ``sum_boxes[c, ab]``    sum over instances of predicted box counts
    ``n_inst[ab]``          number of val instances for the finding
    ``configs``             the config list, aligned with axis 0
    """

    method: str
    configs: list[Params]
    findings: list[int]
    scores: np.ndarray
    sum_det: np.ndarray
    sum_missed: np.ndarray
    sum_boxes: np.ndarray
    n_inst: dict[int, int]
    timing: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.finding_pos = {ab: i for i, ab in enumerate(self.findings)}
        self.n_cfg = len(self.configs)

    # -- convenience metrics -------------------------------------------------
    def mean_det(self) -> np.ndarray:
        """(n_cfg, n_findings) mean detection over the 5 val scores."""
        return (
            self.sum_det.mean(axis=2)
            / np.asarray([self.n_inst[ab] for ab in self.findings], dtype=np.float64)[None, :]
        )

    def mean_missed(self) -> np.ndarray:
        return (
            self.sum_missed.mean(axis=2)
            / np.asarray([self.n_inst[ab] for ab in self.findings], dtype=np.float64)[None, :]
        )

    def mean_boxes(self) -> np.ndarray:
        return (
            self.sum_boxes
            / np.asarray([self.n_inst[ab] for ab in self.findings], dtype=np.float64)[None, :]
        )


# ---------------------------------------------------------------------------
# config-space builders
# ---------------------------------------------------------------------------
def threshold_pairs(levels: Sequence[float]) -> list[tuple[float, float]]:
    lv = sorted({float(v) for v in levels})
    return [(lo, hi) for i, lo in enumerate(lv) for hi in lv[i + 1 :]]


def onebox_configs(
    sigmas: Sequence[float],
    t_lo_levels: Sequence[float],
    t_hi_levels: Sequence[float] | None = None,
    s_grid: Sequence[float] = ONEBOX_S_GRID,
    alpha_grid: Sequence[float] = ONEBOX_ALPHA_GRID,
    min_area: int = GATE_AREA,
    pairs: Sequence[tuple[float, float]] | None = None,
) -> list[OneBoxParams]:
    """Cartesian OneBox grid (deterministic order: sigma, pair, s, alpha)."""
    if t_hi_levels is None:
        t_hi_levels = t_lo_levels
    if pairs is None:
        lo_lv = sorted({float(v) for v in t_lo_levels})
        hi_lv = sorted({float(v) for v in t_hi_levels})
        pairs = [(lo, hi) for lo in lo_lv for hi in hi_lv if lo < hi]
    cfg: list[OneBoxParams] = []
    for sigma in sigmas:
        for lo, hi in pairs:
            for s in s_grid:
                for alpha in alpha_grid:
                    cfg.append(
                        OneBoxParams(
                            sigma=float(sigma),
                            t_lo=float(lo),
                            t_hi=float(hi),
                            s=float(s),
                            alpha=float(alpha),
                            min_area=int(min_area),
                        )
                    )
    return cfg


def uniform_levels(step: float = 0.05, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return np.ascontiguousarray(np.round(np.linspace(lo, hi, n), 6))


def relative_sigma_grid(
    size: int, max_rel: float = SIGMA_REL_MAX, ref_size: float = SIGMA_REL_REF
) -> list[float]:
    """Preferred sigma candidates for an image ``size`` (pixels).

    Returns ``k / ref_size * size`` for ``k = 0..max_rel``; at the reference
    size (224) this is the pixel grid ``{0..4}``, at 512 it spans 0..9.14 px.
    The wider ``{0..6}`` range is rejected.
    """
    n = int(round(max_rel))
    return [k / ref_size * size for k in range(n + 1)]


# ---------------------------------------------------------------------------
# selection strategies (verbatim reference semantics)
# ---------------------------------------------------------------------------
class JointArgmax:
    """Pick the config maximising the aggregate val5 criterion.

    ``cap``: optional val5 AFP budget expressed as a multiple of the
    standard per-finding val5 AFP.  When set, only configs with
    ``afp <= cap * production_afp`` are eligible (infeasible findings fall
    back to the unconstrained argmax).  Ties break on fewer boxes then lower
    config index (the deterministic reference convention).
    """

    name = "joint_argmax"

    def __init__(self, cap: float | None = None, production_afp: dict[int, float] | None = None):
        self.cap = cap
        self.production_afp = production_afp

    def select(self, stats: ValStats) -> dict[int, Params]:
        acc = stats.mean_det()  # (n_cfg, n_findings)
        afp = stats.mean_missed()
        boxes = stats.mean_boxes()
        out: dict[int, Params] = {}
        for ab in stats.findings:
            fi = stats.finding_pos[ab]
            score = acc[:, fi]
            if self.cap is not None and self.production_afp is not None:
                budget = self.cap * self.production_afp[ab]
                ok = afp[:, fi] <= budget
                if ok.any():
                    score = np.where(ok, score, -np.inf)
            order = np.lexsort((np.arange(stats.n_cfg), boxes[:, fi], -score))
            out[ab] = stats.configs[int(order[0])]
        return out


class MeanArgmax:
    """Legacy strategy: per-score argmax config, then average its parameters.

    For SingleCut this is exactly the standard rule (mean of the five best
    thresholds, rounded to 2 dp).  For OneBox each parameter is averaged
    independently and snapped back to the nearest cached level (``t_lo`` is
    forced below ``t_hi`` by swapping if the average inverts them).  Kept for
    reference parity; never the default selection.
    """

    name = "mean_argmax"

    def __init__(
        self,
        sigma_grid: Sequence[float] | None = None,
        thresholds: np.ndarray = THR01,
        s_grid: Sequence[float] | None = None,
        alpha_grid: Sequence[float] | None = None,
        digits: int | None = None,
        box_tie_break: bool = False,
    ):
        self.sigma_grid = np.asarray(sigma_grid) if sigma_grid is not None else None
        self.thresholds = np.asarray(thresholds)
        self.s_grid = np.asarray(s_grid) if s_grid is not None else None
        self.alpha_grid = np.asarray(alpha_grid) if alpha_grid is not None else None
        self.digits = digits
        # Legacy standard takes the first maximal threshold (lowest value);
        # the reference M1 selection ties on fewer boxes instead.
        self.box_tie_break = box_tie_break

    def _snap(self, values: np.ndarray, value: float) -> float:
        return float(values[int(np.argmin(np.abs(values - value)))])

    def select(self, stats: ValStats) -> dict[int, Params]:
        out: dict[int, Params] = {}
        n_scores = stats.sum_det.shape[2]
        for ab in stats.findings:
            fi = stats.finding_pos[ab]
            picks: list[int] = []
            for j in range(n_scores):
                if self.box_tie_break:
                    order = np.lexsort(
                        (np.arange(stats.n_cfg), stats.sum_boxes[:, fi], -stats.sum_det[:, fi, j])
                    )
                else:
                    # first-max: ascending config order (SingleCut legacy)
                    order = np.argsort(-stats.sum_det[:, fi, j], kind="stable")
                picks.append(int(order[0]))
            chosen = [stats.configs[i] for i in picks]
            if stats.method == "singlecut":
                ts = [c.threshold for c in chosen]
                t = float(np.mean(ts))
                if self.digits is not None:
                    t = float(np.round(t, self.digits))
                out[ab] = SingleCutParams(
                    threshold=self._snap(self.thresholds, t),
                    sigma=float(np.mean([c.sigma for c in chosen])),
                    alpha=float(np.mean([c.alpha for c in chosen])),
                    min_area=int(round(np.mean([c.min_area for c in chosen]))),
                )
            else:
                t_lo = float(np.mean([c.t_lo for c in chosen]))
                t_hi = float(np.mean([c.t_hi for c in chosen]))
                if t_lo >= t_hi:
                    t_lo, t_hi = t_hi - 1e-6, t_lo
                sigma = float(np.mean([c.sigma for c in chosen]))
                if self.sigma_grid is not None:
                    sigma = self._snap(self.sigma_grid, sigma)
                s = float(np.mean([c.s for c in chosen]))
                if self.s_grid is not None:
                    s = self._snap(self.s_grid, s)
                alpha = float(np.mean([c.alpha for c in chosen]))
                if self.alpha_grid is not None:
                    alpha = self._snap(self.alpha_grid, alpha)
                out[ab] = OneBoxParams(
                    sigma=sigma,
                    t_lo=self._snap(self.thresholds, t_lo),
                    t_hi=self._snap(self.thresholds, t_hi),
                    s=s,
                    alpha=alpha,
                    min_area=int(round(np.mean([c.min_area for c in chosen]))),
                )
        return out


# ---------------------------------------------------------------------------
# literal per-config accumulator (parity reference / fallback)
# ---------------------------------------------------------------------------
def _accumulate_singlecut(
    instances_by_ab: dict[int, list[Instance]],
    configs: list[SingleCutParams],
    sigmas_used: Sequence[float],
    thresholds: np.ndarray,
) -> tuple:
    """Literal per-config SingleCut accumulation (reference / fallback).

    Literal accumulation: per-finding
    ``(sum_det, sum_missed, sum_boxes, n_inst)`` with the standard
    empty-map rule.  The standard default SingleCut selection is built from
    the raw 91-threshold tables; this is the exact-path fallback if
    that adapter ever disagrees with the reference selection.  ``sigmas_used``
    is unused in the reference body and kept only for signature parity.
    """
    findings = sorted(instances_by_ab)
    n_cfg = len(configs)
    n_score = SCORES.shape[0]
    sum_det = np.zeros((n_cfg, len(findings), n_score), np.float64)
    sum_missed = np.zeros((n_cfg, len(findings), n_score), np.float64)
    sum_boxes = np.zeros((n_cfg, len(findings)), np.float64)
    n_inst = {ab: len(instances_by_ab[ab]) for ab in findings}

    # config plan: level index -> list of (cfg_idx, alpha, min_area)
    plan: dict[tuple[float, int], list[tuple[int, float, int]]] = defaultdict(list)
    for ci, cfg in enumerate(configs):
        k = index_of(thresholds, cfg.threshold)
        plan[(cfg.sigma, k)].append((ci, cfg.alpha, cfg.min_area))

    for fi, ab in enumerate(findings):
        for inst in instances_by_ab[ab]:
            cache_by_sigma: dict[float, InstanceCache] = {}
            for (sigma, k), entries in plan.items():
                ic = cache_by_sigma.get(sigma)
                if ic is None:
                    ic = InstanceCache(inst, thresholds)
                    cache_by_sigma[sigma] = ic
                lc = ic.level(sigma)
                n = int(lc.counts[k])
                if n == 0:
                    for ci, _alpha, _ma in entries:
                        sum_missed[ci, fi, :] += 1.0  # standard empty-map rule
                    continue
                base = lc.boxes[k, :n].astype(np.int32)
                area = lc.areas[k, :n]
                for ci, alpha, min_area in entries:
                    b = base
                    if min_area > 0:
                        b = b[area >= min_area]
                    if b.shape[0] == 0:
                        sum_missed[ci, fi, :] += 1.0
                        continue
                    if alpha != 1.0:
                        b = scale_boxes(b, alpha)
                    ious = iou_vs_gt(b, inst.bbox)
                    nd = (ious[None, :] >= SCORES[:, None]).sum(axis=1)
                    sum_det[ci, fi, :] += nd > 0
                    sum_missed[ci, fi, :] += b.shape[0] - nd
                    sum_boxes[ci, fi] += b.shape[0]
    return sum_det, sum_missed, sum_boxes, n_inst


def build_onebox_plan(
    configs: Sequence[OneBoxParams],
    thresholds: np.ndarray = THR05,
    s_grid: Sequence[float] = ONEBOX_S_GRID,
    alpha_grid: Sequence[float] = ONEBOX_ALPHA_GRID,
) -> dict[tuple[float, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per ``(sigma, k_lo, k_hi)`` -> ``(s_idx, alpha_idx, cfg_idx)`` arrays.

    Verbatim bookkeeping step of the reference ``_accumulate_onebox`` body
    (thresholds/s/alpha snapped with ``argmin``, deterministic insertion
    order).
    """
    s_arr = np.asarray(s_grid, dtype=np.float64)
    a_arr = np.asarray(alpha_grid, dtype=np.float64)
    tmp: dict[tuple[float, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for ci, cfg in enumerate(configs):
        si = int(np.argmin(np.abs(s_arr - cfg.s)))
        ai = int(np.argmin(np.abs(a_arr - cfg.alpha)))
        k_lo = index_of(thresholds, cfg.t_lo)
        k_hi = index_of(thresholds, cfg.t_hi)
        tmp[(float(cfg.sigma), k_lo, k_hi)].append((si, ai, ci))
    plan: dict[tuple[float, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, entries in tmp.items():
        plan[key] = (
            np.fromiter((e[0] for e in entries), dtype=np.intp, count=len(entries)),
            np.fromiter((e[1] for e in entries), dtype=np.intp, count=len(entries)),
            np.fromiter((e[2] for e in entries), dtype=np.intp, count=len(entries)),
        )
    return plan


def accumulate_onebox_instance(
    heatmap: np.ndarray,
    bbox: dict,
    configs: Sequence[OneBoxParams],
    thresholds: np.ndarray = THR05,
    s_grid: Sequence[float] = ONEBOX_S_GRID,
    alpha_grid: Sequence[float] = ONEBOX_ALPHA_GRID,
    min_area: int = GATE_AREA,
) -> tuple:
    """Per-config val5 sums for one (instance, finding): literal reference body.

    Returns ``(sum_det, sum_missed, sum_boxes, sum_det_iobb,
    sum_missed_iobb)`` where ``sum_det`` / ``sum_missed`` /
    ``sum_det_iobb`` / ``sum_missed_iobb`` are ``(n_cfg, 5)`` float64 and
    ``sum_boxes`` is ``(n_cfg,)`` float64.  The IoU fields are the reference
    ``_accumulate_onebox`` arithmetic; the IoBB fields are the standard
    addition.  The plan groups configs by ``(sigma, k_lo,
    k_hi)`` only, exactly like the reference body: every config must share the
    same ``min_area`` (the frozen search space uses ``GATE_AREA``).
    """
    instance = Instance("", -1, heatmap, bbox)
    n_cfg = len(configs)
    n_score = SCORES.shape[0]
    sum_det = np.zeros((n_cfg, n_score), np.float64)
    sum_missed = np.zeros((n_cfg, n_score), np.float64)
    sum_boxes = np.zeros(n_cfg, np.float64)
    sum_det_iobb = np.zeros((n_cfg, n_score), np.float64)
    sum_missed_iobb = np.zeros((n_cfg, n_score), np.float64)

    s_arr = np.asarray(s_grid, dtype=np.float64)
    a_arr = np.asarray(alpha_grid, dtype=np.float64)
    plan = build_onebox_plan(configs, thresholds, s_arr, a_arr)

    ic = InstanceCache(instance, thresholds)
    for (sigma, k_lo, k_hi), (si_arr, ai_arr, ci_arr) in plan.items():
        lc = ic.level(sigma)
        gidx, core, _cp, _lp = gated_parts_r0(
            lc.boxes, lc.counts, lc.areas, lc.peaks, k_lo, k_hi, min_area
        )
        gL = lc.boxes[k_lo, gidx].astype(np.int32) if gidx.shape[0] else np.zeros((0, 4), np.int32)
        if gL.shape[0] == 0:
            # standard empty-map semantics: one missed box in both families
            np.add.at(sum_missed, ci_arr, 1.0)
            np.add.at(sum_missed_iobb, ci_arr, 1.0)
            continue
        n_g = gL.shape[0]
        mid = mid_boxes_batch(gL, core, s_arr)  # (n_s, n_g, 4)
        # scale for every alpha once
        scaled = np.empty((len(a_arr), len(s_arr), n_g, 4), dtype=np.int32)
        for ai, alpha in enumerate(a_arr):
            scaled[ai] = scale_boxes(mid.reshape(-1, 4), alpha).reshape(len(s_arr), n_g, 4)
        flat = scaled.reshape(-1, 4)
        ious = iou_vs_gt(flat, bbox).reshape(len(a_arr), len(s_arr), n_g)
        nd = (ious[..., None] >= SCORES).sum(axis=2)  # (n_a, n_s, n_scores)
        np.add.at(sum_det, ci_arr, nd[ai_arr, si_arr, :] > 0)
        np.add.at(sum_missed, ci_arr, n_g - nd[ai_arr, si_arr, :])
        np.add.at(sum_boxes, ci_arr, n_g)

        iobb = iobb_vs_gt(flat, bbox).reshape(len(a_arr), len(s_arr), n_g)
        nd_iobb = (iobb[..., None] >= SCORES).sum(axis=2)
        np.add.at(sum_det_iobb, ci_arr, nd_iobb[ai_arr, si_arr, :] > 0)
        np.add.at(sum_missed_iobb, ci_arr, n_g - nd_iobb[ai_arr, si_arr, :])
    return sum_det, sum_missed, sum_boxes, sum_det_iobb, sum_missed_iobb


# ---------------------------------------------------------------------------
# Batched hot path (the vectorised form of the reference implementation above)
# ---------------------------------------------------------------------------
@dataclass
class BatchPlan:
    """Per-sigma groups + scatter tables, built once per evaluation.

    ``groups[k]`` is the list of ``(k_lo, k_hi, (s_idx, alpha_idx, cfg_idx))``
    entries for ``sigmas[k]``; ``scatters[k]`` is the
    ``(n_pairs, n_s, n_a) -> cfg index or -1`` table, ``masks[k]`` its flat
    ``>= 0`` mask and ``scf[k]`` the masked flat config indices.  The grids
    are stored so workers do not depend on module globals.
    """

    sigmas: list[float]
    groups: list[list]
    scatters: list[np.ndarray]
    masks: list[np.ndarray]
    scf: list[np.ndarray]
    n_cfg: int
    n_s: int
    n_a: int
    s_grid: np.ndarray
    alpha_grid: np.ndarray
    #: threshold level set the ``(k_lo, k_hi)`` indices refer to
    thresholds: np.ndarray


def build_batch_plan(
    plan: dict,
    n_cfg: int,
    s_grid: Sequence[float] = ONEBOX_S_GRID,
    alpha_grid: Sequence[float] = ONEBOX_ALPHA_GRID,
    thresholds: np.ndarray = THR05,
) -> BatchPlan:
    """Build the per-sigma scatter tables from a ``build_onebox_plan`` dict."""
    s_arr = np.asarray(s_grid, dtype=np.float64)
    a_arr = np.asarray(alpha_grid, dtype=np.float64)
    n_s, n_a = int(s_arr.shape[0]), int(a_arr.shape[0])

    by_sigma: dict[float, list] = defaultdict(list)
    for (sigma, k_lo, k_hi), entries in plan.items():
        by_sigma[float(sigma)].append((k_lo, k_hi, entries))

    sigmas: list[float] = []
    groups: list[list] = []
    scatters: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scf: list[np.ndarray] = []
    for sigma in sorted(by_sigma):
        grp = by_sigma[sigma]
        n_pairs = len(grp)
        sc = np.full((n_pairs, n_s, n_a), -1, np.int64)
        for pi, (_k_lo, _k_hi, entries) in enumerate(grp):
            si_arr, ai_arr, ci_arr = entries
            flat_slots = si_arr * n_a + ai_arr
            if np.unique(flat_slots).shape[0] != flat_slots.shape[0]:
                raise ValueError(
                    "duplicate (sigma, threshold-pair, s, alpha) config: "
                    "config indices must be unique per group"
                )
            if np.any(sc[pi, si_arr, ai_arr] >= 0):
                raise ValueError("duplicate (sigma, threshold-pair, s, alpha) config")
            sc[pi, si_arr, ai_arr] = ci_arr
        flat = sc.reshape(-1)
        m = flat >= 0
        sigmas.append(sigma)
        groups.append(grp)
        scatters.append(sc)
        masks.append(m)
        scf.append(flat[m])
    return BatchPlan(
        sigmas=sigmas,
        groups=groups,
        scatters=scatters,
        masks=masks,
        scf=scf,
        n_cfg=n_cfg,
        n_s=n_s,
        n_a=n_a,
        s_grid=s_arr,
        alpha_grid=a_arr,
        thresholds=np.ascontiguousarray(thresholds, dtype=np.float64),
    )


def _batched_geometry(
    gL: np.ndarray,
    core: np.ndarray,
    gre,
    grs,
    gce,
    gcs,
    gt_area,
    s_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """mid -> scale(all alphas) -> IoU + IoBB for all components at once.

    Arithmetic is exactly ``mid_boxes_batch`` followed by ``scale_boxes`` per
    alpha (the prototype's verified formulation).

    The GT may be a single box (scalars, the historical call) or several boxes
    (arrays, ``multi_box=True``), in which case each predicted box takes its
    **best** IoU/IoBB over the GT boxes. With one GT box the reduction is over a
    length-1 axis and the arithmetic is bit-identical to the historical path, so
    single-box datasets cannot move.
    """
    coref = core.astype(np.float64)
    gLf = gL.astype(np.float64)
    mid = np.rint(coref[None] + s_grid[:, None, None] * (gLf - coref)[None]).astype(np.int32)
    mid[..., 0] = np.clip(mid[..., 0], 0, None)
    mid[..., 2] = np.clip(mid[..., 2], 0, None)
    rs, re, cs, ce = mid[..., 0], mid[..., 1], mid[..., 2], mid[..., 3]
    rc = (rs + re) / 2.0
    cc = (cs + ce) / 2.0
    hh = ((re - rs) / 2.0)[None] * alpha_grid[:, None, None]
    hw = ((ce - cs) / 2.0)[None] * alpha_grid[:, None, None]
    out = np.empty((alpha_grid.shape[0],) + mid.shape, np.int32)
    out[..., 0] = np.clip(np.rint(rc[None] - hh), 0, None)
    out[..., 1] = np.rint(rc[None] + hh)
    out[..., 2] = np.clip(np.rint(cc[None] - hw), 0, None)
    out[..., 3] = np.rint(cc[None] + hw)
    ors, ore, ocs, oce = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
    pred_area = (ore - ors) * (oce - ocs)

    def _per_gt(grs_i, gre_i, gcs_i, gce_i, gt_area_i):
        ih = np.maximum(0.0, np.minimum(ore, gre_i) - np.maximum(ors, grs_i))
        iw = np.maximum(0.0, np.minimum(oce, gce_i) - np.maximum(ocs, gcs_i))
        inter = ih * iw
        union = gt_area_i + pred_area - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.where(union > 0, inter / union, 0.0)
            iobb = np.where(pred_area > 0, inter / pred_area, 0.0)
        return ious, iobb

    # A single GT box keeps the historical scalar arithmetic exactly; several
    # boxes are reduced with an elementwise max (best match per prediction).
    scale = np.broadcast_shapes(np.shape(grs), np.shape(gre))
    if not scale:
        return _per_gt(grs, gre, gcs, gce, gt_area)

    def _iter_gt():
        yield from zip(
            np.atleast_1d(grs), np.atleast_1d(gre),
            np.atleast_1d(gcs), np.atleast_1d(gce), np.atleast_1d(gt_area),
        )

    ious = iobb = None
    for grs_i, gre_i, gcs_i, gce_i, gt_area_i in _iter_gt():
        cur_ious, cur_iobb = _per_gt(grs_i, gre_i, gcs_i, gce_i, gt_area_i)
        ious = cur_ious if ious is None else np.maximum(ious, cur_ious)
        iobb = cur_iobb if iobb is None else np.maximum(iobb, cur_iobb)
    return ious, iobb


def accumulate_onebox_instance_batched(
    inst: Instance, batch_plan: BatchPlan, thresholds: np.ndarray | None = None
) -> tuple:
    """IoU + IoBB per-config val5 sums for one instance.

    Returns ``(sum_det, sum_missed, sum_boxes, sum_det_iobb,
    sum_missed_iobb)``: ``(n_cfg, 5)`` / ``(n_cfg,)`` float64.  Bit-identical
    to :func:`accumulate_onebox_instance` on the IoU fields (all partials are
    exact integer counts); the IoBB fields are the standard addition.
    ``thresholds`` defaults to the level set stored in the ``BatchPlan`` so the
    plan and the accumulator cannot use mismatched grids.
    """
    if thresholds is None:
        thresholds = batch_plan.thresholds
    bp = batch_plan
    n_s, n_a = bp.n_s, bp.n_a
    n_score = SCORES.shape[0]
    sum_det = np.zeros((bp.n_cfg, n_score), np.float64)
    sum_missed = np.zeros((bp.n_cfg, n_score), np.float64)
    sum_boxes = np.zeros(bp.n_cfg, np.float64)
    sum_det_iobb = np.zeros((bp.n_cfg, n_score), np.float64)
    sum_missed_iobb = np.zeros((bp.n_cfg, n_score), np.float64)

    # multi_box=True yields several GT boxes; the geometry below reduces each
    # prediction to its best match so the single-box case is unchanged.
    _gt = box_array(inst.bbox)
    grs, gre, gcs, gce = _gt[:, 0], _gt[:, 1], _gt[:, 2], _gt[:, 3]
    gt_area = np.maximum(0.0, gre - grs) * np.maximum(0.0, gce - gcs)
    if grs.shape[0] == 1:
        grs, gre, gcs, gce, gt_area = (
            float(grs[0]), float(gre[0]), float(gcs[0]), float(gce[0]), float(gt_area[0]),
        )

    ic = InstanceCache(inst, thresholds)
    for sigma, grp, sc, mask, scf in zip(bp.sigmas, bp.groups, bp.scatters, bp.masks, bp.scf):
        lc = ic.level(sigma)
        n_pairs = len(grp)
        n_g_pair = np.zeros(n_pairs, np.int64)
        gL_parts, core_parts = [], []
        for pi, (k_lo, k_hi, _entries) in enumerate(grp):
            gidx, core, _cp, _lp = gated_parts_r0(
                lc.boxes, lc.counts, lc.areas, lc.peaks, k_lo, k_hi, GATE_AREA
            )
            if gidx.shape[0]:
                gL_parts.append(lc.boxes[k_lo, gidx].astype(np.int32))
                core_parts.append(core)
                n_g_pair[pi] = gidx.shape[0]

        if gL_parts:
            gL = np.concatenate(gL_parts)
            core = np.concatenate(core_parts)
            ious, iobb = _batched_geometry(
                gL,
                core,
                gre,
                grs,
                gce,
                gcs,
                gt_area,
                bp.s_grid,
                bp.alpha_grid,
            )
            ok = ious[..., None] >= SCORES
            ok_iobb = iobb[..., None] >= SCORES
            nonempty = np.nonzero(n_g_pair > 0)[0]
            starts = np.concatenate(([0], np.cumsum(n_g_pair[nonempty])))[:-1]
            det_ne = np.add.reduceat(ok, starts, axis=2)
            det_ne_iobb = np.add.reduceat(ok_iobb, starts, axis=2)
            det_count = np.zeros((n_a, n_s, n_pairs, n_score), np.int64)
            det_count_iobb = np.zeros_like(det_count)
            det_count[..., nonempty, :] = det_ne
            det_count_iobb[..., nonempty, :] = det_ne_iobb
        else:
            det_count = np.zeros((n_a, n_s, n_pairs, n_score), np.int64)
            det_count_iobb = np.zeros_like(det_count)

        empty = (n_g_pair == 0)[None, None, :, None]
        det = det_count > 0
        missed = np.where(empty, 1.0, n_g_pair[None, None, :, None] - det_count)
        det_iobb = det_count_iobb > 0
        missed_iobb = np.where(empty, 1.0, n_g_pair[None, None, :, None] - det_count_iobb)
        boxes_flat = np.repeat(n_g_pair, n_s * n_a).astype(np.float64)

        # flatten with pair-major (pair, s, alpha) ordering to match the
        # scatter flat index
        det_flat = np.transpose(det, (2, 1, 0, 3)).reshape(-1, n_score)
        missed_flat = np.transpose(missed, (2, 1, 0, 3)).reshape(-1, n_score)
        det_iobb_flat = np.transpose(det_iobb, (2, 1, 0, 3)).reshape(-1, n_score)
        missed_iobb_flat = np.transpose(missed_iobb, (2, 1, 0, 3)).reshape(-1, n_score)
        sum_det[scf] += det_flat[mask]
        sum_missed[scf] += missed_flat[mask]
        sum_det_iobb[scf] += det_iobb_flat[mask]
        sum_missed_iobb[scf] += missed_iobb_flat[mask]
        sum_boxes[scf] += boxes_flat[mask]
    return sum_det, sum_missed, sum_boxes, sum_det_iobb, sum_missed_iobb


# ---------------------------------------------------------------------------
# fork-pool worker entry points
# ---------------------------------------------------------------------------
#: worker-local state filled once per pool worker by ``onebox_worker_init``
_WORKER_STATE: dict[str, Any] = {}


def onebox_worker_init(batch_plan: BatchPlan, thresholds: np.ndarray | None = None) -> None:
    """Pool initializer: install the config space + ``BatchPlan`` per worker.

    Also pins the worker's torch intra-op pool to one thread when torch is
    already loaded in the forked parent (it always is in the training path).
    This is belt-and-braces against the node-wedge thread oversubscription
    seen in an earlier whole-node job: callers should also cap
    OMP/MKL/OpenBLAS in the job environment and call
    ``th.set_num_threads(1)`` before forking, but a fit pool created outside
    that setup must not fan out to every core either.  The lookup never
    imports torch, so ``localization/onebox.py`` stays usable without it and
    standalone validation pools pay no import cost.
    """
    import sys

    th = sys.modules.get("torch")
    if th is not None and hasattr(th, "set_num_threads"):
        th.set_num_threads(1)
    _WORKER_STATE["batch_plan"] = batch_plan
    if thresholds is None:
        thresholds = batch_plan.thresholds
    _WORKER_STATE["thresholds"] = np.ascontiguousarray(thresholds, dtype=np.float64)


def onebox_partial_job(payload) -> tuple:
    """Pool task: a chunk of instances of ONE finding -> summed partials.

    ``payload`` is an iterable of ``(heatmap, bbox)`` pairs (or
    :class:`Instance` objects).  The worker uses the batched accumulator; the config space + ``BatchPlan`` live in module globals filled by
    :func:`onebox_worker_init`.  All partial values are exact integer counts,
    so chunking / worker grouping cannot change the sums.
    """
    bp: BatchPlan = _WORKER_STATE["batch_plan"]
    # the BatchPlan carries its own threshold levels; the state copy is kept
    # for explicit overrides only
    thresholds = _WORKER_STATE.get("thresholds", bp.thresholds)
    sums: tuple | None = None
    for item in payload:
        if isinstance(item, Instance):
            inst = item
        else:
            heatmap, bbox = item
            inst = Instance("", -1, heatmap, bbox)
        part = accumulate_onebox_instance_batched(inst, bp, thresholds)
        if sums is None:
            sums = tuple(np.array(a, dtype=np.float64) for a in part)
        else:
            for acc, p in zip(sums, part):
                acc += p
    if sums is None:  # defensive: empty chunk returns zeros, never NaN
        n_score = SCORES.shape[0]
        return (
            np.zeros((bp.n_cfg, n_score), np.float64),
            np.zeros((bp.n_cfg, n_score), np.float64),
            np.zeros(bp.n_cfg, np.float64),
            np.zeros((bp.n_cfg, n_score), np.float64),
            np.zeros((bp.n_cfg, n_score), np.float64),
        )
    return sums


def warmup_onebox() -> None:
    """Compile the numba kernels on a tiny dummy instance.

    Call once in the parent process before forking workers so children inherit
    the compiled code instead of compiling per worker.
    """
    hm = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    inst = Instance("warmup", -1, hm, {"row_start": 1, "row_end": 5, "col_start": 1, "col_end": 5})
    ic = InstanceCache(inst, THR05)
    lc = ic.level(0.0)
    gated_parts_r0(lc.boxes, lc.counts, lc.areas, lc.peaks, 0, 1, GATE_AREA)
    ic.level(2.0)
