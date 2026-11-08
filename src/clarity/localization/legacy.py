"""Stubs for the SingleCut evaluation path, which this release does not include.

The companion implements OneBox only.  These names exist so the
shared evaluator module imports cleanly; calling them raises clearly.
"""
from __future__ import annotations


def get_cam_heatmaps(*args, **kwargs):
    raise NotImplementedError("grad-CAM maps other than HiResCAM are not included in this release; use OneBox")


def get_mctfp_attention(*args, **kwargs):
    raise NotImplementedError("MCTformer+ attention maps are not included in this release")


def get_cam_fused_attention(*args, **kwargs):
    raise NotImplementedError("fused grad-CAM maps other than HiResCAM are not included in this release; use OneBox")


def _singlecut_calibration(*args, **kwargs):
    raise NotImplementedError("SingleCut evaluation is not included in this release; use OneBox")


def evaluate_model_localization_singlecut(*args, **kwargs):
    raise NotImplementedError("SingleCut evaluation is not included in this release; use OneBox")


def evaluate_model_localization_sweep_singlecut(*args, **kwargs):
    raise NotImplementedError("SingleCut evaluation is not included in this release; use OneBox")
