"""Localization metric helpers (IoU, IoBB, MIR, AEuD).

Public, method-independent metric arithmetic shared by both evaluators.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _gt_boxes(gt_bbox) -> list[dict[str, int | float]]:
    """Normalise a GT box value (single dict, or a multi_box list) to a list."""
    if isinstance(gt_bbox, dict):
        return [gt_bbox]
    return [box for box in gt_bbox if box is not None]


def calculate_iou(gt_bbox, pred_bbox: dict[str, int | float]) -> float:
    """Calculate the IoU between the ground truth and predicted bounding boxes.

    Parameters:
        gt_bbox: A box dict, or a list of them for ``multi_box=True`` (RSNA
            allows several boxes for one finding). With several boxes the
            prediction scores its **best** match, which is the detection
            convention and degenerates to the historical value for one box.
        pred_bbox (dict[str, int | float]): A dictionary containing the row_start, row_end,
            col_start, and col_end of the predicted bounding box.

    Returns:
        float: IoU between the ground truth and predicted bounding box.
    """
    boxes = _gt_boxes(gt_bbox)
    if not boxes:
        return 0.0
    return max(_calculate_iou_one(box, pred_bbox) for box in boxes)


def _calculate_iou_one(gt_bbox: dict[str, int | float], pred_bbox: dict[str, int | float]) -> float:
    """IoU for exactly one GT box (the historical body)."""

    # extract bounding box coordinates into temp variables
    gt_row_start = gt_bbox["row_start"]
    gt_row_end = gt_bbox["row_end"]
    gt_col_start = gt_bbox["col_start"]
    gt_col_end = gt_bbox["col_end"]

    pred_row_start = pred_bbox["row_start"]
    pred_row_end = pred_bbox["row_end"]
    pred_col_start = pred_bbox["col_start"]
    pred_col_end = pred_bbox["col_end"]

    # calculate area of the ground truth bounding box
    gt_area = (gt_row_end - gt_row_start) * (gt_col_end - gt_col_start)

    # calculate area of the predicted bounding box
    pred_area = (pred_row_end - pred_row_start) * (pred_col_end - pred_col_start)

    # calculate intersection area of the two bounding boxes
    intersection_row_start = max(gt_row_start, pred_row_start)
    intersection_row_end = min(gt_row_end, pred_row_end)
    intersection_col_start = max(gt_col_start, pred_col_start)
    intersection_col_end = min(gt_col_end, pred_col_end)

    intersection_area = max(0, intersection_row_end - intersection_row_start) * max(
        0, intersection_col_end - intersection_col_start
    )

    # calculate union area
    union_area = gt_area + pred_area - intersection_area

    # calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou


def calculate_iobb(gt_bbox, pred_bbox: dict[str, int | float]) -> float:
    """Calculate the IoBB (precision) between the ground truth and predicted bounding boxes.

    Parameters:
        gt_bbox: A box dict, or a list of them for ``multi_box=True``; the
            prediction scores its best match (see :func:`calculate_iou`).
        pred_bbox (dict[str, int | float]): A dictionary containing the row_start, row_end,
            col_start, and col_end of the predicted bounding box.

    Returns:
        float: IoBB between the ground truth and predicted bounding box.
    """
    boxes = _gt_boxes(gt_bbox)
    if not boxes:
        return 0.0
    return max(_calculate_iobb_one(box, pred_bbox) for box in boxes)


def _calculate_iobb_one(gt_bbox: dict[str, int | float], pred_bbox: dict[str, int | float]) -> float:
    """IoBB for exactly one GT box (the historical body)."""

    # extract bounding box coordinates into temp variables
    gt_row_start = gt_bbox["row_start"]
    gt_row_end = gt_bbox["row_end"]
    gt_col_start = gt_bbox["col_start"]
    gt_col_end = gt_bbox["col_end"]

    pred_row_start = pred_bbox["row_start"]
    pred_row_end = pred_bbox["row_end"]
    pred_col_start = pred_bbox["col_start"]
    pred_col_end = pred_bbox["col_end"]

    # calculate area of the predicted bounding box
    pred_area = (pred_row_end - pred_row_start) * (pred_col_end - pred_col_start)

    # calculate intersection area of the two bounding boxes
    intersection_row_start = max(gt_row_start, pred_row_start)
    intersection_row_end = min(gt_row_end, pred_row_end)
    intersection_col_start = max(gt_col_start, pred_col_start)
    intersection_col_end = min(gt_col_end, pred_col_end)

    intersection_area = max(0, intersection_row_end - intersection_row_start) * max(
        0, intersection_col_end - intersection_col_start
    )

    # calculate the IoBB
    iobb = intersection_area / pred_area if pred_area != 0 else 0

    return iobb


def calculate_masked_intensity_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    # get masked intensity sum
    masked_intensity = heatmap[mask].sum()

    # get total intensity
    total_intensity = heatmap.sum()

    # calculate ratio
    masked_intensity_ratio = masked_intensity / total_intensity

    # handle zero-intensity heatmap
    if np.isnan(masked_intensity_ratio):
        return 0.0
    else:
        return masked_intensity_ratio


def calculate_average_euclidean_distance(heatmap: np.ndarray, mask: np.ndarray) -> float:
    # get dimensions
    height, width = mask.shape
    # calculate mean major image dimension
    mean_dim = (height + width) / 2

    # calculate distance map
    distance_map = distance_transform_edt(~mask)

    # element-wise multiply distance map with heatmap
    distance_weighted_heatmap = heatmap * distance_map

    # calculate average Euclidean distance, weighted by mean major image dimension
    AEuD = np.mean(distance_weighted_heatmap) / mean_dim

    # handle all-zero distances
    if np.isnan(AEuD):
        return 0.0

    return AEuD
