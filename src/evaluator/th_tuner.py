import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple

# from .on_blobs import compute_metrics, compute_average_iou
from . import on_blobs


def _get_threshold_min_fnfp(
    masks_gt,
    score_masks,
    score_thresholds,
    min_size: int = None,
    iou_threshold=0.0125,
) -> Tuple[dict, dict]:

    # Initialize variables to store the optimal values
    best_threshold = None
    min_fn = float("inf")
    min_fp = float("inf")

    is_score_thresholds_descending = is_sorted_descending(score_thresholds)

    for threshold in score_thresholds:
        masks_gt = on_blobs.threshold_score_masks(masks_gt, 0.5, min_size)
        # binarized_score_masks = on_blobs.threshold_score_masks(score_masks, threshold, None)
        binarized_score_masks = on_blobs.threshold_score_masks(
            score_masks, threshold, min_size
        )

        tp, tn, fp, fn = on_blobs.compute_metrics(
            masks_gt, binarized_score_masks, iou_threshold=iou_threshold
        )
        print(f"th: {threshold} tp: {tp} tn: {tn} fp: {fp} fn: {fn}")

        if fn < min_fn:
            min_fn = fn
            min_fp = fp
            best_threshold = threshold

        elif fn == min_fn and fp < min_fp:
            min_fp = fp
            best_threshold = threshold

        elif fn == min_fn and fp == min_fp and threshold > best_threshold:
            best_threshold = threshold

        if is_score_thresholds_descending and fn == 0:
            break

    print(f"th_min_fnfp: {best_threshold} min_fn: {min_fn} min_fp: {min_fp}")

    return best_threshold


def is_sorted_descending(arr):
    """Check if the given 1D array is sorted in descending order."""
    if arr.ndim != 1:
        raise ValueError("The input must be a 1D array.")

    return np.all(arr[:-1] >= arr[1:])


def _get_threshold_max_metric(
    masks_gt,
    score_masks,
    score_thresholds,
    min_size: int = None,
    metric_type: str = "iou",
) -> Tuple[dict, dict]:

    # Initialize variables to store the optimal values
    best_threshold = 0
    best_metric = -np.inf

    for threshold in score_thresholds:
        masks_gt = on_blobs.threshold_score_masks(masks_gt, 0.5, min_size)
        binarized_score_masks = on_blobs.threshold_score_masks(
            score_masks, threshold, None
        )

        if metric_type == "iou":
            metric = on_blobs.compute_average_iou(masks_gt, binarized_score_masks)
        elif metric_type == "l1_sim":
            metric = on_blobs.compute_average_lp_sim(masks_gt, binarized_score_masks)
        else:
            raise NotImplementedError()

        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold

        # print(f"th: {threshold} {metric_type}: {best_metric}")

    print(f"best th: {best_threshold} best {metric_type}: {best_metric}")

    return best_threshold


def tune_score_threshold(
    masks_gt,
    score_masks,
    score_thresholds,
    min_size: int = None,
    metric_type: str = "iou",
) -> Tuple[dict, dict]:
    masks_gt = np.array(masks_gt).astype(np.uint8)
    score_masks = np.array(score_masks)

    assert len(masks_gt) == len(score_masks)
    if score_masks.ndim == 3:
        num_classes = 1
    elif score_masks.ndim == 4:
        num_classes = score_masks.shape[-1] - 1
        raise NotImplementedError()
    else:
        raise ValueError()

    score_thresholds = np.sort(score_thresholds)[::-1]

    if metric_type in ["iou", "l1_sim"]:
        best_threshold = _get_threshold_max_metric(
            masks_gt=masks_gt,
            score_masks=score_masks,
            score_thresholds=score_thresholds,
            min_size=min_size,
            metric_type=metric_type,
        )
    elif metric_type == "fnfp":
        best_threshold = _get_threshold_min_fnfp(
            masks_gt=masks_gt,
            score_masks=score_masks,
            score_thresholds=score_thresholds,
            min_size=min_size,
        )
    else:
        raise NotImplementedError()

    return best_threshold
