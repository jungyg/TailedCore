import cv2
import functools
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from typing import Tuple, List, Union


def extract_blobs_binary(binary_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Extract connected components from a binary image mask.

    Args:
        binary_mask (np.ndarray): A binary image mask where regions to be labeled are white (1)
                                  and the background is black (0). Must be a 2D array of type np.uint8.

    Returns:
        Tuple[int, np.ndarray]: A tuple containing the number of unique connected components (excluding background)
                                and an array with the same size as `binary_mask` where each pixel's label is indicated.
    """
    binary_mask = binary_mask.astype(np.uint8)
    num_labels_with_bg, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    # num_labels_with_bg, labels = cv2.connectedComponents(binary_mask, connectivity=4)
    num_labels = num_labels_with_bg - 1
    return num_labels, labels


def extract_blobs(mask: np.ndarray, num_classes: int) -> Tuple[int, np.ndarray, dict]:
    """
    Extracts and classifies blobs from a mask for multiple classes.

    Args:
        mask (np.ndarray): 2D array with each pixel's value indicating its class (shape: [height, width]).
                           0 is background, and positive integer corresponds to an object class.
        num_classes (int): Number of distinct classes in the mask (excluding the background).

    Returns:
        Tuple[int, np.ndarray, dict]:
        - num_labels (int): Total number of unique blobs across all classes.
        - labels (np.ndarray): 2D array with unique labels for each blob (shape: [height, width]).
        - label2class (dict): Maps blob labels to their respective class.

    Processes each class in the mask separately to identify and label blobs. Labels are unique
    across different classes.
    """
    assert num_classes > 0
    num_labels = 0
    label2class = {}
    for c in range(1, num_classes + 1):
        binary_mask = mask == c
        num_blob_labels, blob_labels = extract_blobs_binary(binary_mask)

        for i in range(1, num_blob_labels + 1):
            label2class[i + num_labels] = c

        if c == 1:
            labels = blob_labels
        else:
            blob_labels[blob_labels != 0] += num_labels
            # assert np.sum(labels[blob_labels != 0]) == 0
            labels += blob_labels

        num_labels += num_blob_labels

    return num_labels, labels, label2class


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) of two masks.

    Args:
        mask1 (np.ndarray): First binary mask (2D array).
        mask2 (np.ndarray): Second binary mask (2D array).

    Returns:
        float: IoU score between mask1 and mask2. Returns 0 if the union is empty.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def compute_inclusivity(mask_in: np.ndarray, mask_base: np.ndarray) -> float:
    intersection = np.logical_and(mask_in, mask_base).sum()
    part = mask_in.sum()
    return intersection / part if part != 0 else 0


def compute_inclusive_iou(mask_in: np.ndarray, mask_base: np.ndarray) -> float:
    iou = compute_iou(mask_in, mask_base)
    inclusivity = compute_inclusivity(mask_in, mask_base)

    return max(iou, inclusivity)


def compute_average_iou(masks1, masks2):
    total_iou = 0
    count = 0
    for mask1, mask2 in zip(masks1, masks2):
        # Compute IoU
        iou = compute_iou(mask1, mask2)
        total_iou += iou
        count += 1

    average_iou = total_iou / count if count != 0 else 0
    return average_iou


def _compute_confusion_matrix(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    num_classes: int,
    iou_threshold: float,
    metric_type: str = "inclusive_iou",
) -> np.ndarray:
    """
    Computes a confusion matrix for a single pair of ground truth and predicted masks.

    Args:
        mask_gt (np.ndarray): Ground truth mask (2D array of integers). Each integer corresponds to a class (0 to background).
        mask_pred (np.ndarray): Predicted mask (2D array of integers). Each integer corresponds to a class (0 to background).
        num_classes (int): Number of classes excluding the background.
        iou_threshold (float): Threshold for IoU to consider a prediction as a true positive. It must be between 0 and 1.

    Returns:
        np.ndarray: Confusion matrix (2D array of shape [num_classes + 1, num_classes + 1]). Row for ground truth, and column for prediction.
    """

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    num_bloblabels_gt, bloblabels_gt, bloblabel2class_gt = extract_blobs(
        mask_gt, num_classes
    )
    num_bloblabels_pred, bloblabels_pred, bloblabel2class_pred = extract_blobs(
        mask_pred, num_classes
    )

    # todo: complete the below
    predlabel2gtlabel = defaultdict(list)
    gtlabel2prediou = defaultdict(list)

    ious = _compute_blob_ious(
        bloblabels_gt, bloblabels_pred, num_bloblabels_gt, num_bloblabels_pred
    )

    for i in range(1, num_bloblabels_gt + 1):
        max_iou = 0
        matching_pred_class = 0

        for j in range(1, num_bloblabels_pred + 1):
            # if metric_type == "iou":
            #     iou = compute_iou(
            #         bloblabels_gt == i, bloblabels_pred == j
            #     )  # iou between blob_pred and blob_gt
            # elif metric_type == "inclusive_iou":
            #     iou = compute_inclusive_iou(
            #         bloblabels_gt == i, bloblabels_pred == j
            #     )  # iou between blob_pred and blob_gt
            # else:
            #     raise NotImplementedError()

            iou = ious[i - 1, j - 1]

            if iou > 0:
                gtlabel2prediou[i].append(iou)
            if iou >= iou_threshold:
                predlabel2gtlabel[j].append((i, iou))
                if iou > max_iou:
                    matching_pred_class = bloblabel2class_pred[j]
                    max_iou = max(max_iou, iou)

        confusion_matrix[bloblabel2class_gt[i], matching_pred_class] += 1

    for j in range(1, num_bloblabels_pred + 1):
        if len(predlabel2gtlabel[j]) == 0:
            confusion_matrix[0, bloblabel2class_pred[j]] += 1

    if compute_iou(bloblabels_pred == 0, bloblabels_gt == 0) >= iou_threshold:
        confusion_matrix[0, 0] += 1

    return confusion_matrix


def _compute_blob_ious(
    bloblabels_gt, bloblabels_pred, num_bloblabels_gt=None, num_bloblabels_pred=None
):
    if num_bloblabels_gt is None:
        num_bloblabels_gt = len(np.unique(bloblabels_gt)) - 1
    if num_bloblabels_pred is None:
        num_bloblabels_pred = len(np.unique(bloblabels_pred)) - 1

    ious = np.empty((num_bloblabels_gt, num_bloblabels_pred))
    param_combinations = [
        (i, j)
        for i in range(1, num_bloblabels_gt + 1)
        for j in range(1, num_bloblabels_pred + 1)
    ]

    # Parallel computation
    results = Parallel(n_jobs=8)(
        delayed(assign_iou)(bloblabels_gt, i, bloblabels_pred, j)
        for i, j in param_combinations
    )

    # Fill the ious matrix
    for count, (i, j) in enumerate(param_combinations):
        ious[i - 1, j - 1] = results[count]

    return ious


def assign_iou(bloblabels_gt, i, bloblabels_pred, j, metric_type="inclusive_iou"):
    if metric_type == "iou":
        iou = compute_iou(
            bloblabels_gt == i, bloblabels_pred == j
        )  # iou between blob_pred and blob_gt
    elif metric_type == "inclusive_iou":
        iou = compute_inclusive_iou(
            bloblabels_gt == i, bloblabels_pred == j
        )  # iou between blob_pred and blob_gt
    else:
        raise NotImplementedError()
    return iou


def compute_confusion_matrix(
    masks_gt: Union[List[np.ndarray], np.ndarray],
    masks_pred: Union[List[np.ndarray], np.ndarray],
    num_classes: int,
    iou_threshold: float,
) -> np.ndarray:
    """
    Computes the confusion matrix for a set of ground truth and predicted masks.

    Args:
        masks_gt (Union[List[np.ndarray], np.ndarray]): Ground truth masks. Can be a list of 2D arrays
                                                        or a 3D array with shape (num_masks, height, width).
                                                        Each element should be a natural number (including 0),
                                                        where 0 indicates the background and positive integers
                                                        indicate object classes.
        masks_pred (Union[List[np.ndarray], np.ndarray]): Predicted masks, in the same format as masks_gt.
        num_classes (int): Number of classes, excluding the background.
        iou_threshold (float): IoU threshold for considering a prediction as a true positive.

    Returns:
        np.ndarray: Aggregate confusion matrix. A 2D array of shape [num_classes + 1, num_classes + 1],
                    where rows represent ground truth classes and columns represent predicted classes.

    The function computes a confusion matrix for each pair of corresponding ground truth and predicted masks.
    It sums these matrices to obtain an aggregate confusion matrix. This matrix is useful for evaluating the
    classification performance, with each cell [i, j] indicating the count of samples of ground truth class i
    predicted as class j.
    """

    assert len(np.unique(masks_gt)) <= num_classes + 1
    assert len(np.unique(masks_pred)) <= num_classes + 1

    masks_gt, masks_pred = np.array(masks_gt), np.array(masks_pred)

    assert masks_gt.shape == masks_pred.shape

    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for mask_gt, mask_pred in zip(masks_gt, masks_pred):
        confusion_matrix += _compute_confusion_matrix(
            mask_gt, mask_pred, num_classes, iou_threshold
        )
    return confusion_matrix


def compute_metrics(
    masks_gt: np.ndarray,
    masks_pred: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    assert masks_gt.shape == masks_pred.shape

    num_classes = len(np.unique(masks_gt)) - 1

    if num_classes == 0:
        num_classes = 1  # if no ones in masks_gt, then we assume it is binary (i.e. only single foreground class)

    confusion_matrix = compute_confusion_matrix(
        masks_gt, masks_pred, num_classes, iou_threshold=iou_threshold
    )

    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1:].sum()
    FN = confusion_matrix[1:, 0].sum()
    TP = np.diag(confusion_matrix)[1:].sum()

    return TP, TN, FP, FN


def remove_small_blobs(binary_masks, min_size):
    binary_masks = binary_masks.astype(np.uint8)

    processed_binary_masks = np.array(
        [_remove_small_blobs(binary_mask, min_size) for binary_mask in binary_masks]
    )
    return processed_binary_masks


def _remove_small_blobs(binary_mask: np.ndarray, min_size):
    assert isinstance(binary_mask, np.ndarray)
    assert binary_mask.ndim == 2

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, 8, cv2.CV_32S
    )

    # Filter out small blobs
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if width < min_size and height < min_size:
            binary_mask[labels == label] = 0

    return binary_mask


def compute_average_lp_sim(masks_gt, masks_pred, ord=1, classwise=True):
    assert classwise is True
    assert len(masks_gt) == len(masks_pred)

    sim = 0
    for mask_gt, mask_pred in zip(masks_gt, masks_pred):
        sim += -_compute_overall_lp_distance(mask_gt, mask_pred, ord)

    return sim


def _compute_overall_lp_distance(mask_gt, mask_pred, ord):
    class_idxes = np.unique(mask_gt)
    dist = 0
    for class_idx in class_idxes:
        mask = mask_gt == class_idx
        dist += _compute_lp_distance(
            np.asarray(mask_gt[mask], dtype=bool),
            np.asarray(mask_pred[mask], dtype=bool),
            ord,
        )

    return dist


def _compute_lp_distance(binary_mask_gt, binary_mask_pred, ord, pixel_average=True):
    if ord == 1:
        return _compute_bool_difference(
            binary_mask_gt, binary_mask_pred, pixel_average=True
        )

    diff = binary_mask_gt - binary_mask_pred
    dist = np.linalg.norm(diff, ord=ord)

    if pixel_average:
        dist /= len(diff.flatten())

    return dist


def _compute_bool_difference(binary_mask_gt, binary_mask_pred, pixel_average=True):
    # Compute the XOR to find positions where the masks differ
    diff = np.bitwise_xor(binary_mask_gt, binary_mask_pred)

    # Count the number of True values in the XOR array (sum of absolute differences)
    dist = np.sum(diff)

    if pixel_average:
        dist /= diff.size  # Use .size for the total number of elements in the array

    return dist


def threshold_score_masks(score_masks, th, min_size=None):
    binarized_score_masks = (score_masks >= th).astype(np.uint8)
    if min_size is not None:
        assert isinstance(min_size, int)
        binarized_score_masks = remove_small_blobs(binarized_score_masks, min_size)
    return binarized_score_masks
