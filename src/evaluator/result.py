import os
import numpy as np
import pandas as pd

from . import th_tuner, metrics, on_blobs
from .. import utils


def save_result(
    image_paths,
    image_scores: np.ndarray,
    labels_gt,
    score_masks: np.ndarray,
    masks_gt,
    save_dir_path,
    num_ths=21,
    min_size=60,
):
    save_plot_dir = os.path.join(save_dir_path, "plot")

    # ths = np.around(
    #     np.linspace(np.min(score_masks), np.max(score_masks), num_ths),
    #     3)[1:-1]

    # th_max_l1_sim = th_tuner.tune_score_threshold(masks_gt,
    #                                               score_masks,
    #                                               score_thresholds=ths,
    #                                               metric_type="l1_sim")

    # th_max_iou = th_tuner.tune_score_threshold(masks_gt,
    #                                            score_masks,
    #                                            score_thresholds=ths,
    #                                            metric_type="iou")

    # th_upper_bound = ths[-1] if th_max_iou <= th_max_iou else th_max_iou

    # th_min_fnfp = th_tuner.tune_score_threshold(
    #     masks_gt,
    #     score_masks,
    #     score_thresholds=ths[(ths >= th_max_l1_sim) & (ths <= th_upper_bound)],
    #     min_size=min_size,
    #     metric_type="fnfp")

    # tp, tn, fp, fn = on_blobs.compute_metrics(
    #     masks_gt=on_blobs.threshold_score_masks(masks_gt, 0.5, min_size),
    #     masks_pred=on_blobs.threshold_score_masks(score_masks, th_min_fnfp,
    #                                               min_size),
    #     iou_threshold=0.0125)

    # print(
    #     f"final metrics - tp: {tp} tn: {tn} fp: {fp} fn {fn} - th: {th_min_fnfp}"
    # )

    # ths_tuned = {
    #     "th_max_iou": th_max_iou,
    #     "th_max_l1_sim": th_max_l1_sim,
    #     "th_min_fnfp": th_min_fnfp,
    # }

    # utils.plot_hist(
    #     score_masks,
    #     masks_gt,
    #     filename=os.path.join(save_plot_dir, "hist_pixel_scores_all.png"),
    #     other_points=ths_tuned,
    # )

    # utils.plot_hist(score_masks[masks_gt == 1],
    #                 filename=os.path.join(save_plot_dir,
    #                                       "hist_pixel_scores_anomaly.png"))

    # ths_raw = {
    #     f"th_{th:.3f}": th
    #     for th in ths[(ths >= th_max_l1_sim) & (ths <= th_max_iou)]
    # }

    # for th_name, th_val in {**ths_raw, **ths_tuned}.items():
    #     utils.plot_score_masks(
    #         save_dir_path=os.path.join(save_plot_dir, f"{th_name}_filtered"),
    #         image_paths=image_paths,
    #         masks_gt=on_blobs.threshold_score_masks(masks_gt, 0.5, min_size),
    #         score_masks=score_masks,
    #         image_scores=image_scores,
    #         binary_masks=on_blobs.threshold_score_masks(
    #             score_masks, th_val, min_size))

    #     utils.plot_score_masks(
    #         save_dir_path=os.path.join(save_plot_dir, f"{th_name}"),
    #         image_paths=image_paths,
    #         masks_gt=masks_gt,
    #         score_masks=score_masks,
    #         image_scores=image_scores,
    #         binary_masks=on_blobs.threshold_score_masks(
    #             score_masks, th_val, None))

    # utils.plot_score_masks(
    #     save_dir_path=os.path.join(save_plot_dir, "scores"),
    #     image_paths=image_paths,
    #     masks_gt=masks_gt,
    #     score_masks=score_masks,
    #     image_scores=image_scores,
    # )
    utils.plot_mvtec_score_masks(
        save_dir_path=os.path.join(save_plot_dir, "scores"),
        image_paths=image_paths,
        masks_gt=masks_gt,
        score_masks=score_masks,
    )

    is_loco = 'loco-multiclass' in save_dir_path

    try:
        print("Computing image auroc...")
        image_auroc = metrics.compute_imagewise_retrieval_metrics(
            image_scores, labels_gt
        )["auroc"]
        print("Computing pixel auroc...")
        pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
            score_masks, masks_gt
        )["auroc"]

        if is_loco:
            good_idx = [i for i,s in enumerate(image_paths) if 'good' in s]
            # good_idx = [1 if 'good' in path else 0 for path in image_paths]
            good_scores = [image_scores[x] for x in good_idx]
            good_labels = [labels_gt[x] for x in good_idx]
            good_segs = [score_masks[x] for x in good_idx]
            good_masks = [masks_gt[x] for x in good_idx]

            ## logical
            logical_idx = [i for i,s in enumerate(image_paths) if 'logical_anomalies' in s]
            # logical_idx = [1 if 'logical_anomalies' in path else 0 for path in image_paths]
            logical_scores = [image_scores[x] for x in logical_idx] + good_scores
            logical_labels = [labels_gt[x] for x in logical_idx] + good_labels
            logical_segs = [score_masks[x] for x in logical_idx] + good_segs
            logical_masks = [masks_gt[x] for x in logical_idx] + good_masks
            logical_image_auroc = metrics.compute_imagewise_retrieval_metrics(
                logical_scores, logical_labels
            )["auroc"]
            logical_pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
                logical_segs, logical_masks
            )["auroc"]

            ## structural
            structural_idx = [i for i,s in enumerate(image_paths) if 'structural_anomalies' in s]
            # structural_idx = [1 if 'structural_anomalies' in path else 0 for path in image_paths]
            structural_scores = [image_scores[x] for x in structural_idx] + good_scores
            structural_labels = [labels_gt[x] for x in structural_idx] + good_labels
            structural_segs = [score_masks[x] for x in structural_idx] + good_segs
            structural_masks = [masks_gt[x] for x in structural_idx] + good_masks
            structural_image_auroc = metrics.compute_imagewise_retrieval_metrics(
                structural_scores, structural_labels
            )["auroc"]
            print("Computing pixel auroc...")
            structural_pixel_auroc = metrics.compute_pixelwise_retrieval_metrics(
                structural_segs, structural_masks
            )["auroc"]

    except:
        image_auroc = 0.0
        pixel_auroc = 0.0
    print("Failed at computing image auroc...")

    result = {"test_data_name": os.path.basename(save_dir_path)}
    result["image_auroc"] = image_auroc * 100
    result["pixel_auroc"] = pixel_auroc * 100

    if is_loco:
        result["structural_image_auroc"] = structural_image_auroc * 100
        result["structural_pixel_auroc"] = structural_pixel_auroc * 100
        result["logical_image_auroc"] = logical_image_auroc * 100
        result["logical_pixel_auroc"] = logical_pixel_auroc * 100


    return result


def summarize_result(result_list, save_dir_path):
    df = pd.DataFrame(result_list)

    # Save to CSV
    save_path = os.path.join(save_dir_path, "result.csv")
    df.to_csv(save_path, index=False)  # 'index=False' to avoid writing row numbers

    return df
