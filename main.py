import os
import numpy as np
import argparse

import src.evaluator.result as result

from src import utils
from src.dataloader import get_dataloaders
from src.engine import Engine


def main(args):
    utils.set_seed(args.config.seed)

    config = args.config

    device = utils.set_torch_device(args.gpu)

    input_shape = (3, config.data.inputsize, config.data.inputsize)

    dataloaders = get_dataloaders(
        config.data,
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
    )

    result_list = []

    for _dataloaders in dataloaders:
        _train_dataloader = _dataloaders["train"]
        _test_dataloader = _dataloaders["test"]

        save_train_dir_path = os.path.join(
            "./results", args.data_sub_path, args.config_name, _train_dataloader.name
        )
        save_test_dir_path = os.path.join(save_train_dir_path, _test_dataloader.name)
        save_outputs_path = os.path.join(save_test_dir_path, "outputs.pkl")

        if os.path.exists(save_outputs_path):
            outputs = utils.load_dict(save_outputs_path)
            image_scores = outputs["image_scores"]
            score_masks = outputs["score_masks"]

            labels_gt = outputs["labels_gt"]
            masks_gt = outputs["masks_gt"]
            image_paths = outputs["image_paths"]
            del outputs
        else:

            labels_gt = []
            masks_gt = []

            # move to get_dataloaders part
            if args.dataset in ["mvtec", "visa", "loco", "realiad"]:
                for data in _test_dataloader:
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    masks_gt.extend(data["mask"].numpy().tolist())

            elif args.data_format in ["labelme"]:
                labels_gt = np.array(list(_test_dataloader.dataset.is_anomaly.values()))
                masks_gt = np.array(
                    [
                        np.array(mask_gt).astype(np.uint8)
                        for mask_gt in list(
                            _test_dataloader.dataset.get_masks().values()
                        )
                    ]
                ).squeeze()
            else:
                raise NotImplementedError()

            image_scores = []
            score_masks = []

            for backbone_name in config.model.backbone_names:

                engine = Engine(
                    config=config,
                    backbone_name=backbone_name,
                    device=device,
                    input_shape=input_shape,
                    train_dataloader=_train_dataloader,
                    test_dataloader=_test_dataloader,
                    faiss_on_gpu=args.faiss_on_gpu,
                    faiss_num_workers=args.faiss_num_workers,
                    sampler_on_gpu=args.sampler_on_gpu,
                    save_dir_path=save_train_dir_path,
                    patch_infer=args.patch_infer,
                    train_mode=getattr(config.model, "train_mode", None),
                )

                engine.train()

                # FIXME: For truely large-scale experiment, image_socre and score mask needs to be saved for each image separately, and tested accordingly in a separate manner.
                (
                    image_scores_per_backbone,
                    score_masks_per_backbone,
                    _image_paths,
                ) = engine.infer()

                image_scores.append(image_scores_per_backbone)
                score_masks.append(score_masks_per_backbone)

                del engine

            image_scores = np.array(image_scores)
            score_masks = np.array(score_masks)
            image_paths = _image_paths

            outputs = {
                "image_scores": image_scores,
                "score_masks": score_masks,
                "labels_gt": labels_gt,
                "masks_gt": masks_gt,
                "image_paths": image_paths,
            }

            utils.save_dict(outputs, save_outputs_path)

        image_scores = utils.minmax_normalize_image_scores(
            image_scores
        )  # this part incldues ensembling of different backbone outputs
        score_masks = utils.minmax_normalize_score_masks(
            score_masks
        )  # this part incldues ensembling of different backbone outputs

        
        masks_gt = np.array(masks_gt).astype(np.uint8)[:, 0, :, :]
        score_masks = np.array(score_masks)
        # FIXME: min_size is currently hard-coded
        result_list.append(
            result.save_result(
                image_paths,
                image_scores,
                labels_gt,
                score_masks,
                masks_gt,
                save_test_dir_path,
                num_ths=41,  # 41
            )
        )

    save_log_root = os.path.join(save_train_dir_path, 'metrics')
    os.makedirs(save_log_root, exist_ok=True)
    save_log_path = os.path.join(save_log_root, f'performance_{args.data_sub_path}_{args.config_name}.csv')
    result_df = utils.save_dicts_to_csv(result_list, save_log_path)

    performance = result_df["image_auroc"].mean()
    return performance



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root_path",type=str,default="./data",)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--noisy_lt_dataset", type=str, default="pareto_nr0.1_seed42")
    parser.add_argument("--config_name", type=str, default="tailedcore_mvtec")
    parser.add_argument("--step_k", type=int, default=4)
    parser.add_argument("--noise_ratio", type=float, default=0.1)
    parser.add_argument("--tail_ratio", type=float, default=0.6, 
                        help='ratio of tail classes (only for step_k4 and step_k1)')

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--faiss_on_gpu", type=bool, default=True)
    parser.add_argument("--faiss_num_workers", type=int, default=0)
    parser.add_argument("--sampler_on_gpu", type=bool, default=True)


    #####################################################################

    args = parser.parse_args()

    args.config = utils.load_config_args(
        os.path.join("./configs", args.config_name + ".yaml")
    )

    args.data_sub_path = f'{args.dataset}_{args.noisy_lt_dataset}'
    args.data_path = os.path.join(args.data_root_path, args.data_sub_path)
    args.patch_infer = False

    return args




if __name__ == "__main__":
    args = parse_args()
    performance = main(args)
