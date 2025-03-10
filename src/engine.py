import os
import torch
import gc
import numpy as np
import GPUtil
import torchvision.models as models
import torch.nn.functional as F
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim


from .coreset_model import BaseCore
from .backbone import get_backbone
from .feature_embedder import FeatureEmbedder, FeatureEmbedderTransformer
from .coreset_model import get_coreset_model
from . import automl
from . import utils


class Engine:
    def __init__(
        self,
        config,
        backbone_name,
        device,
        input_shape,
        train_dataloader,
        test_dataloader,
        faiss_on_gpu,
        faiss_num_workers,
        sampler_on_gpu,
        save_dir_path,
        patch_infer,
        train_mode,
    ):
        self.config = config
        self.backbone_name = backbone_name
        self.device = device
        self.input_shape = input_shape
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.sampler_on_gpu = sampler_on_gpu
        self.save_dir_path = save_dir_path
        self.patch_infer = patch_infer
        self.train_mode = train_mode

    def train(self):
        if self.train_mode is None:
            self._basic_train()
        elif self.train_mode == "partition":
            self._partition_train()
        else:
            raise NotImplementedError()

    def _basic_train(self):
        backbone = get_backbone(self.backbone_name)

        # FIXME: move inside to coreset (it depends on coreset model)
        embedding_to_extract_from = None
        if hasattr(self.config.model, 'embedding_to_extract_from'):
            embedding_to_extract_from = self.config.model.embedding_to_extract_from 
        feature_embedder = FeatureEmbedder(
            device=self.device,
            input_shape=self.input_shape,
            backbone=backbone,
            layers_to_extract_from=self.config.model.layers_to_extract,
            embedding_to_extract_from=embedding_to_extract_from
        )

        coreset_model = get_coreset_model(
            self.config.model,
            feature_embedder=feature_embedder,
            imagesize=self.config.data.imagesize,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=True,
        )

        coreset_model.fit(self.train_dataloader)

        self.coreset_model: BaseCore = coreset_model

    # FIXME: refactor
    def _partition_train(self):
        backbone = get_backbone(self.backbone_name)
        # FIXME: move inside to coreset (it depends on coreset model)
        feature_embedder = FeatureEmbedder(
            self.device, self.input_shape, backbone, self.config.model.layers_to_extract
        )

        partition_train_info = _get_partition_train_info(
            train_dataloader=self.train_dataloader,
            feature_map_shape=feature_embedder.get_feature_map_shape(),
            save_dir_path=self.save_dir_path,
            device=self.device,
        )

        len_feas = partition_train_info["len_feas"]
        data_partition_info = partition_train_info["data_partition_info"]

        fea_dim = feature_embedder.get_feature_map_shape()[-1]
        coreset_ratio = automl.get_max_coreset_ratio(fea_dim, len_feas)
        max_coreset_size = (
            automl.get_max_len_feas(fea_dim, max_usage=0.25, memory_type="available")
            if getattr(self.config.model, "suppress_coreset", False)
            else None
        )

        print(f"max_coreset_size: {max_coreset_size}")

        coreset_model = get_coreset_model(
            config=self.config,
            feature_embedder=feature_embedder,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=False,
            coreset_ratio=coreset_ratio,
            max_coreset_size=max_coreset_size,
        )

        _fit_coreset_by_partition_train(
            coreset_model=coreset_model,
            train_dataloader=self.train_dataloader,
            data_partition_info=data_partition_info,
            save_dir_path=self.save_dir_path,
        )

        self.coreset_model: BaseCore = coreset_model

    def infer(self) -> np.ndarray:
        if self.patch_infer:
            return self._patch_infer()
        else:
            return self._image_infer()

    def _image_infer(self) -> np.ndarray:
        image_scores = []
        score_masks = []
        image_paths = []

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for data in data_iterator:
                images = data["image"]

                _image_scores, _score_masks = self.coreset_model.predict(images)

                image_scores.extend(_image_scores)
                score_masks.extend(_score_masks)
                image_paths.extend(data["image_path"])

        return image_scores, score_masks, image_paths

    def _patch_infer(self) -> np.ndarray:
        image_scores = {}
        score_masks = {}

        image_sizes = self.test_dataloader.dataset.image_sizes
        for _image_path, _image_size in image_sizes.items():
            image_scores[_image_path] = 0
            score_masks[_image_path] = np.zeros((_image_size[0], _image_size[1]))

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for _data in data_iterator:
                _images = _data["image"]
                _patches = _data["patch"]
                _image_paths = _data["image_path"]
                _image_scores, _score_masks = self.coreset_model.predict(_images)

                for i, _image_path in enumerate(_image_paths):
                    image_scores[_image_path] = max(
                        _image_scores[i], image_scores[_image_path]
                    )
                    _x0, _y0, _x1, _y1 = _patches[i]
                    score_masks[_image_path][_y0:_y1, _x0:_x1] = np.maximum(
                        score_masks[_image_path][_y0:_y1, _x0:_x1], _score_masks[i]
                    )  # FIXME: debug score_masks[_image_path][_y0:_y1, _x0:_x1] = _score_masks[i]

        return (
            list(image_scores.values()),
            list(score_masks.values()),
            list(image_scores.keys()),
        )




class Engine_Transformer:
    def __init__(
        self,
        config,
        backbone_name,
        transformer_name,
        device,
        input_shape,
        train_dataloader,
        test_dataloader,
        faiss_on_gpu,
        faiss_num_workers,
        sampler_on_gpu,
        save_dir_path,
        patch_infer,
        train_mode,
    ):
        self.config = config
        self.backbone_name = backbone_name
        self.transformer_name = transformer_name
        self.device = device
        self.input_shape = input_shape
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.sampler_on_gpu = sampler_on_gpu
        self.save_dir_path = save_dir_path
        self.patch_infer = patch_infer
        self.train_mode = train_mode

    def train(self):
        if self.train_mode is None:
            self._basic_train()
        elif self.train_mode == "partition":
            self._partition_train()
        else:
            raise NotImplementedError()

    def _basic_train(self):
        backbone = get_backbone(self.backbone_name)
        transformer = get_backbone(self.transformer_name)

        # FIXME: move inside to coreset (it depends on coreset model)
        embedding_to_extract_from = None
        if hasattr(self.config.model, 'embedding_to_extract_from'):
            embedding_to_extract_from = self.config.model.embedding_to_extract_from 
        feature_embedder = FeatureEmbedderTransformer(
            device=self.device,
            input_shape=self.input_shape,
            backbone=backbone,
            transformer=transformer,
            layers_to_extract_from=self.config.model.layers_to_extract,
            embedding_to_extract_from=embedding_to_extract_from
        )

        coreset_model = get_coreset_model(
            self.config.model,
            feature_embedder=feature_embedder,
            imagesize=self.config.data.imagesize,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=True,
        )

        coreset_model.fit(self.train_dataloader)

        self.coreset_model: BaseCore = coreset_model

    # FIXME: refactor
    def _partition_train(self):
        backbone = get_backbone(self.backbone_name)
        # FIXME: move inside to coreset (it depends on coreset model)
        feature_embedder = FeatureEmbedder(
            self.device, self.input_shape, backbone, self.config.model.layers_to_extract
        )

        partition_train_info = _get_partition_train_info(
            train_dataloader=self.train_dataloader,
            feature_map_shape=feature_embedder.get_feature_map_shape(),
            save_dir_path=self.save_dir_path,
            device=self.device,
        )

        len_feas = partition_train_info["len_feas"]
        data_partition_info = partition_train_info["data_partition_info"]

        fea_dim = feature_embedder.get_feature_map_shape()[-1]
        coreset_ratio = automl.get_max_coreset_ratio(fea_dim, len_feas)
        max_coreset_size = (
            automl.get_max_len_feas(fea_dim, max_usage=0.25, memory_type="available")
            if getattr(self.config.model, "suppress_coreset", False)
            else None
        )

        print(f"max_coreset_size: {max_coreset_size}")

        coreset_model = get_coreset_model(
            config=self.config,
            feature_embedder=feature_embedder,
            device=self.device,
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=False,
            coreset_ratio=coreset_ratio,
            max_coreset_size=max_coreset_size,
        )

        _fit_coreset_by_partition_train(
            coreset_model=coreset_model,
            train_dataloader=self.train_dataloader,
            data_partition_info=data_partition_info,
            save_dir_path=self.save_dir_path,
        )

        self.coreset_model: BaseCore = coreset_model

    def infer(self) -> np.ndarray:
        if self.patch_infer:
            return self._patch_infer()
        else:
            return self._image_infer()

    def _image_infer(self) -> np.ndarray:
        image_scores = []
        score_masks = []
        image_paths = []

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for data in data_iterator:
                images = data["image"]

                _image_scores, _score_masks = self.coreset_model.predict(images)

                image_scores.extend(_image_scores)
                score_masks.extend(_score_masks)
                image_paths.extend(data["image_path"])

        return image_scores, score_masks, image_paths

    def _patch_infer(self) -> np.ndarray:
        image_scores = {}
        score_masks = {}

        image_sizes = self.test_dataloader.dataset.image_sizes
        for _image_path, _image_size in image_sizes.items():
            image_scores[_image_path] = 0
            score_masks[_image_path] = np.zeros((_image_size[0], _image_size[1]))

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for _data in data_iterator:
                _images = _data["image"]
                _patches = _data["patch"]
                _image_paths = _data["image_path"]
                _image_scores, _score_masks = self.coreset_model.predict(_images)

                for i, _image_path in enumerate(_image_paths):
                    image_scores[_image_path] = max(
                        _image_scores[i], image_scores[_image_path]
                    )
                    _x0, _y0, _x1, _y1 = _patches[i]
                    score_masks[_image_path][_y0:_y1, _x0:_x1] = np.maximum(
                        score_masks[_image_path][_y0:_y1, _x0:_x1], _score_masks[i]
                    )  # FIXME: debug score_masks[_image_path][_y0:_y1, _x0:_x1] = _score_masks[i]

        return (
            list(image_scores.values()),
            list(score_masks.values()),
            list(image_scores.keys()),
        )






def _read_integer(filepath):
    with open(filepath, "r") as file:
        int_read = int(file.read())
    return int_read


def _save_integer(filepath, integer):
    with open(filepath, "w") as file:
        file.write(str(integer))


def _get_partition_train_info(
    train_dataloader, feature_map_shape, save_dir_path, **kwargs
):
    device = kwargs["device"]
    partition_train_info_path = os.path.join(save_dir_path, "partition_train_info.pkl")

    # FIXME: revise the interface to simplify and clean the below
    from .datasets.labelme import IterablePatchDataset

    assert isinstance(
        train_dataloader.dataset, IterablePatchDataset
    ), f"train_mode=partition works only for {IterablePatchDataset.__name__}"

    image_paths = deepcopy(train_dataloader.dataset.image_paths)

    if os.path.exists(partition_train_info_path):
        return utils.load_dict(partition_train_info_path)

    else:
        patch_partitions = {}

        available_gpu_memory = (
            GPUtil.getGPUs()[device.index].memoryFree * 1024**2 * 0.75
        )  # in byte
        # FIXME: refactor the below block
        len_feas = 0
        for image_path in image_paths:
            num_patches_per_image = len(train_dataloader.dataset.patch_dict[image_path])
            len_feas_per_image = (
                feature_map_shape[0] * feature_map_shape[1] * num_patches_per_image
            )
            num_distance_map = 4
            required_memory = len_feas_per_image**2 * 4 * num_distance_map

            n_divide = int(np.ceil(np.sqrt(required_memory / (available_gpu_memory))))

            patch_idxes = list(range(num_patches_per_image))
            np.random.shuffle(patch_idxes)
            patch_partitions[image_path] = np.array_split(patch_idxes, n_divide)
            print(
                f"avg. num. of patches in each partition: {np.array([len(partition) for partition in patch_partitions[image_path]]).mean()}"
            )

            len_feas += len_feas_per_image

        _train_dataloader = deepcopy(train_dataloader)

        data_partition_info = []
        for image_path in tqdm(image_paths, desc="getting data partition info..."):
            for patch_partition in patch_partitions[image_path]:
                assert len(patch_partition) > 1

                _data_info = {
                    "image_paths": [image_path],
                    "patch_dict": {
                        image_path: [
                            _train_dataloader.dataset.patch_dict[image_path][idx]
                            for idx in patch_partition
                        ]
                    },
                    "is_anomaly": {
                        image_path: _train_dataloader.dataset.is_anomaly[image_path]
                    },
                    "image_sizes": {
                        image_path: _train_dataloader.dataset.image_sizes[image_path]
                    },
                    "blob_areas_dict": {
                        image_path: _train_dataloader.dataset.blob_areas_dict[
                            image_path
                        ]
                    },
                }

                data_partition_info.append(_data_info)

        partition_train_info = {
            # "patch_partitions": patch_partitions,
            "len_feas": len_feas,
            "data_partition_info": data_partition_info,
        }

        utils.save_dict(partition_train_info, partition_train_info_path)

    return partition_train_info


def _fit_coreset_by_partition_train(
    coreset_model, train_dataloader, data_partition_info, save_dir_path
):
    _tmp_dataloader = deepcopy(train_dataloader)

    train_status_file_path = os.path.join(save_dir_path, "train_status.txt")
    if os.path.exists(train_status_file_path):
        idx_train_complete = _read_integer(train_status_file_path)
    else:
        idx_train_complete = 0

    for i in tqdm(
        range(idx_train_complete, len(data_partition_info)),
        desc="partition training...",
    ):
        _data_info = data_partition_info[i]

        _tmp_dataloader.dataset.image_paths = _data_info["image_paths"]
        _tmp_dataloader.dataset.patch_dict = _data_info["patch_dict"]
        _tmp_dataloader.dataset.is_anomaly = _data_info["is_anomaly"]
        _tmp_dataloader.dataset.image_sizes = _data_info["image_sizes"]
        _tmp_dataloader.dataset.blob_areas_dict = _data_info["blob_areas_dict"]

        if i == 0:
            coreset_model.fit(_tmp_dataloader, set_predictor=False)
        elif (i % 10 == 0) or (i == len(data_partition_info) - 1):
            coreset_model.incremental_fit(
                _tmp_dataloader, set_predictor=False, save_coreset=True
            )
            _save_integer(train_status_file_path, i + 1)
        else:
            coreset_model.incremental_fit(
                _tmp_dataloader, set_predictor=False, save_coreset=False
            )

    # just load
    if _read_integer(train_status_file_path) == len(data_partition_info):
        gc.collect()
        torch.cuda.empty_cache()
        coreset_model.fit(None)

    return coreset_model


class AblationEngine:
    def __init__(
        self,
        config,
        backbone_name,
        device,
        input_shape,
        train_dataloader,
        test_dataloader,
        faiss_on_gpu,
        faiss_num_workers,
        sampler_on_gpu,
        save_dir_path,
        patch_infer,
        train_mode,
    ):
        self.config = config
        self.backbone_name = backbone_name
        self.device = device
        self.input_shape = input_shape
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.sampler_on_gpu = sampler_on_gpu
        self.save_dir_path = save_dir_path
        self.patch_infer = patch_infer
        self.train_mode = train_mode

    def train(self):
        self._basic_train()

    def _basic_train(self):
        backbone = get_backbone(self.backbone_name)

        # FIXME: move inside to coreset (it depends on coreset model)
        embedding_to_extract_from = None
        if hasattr(self.config.model, 'embedding_to_extract_from'):
            embedding_to_extract_from = self.config.model.embedding_to_extract_from 
        feature_embedder = FeatureEmbedder(
            device=self.device,
            input_shape=self.input_shape,
            backbone=backbone,
            layers_to_extract_from=self.config.model.layers_to_extract,
            embedding_to_extract_from=embedding_to_extract_from
        )

        embedding_extractor = self.set_embedding_extractor(iter=self.config.model.embedding_extractor_iter)
        embedding_extractor.fc = nn.Identity()
        embedding_extractor.eval()

        from .coreset_model import AblationTailedPatch
        coreset_model = AblationTailedPatch(
            feature_embedder=feature_embedder,
            embedding_extractor=embedding_extractor,
            device=self.device,
            imagesize=self.config.data.imagesize,
            coreset_ratio=getattr(self.config.model, "greedy_ratio", 0.01),
            faiss_on_gpu=self.faiss_on_gpu,
            faiss_num_workers=self.faiss_num_workers,
            sampler_on_gpu=self.sampler_on_gpu,
            save_dir_path=self.save_dir_path,
            brute=True,
            tail_th_type=self.config.model.tail_th_type,
            tail_data_augment_type=self.config.model.tail_data_augment_type,
            tail_lof=self.config.model.tail_lof,
        )

        coreset_model.fit(self.train_dataloader)

        self.coreset_model: BaseCore = coreset_model
    
    def set_embedding_extractor(self, iter):
        save_dir = "/".join(self.save_dir_path.split("/")[:3])
        filename_pattern = os.path.join(save_dir, f"resnet18_iter_{iter:03d}_*.pt")

        try:
            filename = glob(filename_pattern)[0]
        except:
            filename = None
        
        model = self._get_resnet18()
        if filename is None:
            self._train_embedding_extractor(model, num_epochs=20, save_dir=save_dir)
            filename = glob(filename_pattern)[0]
        
        model = self._load_embedding_extractor(model, filename)
        
        model.eval()
        return model
    
    def _get_resnet18(self):
        resnet18 = models.resnet18(pretrained=False).to(self.device)
        embedding_size = resnet18.fc.in_features
        num_classes, _ = self._get_dataset_info()
        # resnet18.fc = ScaledCosineSimilarityLayer(embedding_size, num_classes, 16.)
        resnet18.fc = nn.Linear(embedding_size, num_classes)
        resnet18 = resnet18.to(self.device)
        return resnet18

    def _load_embedding_extractor(self, model, filename):
        
        model.load_state_dict(torch.load(filename, map_location=self.device))

        return model

    def _train_embedding_extractor(self, model, num_epochs, save_dir):

        device = self.device  # Assuming device is defined

        # Initialize and configure the model
        resnet18 = model
        num_classes, names_to_ints = self._get_dataset_info()

        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(resnet18.parameters(), lr=0.00005, weight_decay=1e-6)

        train_dataloader = self._shuffle_dataloader(self.train_dataloader)

        # Training loop with tqdm for epochs
        i = 0
        freq = 1
        for epoch in tqdm(range(num_epochs), desc="Training embedding extractor...", leave=False):
            for data in train_dataloader:
                
                if i == 10:
                    freq = 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.0001

                if i % freq == 0:
                    overall_accuracy, tail_noise_accuracy = self.calculate_accuracies(resnet18, self.train_dataloader, device, names_to_ints)
                    
                    tqdm.write(f"Iteration: {i}, tail_noise_accuracy: {tail_noise_accuracy:.2f}% overall_accuracy:{overall_accuracy:.2f}")
                    torch.save(resnet18.state_dict(), os.path.join(save_dir, f"resnet18_iter_{i:03d}_tnacc{int(tail_noise_accuracy*100):03d}_oacc{int(overall_accuracy*100):03d}.pt"))

                images = data["image"].to(device)
                label_names = data["classname"]
                labels = names_to_ints(label_names).to(device)

                optimizer.zero_grad()
                outputs = resnet18(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if i % 1 == 0:
                    tqdm.write(f"loss: {loss.item()}")
                
                i += 1                
            
    def _shuffle_dataloader(self, dataloader):
        dataset = dataloader.dataset  # Access the dataset used by the original DataLoader

        # Now create a new DataLoader with shuffling enabled
        shuffled_dataloader = DataLoader(
            dataset, 
            batch_size=dataloader.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last
        )

        return shuffled_dataloader

    def _get_dataset_info(self):
        dataset = self.train_dataloader.dataset
        unique_classes = set()
        for _dataset in dataset.datasets:
            unique_classes.update(_dataset.classnames_to_use)
        
        # The number of unique classes
        num_classes = len(unique_classes)

        sorted_classes = sorted(unique_classes)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted_classes)}

        def names_to_ints(labels):
            # Convert list of class names to list of corresponding indices
            indices = [class_to_idx[label] for label in labels]
            # Convert list of indices to a torch tensor
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            return indices_tensor

        return num_classes, names_to_ints
    
    def calculate_accuracies(self, model, dataloader, device, names_to_ints):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        tail_noise_correct = 0
        tail_noise_total = 0

        if not hasattr(self, 'class_distribution'):
            self.class_distribution = self.get_class_distribution(dataloader.dataset)  # You need to implement this
        
        with torch.no_grad():  # No need to track gradients
            for data in tqdm(dataloader, desc='Measuring accuracies', leave=False):
                images = data["image"].to(device)
                label_names = data["classname"]
                labels = names_to_ints(label_names).to(device)
                image_names = data["image_name"]
                filenames = [os.path.splitext(os.path.basename(image_name))[0] for image_name in image_names]

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Determine valid indices for tail/noise based on few-shot criteria and filename pattern
                valid_indices = [i for i, (filename, label_name) in enumerate(zip(filenames, label_names))
                                if not filename.isnumeric() or self.class_distribution[label_name] < 20]

                # Filter labels for valid (few-shot and noise) samples
                valid_labels = labels[valid_indices]
                valid_predicted = predicted[valid_indices]

                tail_noise_total += valid_labels.size(0)
                tail_noise_correct += (valid_predicted == valid_labels).sum().item()

        overall_accuracy = 100 * correct / total if total > 0 else 0
        tail_noise_accuracy = 100 * tail_noise_correct / tail_noise_total if tail_noise_total > 0 else 0

        model.train()  # Set the model back to training mode
        return overall_accuracy, tail_noise_accuracy

    def get_class_distribution(self, dataset):
        # Implement logic to count the number of samples per class in the dataset
        # This is a placeholder function; actual implementation will vary
        class_counts = {}
        for sample in dataset:
            label_name = sample['classname']  # Adjust based on your dataset structure
            if label_name in class_counts:
                class_counts[label_name] += 1
            else:
                class_counts[label_name] = 1
        return class_counts
    def infer(self) -> np.ndarray:
        if self.patch_infer:
            return self._patch_infer()
        else:
            return self._image_infer()

    def _image_infer(self) -> np.ndarray:
        image_scores = []
        score_masks = []
        image_paths = []

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for data in data_iterator:
                images = data["image"]

                _image_scores, _score_masks = self.coreset_model.predict(images)

                image_scores.extend(_image_scores)
                score_masks.extend(_score_masks)
                image_paths.extend(data["image_path"])

        return image_scores, score_masks, image_paths

    def _patch_infer(self) -> np.ndarray:
        image_scores = {}
        score_masks = {}

        image_sizes = self.test_dataloader.dataset.image_sizes
        for _image_path, _image_size in image_sizes.items():
            image_scores[_image_path] = 0
            score_masks[_image_path] = np.zeros((_image_size[0], _image_size[1]))

        with tqdm(
            self.test_dataloader,
            desc="Inferring...",
        ) as data_iterator:
            for _data in data_iterator:
                _images = _data["image"]
                _patches = _data["patch"]
                _image_paths = _data["image_path"]
                _image_scores, _score_masks = self.coreset_model.predict(_images)

                for i, _image_path in enumerate(_image_paths):
                    image_scores[_image_path] = max(
                        _image_scores[i], image_scores[_image_path]
                    )
                    _x0, _y0, _x1, _y1 = _patches[i]
                    score_masks[_image_path][_y0:_y1, _x0:_x1] = np.maximum(
                        score_masks[_image_path][_y0:_y1, _x0:_x1], _score_masks[i]
                    )  # FIXME: debug score_masks[_image_path][_y0:_y1, _x0:_x1] = _score_masks[i]

        return (
            list(image_scores.values()),
            list(score_masks.values()),
            list(image_scores.keys()),
        )


class ScaledCosineSimilarityLayer(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=20.0):
        super(ScaledCosineSimilarityLayer, self).__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings):
        if embeddings.ndim == 4:
            embeddings = embeddings[:, :, 0, 0]
        
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_weights = F.normalize(self.weights, p=2, dim=1)
        # Compute scaled cosine similarity
        cosine_sim_matrix = F.linear(normalized_embeddings, normalized_weights) * self.scale
        return cosine_sim_matrix