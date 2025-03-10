import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import numpy as np
import json


_CLASSNAMES = [
'breakfast_box', 
'juice_bottle', 
'pushpins',  
'screw_bag', 
'splicing_connectors'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RealIADDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for mvad.
    """

    def __init__(
        self,
        data_path,
        data_config,
        training,
        resize=256,
        imagesize=224,
    ):
        """
        Args:
            data_path: [str]. Path to the mvad data folder.
            classname: [str or None]. Name of mvad class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvad.DatasetSplit.TRAIN. Note that
                   mvad.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.data_path = data_path
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.training = training
        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            # transforms.RandomHorizontalFlip(h_flip_p),
            # transforms.RandomVerticalFlip(v_flip_p),
            # transforms.RandomGrayscale(gray_p),
            # transforms.RandomAffine(rotate_degrees, 
            #                         translate=(translate, translate),
            #                         scale=(1.0-scale, 1.0+scale),
            #                         interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
        self.imagesize = (3, imagesize, imagesize)

        self.data_config = data_config
        self.meta_file = self.data_config.train.meta_file


        if isinstance(self.meta_file, str):
            self.meta_file = [self.meta_file]

        # construct metas
        self.metas = sum((self.load_explicit(path, self.training)
                          for path in self.meta_file), [])



    @staticmethod
    def load_explicit(path: str, is_training: bool):
        SAMPLE_KEYS = {'category', 'anomaly_class', 'image_path', 'mask_path'}

        with open(path, 'r') as fp:
            info = json.load(fp)
            assert isinstance(info, dict) and all(
                key in info for key in ('meta', 'train', 'test')
            )
            meta = info['meta']
            train = info['train']
            test = info['test']
            raw_samples = train if is_training else test

        assert isinstance(raw_samples, list) and all(
            isinstance(sample, dict) and set(sample.keys()) == SAMPLE_KEYS
            for sample in raw_samples
        )
        assert isinstance(meta, dict)
        prefix = meta['prefix']
        normal_class = meta['normal_class']

        if is_training:
            return [dict(filename=os.path.join(prefix, sample['image_path']),
                         label_name=normal_class, label=0,
                         clsname=sample['category'])
                    for sample in raw_samples]
        else:
            def as_normal(sample):
                return (sample['mask_path'] is None or
                        sample['anomaly_class'] == normal_class)

            return [dict(
                filename=os.path.join(prefix, sample['image_path']),
                maskname=None if as_normal(sample)
                else os.path.join(prefix, sample['mask_path']),
                label=0 if as_normal(sample) else 1,
                label_name=sample['anomaly_class'],
                clsname=sample['category']
            ) for sample in raw_samples]



    def __getitem__(self, idx):
        data = {}
        meta = self.metas[idx]

        filename = meta["filename"]
        is_anomaly = meta["label"]
        anomaly = meta["label_name"]
        image_path = os.path.join(self.data_path, filename)
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # read / generate mask
        if meta.get("maskname", None):
            data['maskname'] = meta['maskname']
            mask_path = os.path.join(self.data_path, data['maskname'])
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
            mask = (mask>0).type(torch.LongTensor)
        else:
            data['maskname'] = ''
            if is_anomaly == 0:  # good
                mask = torch.zeros((1, self.imagesize[1], self.imagesize[2])).type(torch.int8)
            elif is_anomaly == 1:  # defective
                # mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
                mask = (torch.ones((1, self.imagesize[1], self.imagesize[2]))).type(torch.int8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")


        data.update(
            {
                "image": image,
                "height": self.imagesize[1],
                "width": self.imagesize[2],
                "anomaly": anomaly,
                "is_anomaly": is_anomaly,
                "image_name": filename,
                "image_path": image_path,
                "mask": mask
            }
        )

        if meta.get("clsname", None):
            data["classname"] = meta["clsname"]
        else:
            data["classname"] = filename.split("/")[-4]

        return data



    def __len__(self):
        return len(self.metas)




