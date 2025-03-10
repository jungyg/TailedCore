import numpy as np
import torch

import gc
import random

import src.utils as utils

from torch.utils.data import Subset, ConcatDataset, DataLoader

_DATASETS = {
    "mvtec": ["src.datasets.mvtec", "MVTecDataset"],
    "btad": ["src.datasets.btad", "BTADDataset"],
    "loco": ["src.datasets.loco", "LocoDataset"],
    "realiad": ["src.datasets.realiad", "RealIADDataset"],
}

NUM_TRAIN_SAMPLES_MVTEC = {
    "bottle": 209,
    "cable": 224,
    "capsule": 219,
    "carpet": 280,
    "grid": 264,
    "hazelnut": 391,
    "leather": 245,
    "metal_nut": 220,
    "pill": 267,
    "screw": 320,
    "tile": 230,
    "toothbrush": 60,
    "transistor": 213,
    "wood": 247,
    "zipper": 240,
}

_LOCO_CLS = [
    'breakfast_box', 
    'juice_bottle', 
    'pushpins',  
    'screw_bag', 
    'splicing_connectors'
]

_MVTEC_CLS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

_VISA_CLS = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]



def get_head_tail_dataloaders(data_config, data_format, data_path, batch_size):

    if data_format == "mvtec" or data_format == 'visa':
        return get_mvtec_dataloaders(
            data_path,
            batch_size,
            data_config.imagesize,
            data_config.resize,
            multiclass=True,
        )
    elif data_format == "loco":
        return get_loco_dataloaders(
            data_path,
            batch_size,
            data_config.imagesize,
            data_config.resize,
            multiclass=True,
        )
    elif data_format == "realiad":
        return get_realiad_dataloaders(
            data_path,
            batch_size,
            data_config,
        )
    else:
        raise NotImplementedError()



def get_dataloaders(data_config, dataset, data_path, batch_size):

    if dataset == "mvtec" or dataset == 'visa':
        return get_mvtec_dataloaders(
            data_path,
            batch_size,
            data_config.imagesize,
            data_config.resize,
            multiclass=True,
        )
    elif dataset == "loco":
        return get_loco_dataloaders(
            data_path,
            batch_size,
            data_config.imagesize,
            data_config.resize,
            multiclass=True,
        )
    elif dataset == "realiad":
        return get_realiad_dataloaders(
            data_path,
            batch_size,
            data_config,
        )
    else:
        raise NotImplementedError()


def get_mvtec_dataloaders(
    data_path, batch_size, imagesize, resize=None, multiclass=True
):
    import src.datasets.mvtec as mvtec

    classname_list = utils.get_folder_names(data_path)

    train_datasets = []
    test_dataloaders = []
    data_index = {}

    cnt = 0
    for classname in classname_list:
        _train_dataset = mvtec.MVTecDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=mvtec.DatasetSplit.TRAIN,
        )

        _test_dataset = mvtec.MVTecDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=mvtec.DatasetSplit.TEST,
        )


        if len(_train_dataset) - NUM_TRAIN_SAMPLES_MVTEC[classname] < 100:
            cnt += len(_train_dataset)


        data_index[classname] = len(_train_dataset)
        
        # Packaging
        train_datasets.append(_train_dataset)

        _test_dataloader = torch.utils.data.DataLoader(
            _test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        _test_dataloader.name = classname

        test_dataloaders.append(_test_dataloader)

    if multiclass:
        train_dataset = ConcatDataset(train_datasets)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        train_dataloader.name = "all"
        dataloaders = [
            {"train": train_dataloader, "test": test_dataloader}
            for test_dataloader in test_dataloaders
        ]

    else:
        dataloaders = []
        for _train_dataset, _test_dataloader in zip(train_datasets, test_dataloaders):
            _train_dataloader = torch.utils.data.DataLoader(
                _train_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
            dataloaders.append({"train": _train_dataloader, "test": _test_dataloader})
    print(f'num tail samples: {cnt}')
    return dataloaders


def get_loco_dataloaders(
    data_path, batch_size, imagesize, resize=None, multiclass=True
):
    import src.datasets.mvtec as mvtec
    import src.datasets.loco as loco

    classname_list = utils.get_folder_names(data_path)

    train_datasets = []
    test_dataloaders = []
    data_index = {}

    for classname in classname_list:
        _train_dataset = loco.LocoDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=loco.DatasetSplit.TRAIN,
        )

        _test_dataset = loco.LocoDataset(
            source=data_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=loco.DatasetSplit.TEST,
        )

        data_index[classname] = len(_train_dataset)
        
        # Packaging
        train_datasets.append(_train_dataset)

        _test_dataloader = torch.utils.data.DataLoader(
            _test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        _test_dataloader.name = classname

        test_dataloaders.append(_test_dataloader)

    if multiclass:
        train_dataset = ConcatDataset(train_datasets)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        train_dataloader.name = "all"

        dataloaders = [
            {"train": train_dataloader, "test": test_dataloader}
            for test_dataloader in test_dataloaders
        ]

    else:
        dataloaders = []
        for _train_dataset, _test_dataloader in zip(train_datasets, test_dataloaders):
            _train_dataloader = torch.utils.data.DataLoader(
                _train_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
            dataloaders.append({"train": _train_dataloader, "test": _test_dataloader})
    return dataloaders




def get_realiad_dataloaders(
    data_path, batch_size, data_config,
):
    import src.datasets.realiad as realiad

    train_dataset = realiad.RealIADDataset(data_path, data_config, training=True)
    test_dataset = realiad.RealIADDataset(data_path, data_config, training=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    train_dataloader.name = "all"

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_dataloader.name = "all"

    dataloaders = [
        {"train": train_dataloader, "test": test_dataloader}
    ]

    return dataloaders
