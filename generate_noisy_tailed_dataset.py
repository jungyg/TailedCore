import os
import re
import random
import glob
import subprocess
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
from copy import deepcopy
from collections import defaultdict
from typing import List
import argparse

from src.utils import (
    set_seed,
    modify_subfolders_in_path,
    save_dict,
    load_dict,
    save_dicts_to_csv,
)

import argparse

STEP_TAIL_CLASSES_HARD = {
    "mvtec": [
        "bottle",
        "hazelnut",
        "leather",
        "tile",
        "toothbrush",
        "grid",
        "capsule",
        "pill",
        "screw",
        "zipper",
    ]
}

STEP_TAIL_CLASSES_EASY = {
    "mvtec": [
        "bottle",
        "hazelnut",
        "leather",
        "tile",
        "toothbrush",
        "cable",
        "metal_nut",
        "transistor",
        "carpet",
        "wood",
    ]
}


PARETO_CLASS_ORDER_HARD = {
    "mvtec": [
        "carpet",
        "wood",
        "cable",
        "metal_nut",
        "transistor",
        "bottle",
        "hazelnut",
        "leather",
        "tile",
        "toothbrush",
        "grid",
        "capsule",
        "pill",
        "screw",
        "zipper",
    ]
}

PARETO_CLASS_ORDER_EASY = {
    "mvtec": [
        "screw",
        "pill",
        "grid",
        "zipper",
        "capsule",
        "bottle",
        "hazelnut",
        "leather",
        "tile",
        "toothbrush",
        "carpet",
        "wood",
        "cable",
        "metal_nut",
        "transistor",
    ]
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

NUM_TRAIN_SAMPLES_VISA = {
    "candle": 900,
    "capsules": 542,
    "cashew": 450,
    "chewinggum": 453,
    "fryum": 450,
    "macaroni1": 900,
    "macaroni2": 900,
    "pcb1": 904,
    "pcb2": 901,
    "pcb3": 905,
    "pcb4": 904,
    "pipe_fryum": 450,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _load_data_config(data_config_path):

    data_config = load_dict(data_config_path)
    tailed_files = data_config["tailed_files"]
    noisy_files = data_config["noisy_files"]
    return tailed_files, noisy_files


def _save_data_config(
    tailed_files,
    noisy_files,
    data_config_path,
):
    data_config = {"tailed_files": tailed_files, "noisy_files": noisy_files}

    save_dict(data_config, data_config_path)

    


def _make_data(
    source_dir,
    target_dir,
    tailed_files,
    noisy_files,
    copy=False,
) -> None:

    file_mapper_tail = _make_file_mapper(source_dir, target_dir, file_list=tailed_files)
    file_mapper_noise = _make_file_mapper(
        source_dir,
        target_dir,
        file_list=noisy_files,
        reflect_subfolder_depth=1,
        modify_subfolder_by={-2: "good", -3: "train"},
    )

    create_files(file_mapper_tail, symlink=not copy)
    create_files(file_mapper_noise, symlink=not copy)


def _get_mvtec_base_file_info(source_dir):
    # tail_files contain all file paths but with modified distributions on the tail classes
    files = {}
    train_files = {}
    test_files = {}
    anomaly_files = {}

    _MVTEC_CLASS_LIST = get_subdirectories(source_dir)

    for class_name in _MVTEC_CLASS_LIST:
        files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name),
            source_dir,
        )
        train_files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name, "train"),
            source_dir,
        )
        test_files[class_name] = list_files_in_folders(
            os.path.join(source_dir, class_name, "test"),
            source_dir,
        )

        _test_good_files = list_files_in_folders(
            os.path.join(source_dir, class_name, "test", "good"),
            source_dir,
        )
        anomaly_files[class_name] = [
            file for file in test_files[class_name] if file not in _test_good_files
        ]

    return files, train_files, anomaly_files


def _select_tail_classes_and_noise_samples(
    files, train_files, anomaly_files, num_tail_samples, num_noise_samples, head_classes
):
    # select tailed samples
    tailed_files = {}
    for tail_class, num_samples in num_tail_samples.items():
        _remove_files = list_files_to_remove(train_files[tail_class], k=num_samples)
        tailed_files[tail_class] = [
            file for file in files[tail_class] if file not in _remove_files
        ]

    for head_class in head_classes:
        tailed_files[head_class] = files[head_class]

    # select noise samples to add
    noisy_files = {}
    for noisy_class, num_samples in num_noise_samples.items():
        noisy_files[noisy_class] = random.sample(
            anomaly_files[noisy_class], min(num_samples, len(anomaly_files[noisy_class]))
        )

    tailed_files = [item for sublist in tailed_files.values() for item in sublist]
    noisy_files = [item for sublist in noisy_files.values() for item in sublist]

    return tailed_files, noisy_files


def _make_class_info_pareto_tail(
    class_list,
    train_files,
    noise_ratio,
    tail_classes,
    n_iter=1,
):

    pareto_alpha = 6.0  # hard-coded
    target_class_dist = get_discrete_pareto_pmf(
        alpha=pareto_alpha, sampe_space_size=len(class_list)
    )
    num_train_samples = {}
    for train_class in train_files.keys():
        num_train_samples[train_class] = len(train_files[train_class])

    total_num_tail_samples = 0
    _target_class_dist = deepcopy(target_class_dist)

    if tail_classes is not None:
        for _ in range(n_iter):
            class_names = list(num_train_samples.keys())

            # assert set(class_order) == set(class_names)

            _current_class_dist = list(np.sort(_target_class_dist)[::-1])
            _target_class_dist = np.empty_like(_target_class_dist)

            for i, class_name in enumerate(tail_classes):
                _target_class_dist[class_names.index(class_name)] = _current_class_dist[
                    i
                ]
            _target_num_class_samples = redistribute_num_class_samples(
                list(num_train_samples.values()), _target_class_dist
            )

            total_num_tail_samples = sum(_target_num_class_samples)
            target_num_class_samples = _target_num_class_samples

    else:
        for _ in range(n_iter):
            np.random.shuffle(_target_class_dist)

            _target_num_class_samples = redistribute_num_class_samples(
                list(num_train_samples.values()), _target_class_dist
            )
            if sum(_target_num_class_samples) > total_num_tail_samples:
                total_num_tail_samples = sum(_target_num_class_samples)
                target_num_class_samples = _target_num_class_samples

    num_tail_samples = {}
    for i, class_name in enumerate(train_files.keys()):
        num_tail_samples[class_name] = target_num_class_samples[i]

    min_size = 20
    total_num_noise_samples = round(total_num_tail_samples * noise_ratio)
    num_noise_samples = sample_name2size(
        num_tail_samples, total_num_noise_samples, min_size
    )

    return num_tail_samples, num_noise_samples, []


def sample_name2size(name2size, n_samples, min_size=20):
    # Flatten the dictionary: [(key, value), ...]
    flattened = [(key, value) for key, value in name2size.items() for _ in range(value)]

    random.shuffle(flattened)

    # Extract and return keys and values of the sampled elements
    num_samples_by_keys = defaultdict(int)

    counter = 0
    for class_name, class_size in flattened:
        if class_size >= min_size:
            num_samples_by_keys[class_name] += 1
            counter += 1

        if counter == n_samples:
            break

    return dict(num_samples_by_keys)


def redistribute_num_class_samples(
    num_class_samples: list, target_class_dist: list, min_num_samples: int = 5
) -> list:

    num_class_samples = np.array(num_class_samples)
    target_class_dist = np.array(target_class_dist)

    assert (num_class_samples > min_num_samples).all()

    idx_max = target_class_dist.argmax()
    factor = num_class_samples[idx_max] / target_class_dist[idx_max]
    desired_samples = np.round(target_class_dist * factor).astype(int)

    # Adjusting downwards only
    for i in range(len(desired_samples)):
        desired_samples[i] = max(
            min(num_class_samples[i], desired_samples[i]), min_num_samples
        )

    return desired_samples.tolist()


def get_discrete_pareto_pmf(alpha, sampe_space_size, epsilon=0.01):
    assert epsilon > 0
    x = np.linspace(
        pareto.ppf(epsilon, alpha), pareto.ppf(1 - epsilon, alpha), sampe_space_size + 1
    )
    cdf = pareto.cdf(x, alpha)
    pmf = (cdf - np.concatenate([[0], cdf])[:-1])[1:]

    return pmf


def _make_class_info_step_tail(
    class_list,
    train_files,
    anomaly_files,
    noise_ratio,
    tail_k,
    tail_class_ratio,
    tail_classes: List[str] = None,
):
    _num_tail_classes = round(len(class_list) * tail_class_ratio)

    if tail_classes is None:
        tail_classes = random.sample(class_list, _num_tail_classes)

    assert len(tail_classes) == _num_tail_classes
    head_classes = [cls_name for cls_name in class_list if cls_name not in tail_classes]

    num_tail_samples = {}
    num_noise_samples = {}

    for tail_class in tail_classes:
        num_tail_samples[tail_class] = tail_k

    noisy_classes = head_classes

    for noisy_class in noisy_classes:
        num_noise_samples[noisy_class] = min(
            len(anomaly_files[noisy_class]),
            round(len(train_files[noisy_class]) * noise_ratio),
        )

    return num_tail_samples, num_noise_samples, head_classes


def _make_file_mapper(
    source_dir,
    target_dir,
    file_list=None,
    reflect_subfolder_depth=0,
    modify_subfolder_by=None,
):
    assert file_list is not None
    file_mapper = {}
    for file in file_list:
        _rel_path = file
        _source_file_path = os.path.join(source_dir, _rel_path)
        _target_file_path = os.path.join(target_dir, _rel_path)
        if reflect_subfolder_depth > 0:
            _target_file_path = os.path.join(
                os.path.dirname(_target_file_path),
                "_".join(_rel_path.split("/")[-(reflect_subfolder_depth + 1) :]),
            )

        if modify_subfolder_by:
            _target_file_path = modify_subfolders_in_path(
                _target_file_path, modify_subfolder_by
            )
        file_mapper[_source_file_path] = _target_file_path

    return file_mapper


def list_files_in_folders(
    directory, source_dir, exts=["png", "jpg", "jpeg", "JPG", "JPEG"]
):
    """
    List all files with given extensions in a directory and its subdirectories.

    :param directory: The path to the directory to search in.
    :param source_dir: The source directory to calculate relative paths.
    :param exts: The file extensions to look for (default is ['png', 'jpg', 'jpeg']).
    :return: A list of paths to files with the specified extensions.
    """
    all_files = []
    for ext in exts:
        files = _list_files_in_folders(directory, source_dir, ext)
        all_files.extend(files)

    return all_files


def _list_files_in_folders(directory, source_dir, ext="png"):
    """
    List all files with a given extension in a directory and its subdirectories using glob.

    :param directory: The path to the directory to search in.
    :param ext: The file extension to look for (default is 'png').
    :return: A list of paths to files with the specified extension.
    """
    # Construct the search pattern
    search_pattern = os.path.join(directory, "**", f"*.{ext}")

    # Use glob to find files recursively
    files_with_ext = glob.glob(search_pattern, recursive=True)
    rel_files_with_ext = [os.path.relpath(path, source_dir) for path in files_with_ext]

    return rel_files_with_ext


def list_files_to_remove(file_list, k):
    """
    List file paths to be removed, keeping only K random files in the given directory.

    :param directory: The path to the directory.
    :param k: The number of files to keep.
    :return: A list of file paths to be removed.
    """
    # # List all files in the directory
    # all_files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    all_files = deepcopy(file_list)

    # # If there are fewer or equal files than k, return an empty list as there's nothing to remove
    if len(all_files) <= k:
        return []

    # Randomly permute the list of files
    random.shuffle(all_files)

    # Select the first K files to keep
    files_to_keep = set(all_files[:k])

    # List the files not in the keep list
    files_to_remove = [file for file in all_files if file not in files_to_keep]

    return files_to_remove


def _create_symlink(source_path, target_path):
    resolved_source_path = os.path.realpath(source_path)

    if not os.path.exists(target_path):
        os.symlink(resolved_source_path, target_path)

def _copy_file(source_path, target_path):
    # Ensure the target file does not exist to mimic the symlink behavior
    if not os.path.exists(target_path):
        shutil.copy2(source_path, target_path)


def create_files(file_mapper, symlink=True):
    """
    Create symbolic links based on the file_mapper dictionary.

    :param file_mapper: Dictionary where key is source path and value is target path.
    """
    for source, target in file_mapper.items():
        # Ensure the directory of the target path exists
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)

        if symlink:
            _create_symlink(source, target)
        else:
            _copy_file(source, target)


def compare_directories(
    original_dir, modified_dir, is_file_to_exclude, extension="*.png"
):
    # Function to check the relative depth of a file

    original_files = glob.glob(
        os.path.join(original_dir, "**", extension), recursive=True
    )
    original_files = [
        f for f in original_files if not is_file_to_exclude(f, original_dir)
    ]

    modified_files = glob.glob(
        os.path.join(modified_dir, "**", extension), recursive=True
    )
    modified_files = [
        f for f in modified_files if not is_file_to_exclude(f, modified_dir)
    ]

    # Check if the count of files is the same
    if len(original_files) != len(modified_files):
        return False, "Number of PNG files differ between directories."

    # Check each file
    for original_file in original_files:
        modified_file = original_file.replace(original_dir, modified_dir)

        # Check if the file exists in the modified directory
        if not modified_file in modified_files:
            return False, f"File {modified_file} does not exist in modified directory."

        # Check if file contents are the same
        if not os.path.samefile(original_file, modified_file):
            return False, f"File contents differ for {original_file}"

    return True, "Directories are equivalent except for 'train' folders."


def is_in_mvtec_train_folder(file_path, base_dir):
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    return "train" in parts and parts.index("train") == 1


def get_subdirectories(directory_path):
    """
    Returns a list of subdirectory names found in the given directory path.

    Args:
    directory_path (str): The path to the directory whose subdirectories are to be listed.

    Returns:
    list: A list of subdirectory names.
    """
    subdirectories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]
    return subdirectories


def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def list_end_branch_files(base_path, extensions=[".png", ".jpg", ".jpeg", ".bmp"]):
    # Construct a pattern for recursive search
    pattern = os.path.join(base_path, "**", "*")
    all_paths = glob.glob(pattern, recursive=True)

    # Define the extensions you're interested in, in lowercase

    # Filter out directories, then filter for specific extensions, ignoring case
    only_files_with_extensions = [
        p
        for p in all_paths
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in extensions
    ]

    return only_files_with_extensions


def subtract_lists(list_a, list_b):
    """
    Subtract elements of list_b from list_a.

    Parameters:
    - list_a (list): The list from which elements will be subtracted.
    - list_b (list): The list containing elements to subtract from list_a.

    Returns:
    - list: A new list with elements of list_b removed from list_a.
    """
    # Convert list_b to a set for more efficient lookups
    set_b = set(list_b)

    # Subtract elements of list_b from list_a using list comprehension
    result_list = [item for item in list_a if item not in set_b]

    return result_list


def count_files_with_numbers_and_chars(file_paths):
    """
    Counts the number of files in a list of file paths where the file name
    (ignoring the extension) includes not only numbers but also English characters.

    Parameters:
    - file_paths (list): A list of strings representing file paths.

    Returns:
    - int: The count of files matching the criteria.
    """
    # Regex pattern to match file names with at least one number and one English letter
    pattern = re.compile(r"(?=.*\d)(?=.*[a-zA-Z])")

    count = 0
    for path in file_paths:
        # Extract the base file name without the extension
        file_name = os.path.splitext(os.path.basename(path))[0]

        # Check if the file name matches the pattern
        if pattern.search(file_name):
            count += 1

    return count


def get_files_with_numbers_and_chars(file_paths):
    """
    Retrieves a list of file paths where the file name
    (ignoring the extension) includes not only numbers but also English characters.

    Parameters:
    - file_paths (list): A list of strings representing file paths.

    Returns:
    - list: A list of file paths matching the criteria.
    """
    # Regex pattern to match file names with at least one number and one English letter
    pattern = re.compile(r"(?=.*\d)(?=.*[a-zA-Z])")

    matching_files = []
    for path in file_paths:
        # Extract the base file name without the extension
        file_name = os.path.splitext(os.path.basename(path))[0]

        # If the file name matches the pattern, add it to the list
        if pattern.search(file_name):
            matching_files.append(path)

    return matching_files


def get_relative_paths(base_path, file_paths):
    """
    Converts a list of absolute file paths to relative paths given a base path.

    Parameters:
    - base_path (str): The base directory path from which to calculate relative paths.
    - file_paths (list): A list of strings representing the absolute file paths.

    Returns:
    - list: A list of strings representing the relative file paths.
    """
    relative_paths = [os.path.relpath(path, base_path) for path in file_paths]
    return relative_paths


def train_anomaly_to_test(rel_file_paths):
    return [_train_anomaly_to_test(rel_file_path) for rel_file_path in rel_file_paths]


def _train_anomaly_to_test(rel_file_path):
    """
    Transforms the file path from formats like 'wood/train/good/scratch_016.png' or
    'wood/train/good/abc_def_ghi_000.png' to 'wood/test/scratch/016.png' or
    'wood/test/abc_def_ghi/000.png', correctly separating the character and numeric parts.

    Parameters:
    - original_path (str): Original file path.

    Returns:
    - str: Transformed file path.
    """
    # Split the path into its components
    path_parts = rel_file_path.split("/")

    # Extract the file name and extension
    base_name, extension = os.path.splitext(path_parts[-1])

    # Use regex to separate the numeric part from the rest of the file name
    match = re.search(r"([^_]+)_?(\d+)$", base_name)
    if not match:
        raise ValueError("File name does not match expected pattern")

    # Everything before the last numeric part is considered as character part
    char_part = base_name[
        : match.start(2) - 1
    ]  # Exclude the underscore before the numeric part
    num_part = match.group(2)

    # Construct the new path
    new_path = os.path.join(path_parts[0], "test", char_part, f"{num_part}{extension}")

    return new_path


# FIXME: hard-coded; revision is required
def make_config_pkl_from_data(data_dir, dataset="mvtec", save_pkl=False):

    if dataset == "mvtec":
        num_train_samples = NUM_TRAIN_SAMPLES_MVTEC
    elif dataset == "visa":
        num_train_samples = NUM_TRAIN_SAMPLES_VISA
    else:
        raise NotImplementedError()

    class_names = list_directories(data_dir)

    files = {}
    train_files = {}
    _test_files = {}
    _test_good_files = {}
    anomaly_files = {}
    num_tail_samples = {}
    num_noise_samples = {}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)

        files[class_name] = get_relative_paths(
            data_dir, list_end_branch_files(class_path)
        )
        train_files[class_name] = get_relative_paths(
            data_dir, list_end_branch_files(os.path.join(class_path, "train"))
        )
        _test_files[class_name] = get_relative_paths(
            data_dir, list_end_branch_files(os.path.join(class_path, "test"))
        )
        _test_good_files[class_name] = get_relative_paths(
            data_dir, list_end_branch_files(os.path.join(class_path, "test", "good"))
        )
        anomaly_files[class_name] = subtract_lists(
            _test_files[class_name], _test_good_files[class_name]
        )

        if len(train_files[class_name]) < num_train_samples[class_name]:
            num_tail_samples[class_name] = len(train_files[class_name])

        _num_noise_samples_on_this_class = count_files_with_numbers_and_chars(
            train_files[class_name]
        )

        if _num_noise_samples_on_this_class > 0:
            num_noise_samples[class_name] = _num_noise_samples_on_this_class

    head_classes = [
        class_name for class_name in class_names if class_name not in num_tail_samples
    ]

    _all_files = get_relative_paths(data_dir, list_end_branch_files(data_dir))
    _train_files = get_relative_paths(
        data_dir, list_end_branch_files(os.path.join(data_dir, "*", "train", "good"))
    )

    _anomaly_train_files = get_files_with_numbers_and_chars(_train_files)

    noisy_files = train_anomaly_to_test(_anomaly_train_files)
    tailed_files = subtract_lists(_all_files, _anomaly_train_files)

    data_config = {
        "tailed_files": tailed_files,
        "noisy_files": noisy_files,
    }

    if save_pkl:
        save_dict(data_config, f"{os.path.basename(data_dir)}.pkl")

    return data_config





def get_args():
    parser = argparse.ArgumentParser(description="Data processing script.")
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa"], default="mvtec", help="")
    parser.add_argument("--tail_type", type=str, choices=["pareto", "k4", "k1"], default="pareto", help="")
    parser.add_argument("--step_k", type=int, default=4, choices=[1, 4])
    parser.add_argument("--tail_ratio", type=float, default=0.6, help="") # only for k1 and k4
    parser.add_argument("--noise_ratio", type=int, default=0.1, help="ratio in %") # mvtec 10, visa 5
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--copy", action='store_true', 
                        help="Copy files. If false, then symlink")
    parser.add_argument("--random_tail", type=str2bool, default=True, help="")    
    parser.add_argument('--tail_classes', nargs='+', help='list of tail classes')
    args = parser.parse_args()

    if not args.random_tail:
        assert args.tail_classes != None
        tail_classes = 'Tails_' + '_'.join([x for x in sorted(args.tail_classes)])
    else:
        tail_classes = 'random'

    if args.tail_type == 'pareto':
        args.sub_data_path = f'{args.dataset}_{args.tail_type}_nr{args.noise_ratio}_seed{args.seed}_{tail_classes}'
    else:
        args.sub_data_path = f'{args.dataset}_{args.tail_type}_nr{args.noise_ratio}_tr{args.tail_ratio}_seed{args.seed}_{tail_classes}'

    if args.tail_type != 'pareto':
        assert args.step_k == int(args.tail_type[-1])
    return args




def main(args):

    source_dir = f"./data/{args.dataset}"
    target_dir = os.path.join('./data', args.sub_data_path)
    make_data(args, source_dir, target_dir, args.tail_classes)

    print(f"target_dir: {target_dir}")

    compare_directories(
        source_dir, target_dir, is_file_to_exclude=is_in_mvtec_train_folder
    )


def make_data(args, source_dir, target_dir, tail_classes):

    set_seed(args.seed)
    data_config_path = (
        f"{os.path.join('./data_configs',  os.path.basename(target_dir))}.pkl"
    )

    if os.path.exists(data_config_path):
        tailed_files, noisy_files = _load_data_config(data_config_path)
    elif os.path.exists(target_dir):
        data_config = make_config_pkl_from_data(target_dir, dataset=args.dataset)
        tailed_files, noisy_files = (
            data_config["tailed_files"],
            data_config["noisy_files"],
        )
        _save_data_config(
            tailed_files=tailed_files,
            noisy_files=noisy_files,
            data_config_path=data_config_path,
        )
    else:
        class_list = get_subdirectories(source_dir)
        src_files, src_train_files, src_anomaly_files = _get_mvtec_base_file_info(
            source_dir
        )

        if args.tail_type == 'k1' or args.tail_type == 'k4':
            num_tail_samples, num_noise_samples, head_classes = _make_class_info_step_tail(
                class_list=class_list,
                train_files=src_train_files,
                anomaly_files=src_anomaly_files,
                noise_ratio=args.noise_ratio,
                tail_k=args.step_k,
                tail_class_ratio=args.tail_ratio,
                tail_classes=tail_classes,
            )
        elif args.tail_type == 'pareto':
            num_tail_samples, num_noise_samples, head_classes = _make_class_info_pareto_tail(
                class_list=class_list,
                train_files=src_train_files,
                noise_ratio=args.noise_ratio,
                tail_classes=tail_classes
            )
        else:
            raise("tail type not implemented error")


        tailed_files, noisy_files = _select_tail_classes_and_noise_samples(
            files=src_files,
            train_files=src_train_files,
            anomaly_files=src_anomaly_files,
            num_tail_samples=num_tail_samples,
            num_noise_samples=num_noise_samples,
            head_classes=head_classes,
        )

        save_dicts_to_csv(
            [num_tail_samples, num_noise_samples],
            os.path.splitext(data_config_path)[0] + ".csv",
        )

        _save_data_config(
            tailed_files=tailed_files,
            noisy_files=noisy_files,
            data_config_path=data_config_path,
        )

    _make_data(
        source_dir=source_dir,
        target_dir=target_dir,
        tailed_files=tailed_files,
        noisy_files=noisy_files,
        copy=True,
    )


if __name__ == "__main__":

    args = get_args()
    main(args)
