import argparse
import logging
import os
import json
import hashlib
import sys
import time

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_dataset import get_dataset

_logger = logging.getLogger(__name__)

_DATASET_SPLIT_SYNONYM = dict(train=None, validation=None, test=None)


def create_dataset(
        name,
        root,
        tsp_data_info,
        split='validation',
        expensive_prediction_path=None,
        **kwargs
):
    """
    Create torch dataset object that cloud replace timm dataset.

    Parameters
    ----------
    expensive_prediction_path: string
            path of expensive model prediction result
    name : string
        name of dataset
    root : string
        path to store the dataset
    tsp_data_info : string
        path get dataset info to generate sub-dataset
    split : string
        dataset type

    Returns
    -------
    dataset : torchvision.datasets
        object torch sub-dataset
    """
    _logger.info(f"Create dataset '{name}'...")
    if not _DATASET_SPLIT_SYNONYM.__contains__(split):
        msg = f"Argument \'split\' get unexpected \'{split}\'. It should be the following {list(_DATASET_SPLIT_SYNONYM.keys())}"
        _logger.error(msg)
        raise ValueError(msg)
    _logger.info(f"Try to get dataset '{name}'.")
    dataset = get_dataset(name=name, root=root, expensive_prediction_path=expensive_prediction_path)
    
    
    if "test" in name:
        _logger.info(f"Dataset contain: {len(dataset)}.")
        return dataset
    else:
        # Read json
        tsp_data_info = os.path.abspath(tsp_data_info)
        _logger.info(f"Try to read '{tsp_data_info}'.")
        if not os.path.isfile(tsp_data_info):
            msg = f"File {tsp_data_info} not existed."
            _logger.error(msg)
            raise ValueError(msg)

        f = open(tsp_data_info, 'r')
        data_info = json.load(f)
        f.close()
        _logger.info(f"Success loading data info from {tsp_data_info}.")

        # Parse index
        index_train = data_info.get("index_train", [])
        index_val = data_info.get("index_val", [])
        index_est = data_info.get("index_est", [])

        # analysis
        _logger.info(f"Dataset info contain: "
                     f"\n\t len_train:{len(index_train)} "
                     f"\n\t len_val:{len(index_val)} "
                     f"\n\t index_est:{len(index_est)} "
                     f"\n\t sum:{len(index_train) + len(index_val) + len(index_est)}")

        # create subset
        train_split_dataset = Subset(dataset, index_train)
        val_split_dataset = Subset(dataset, index_val)
        est_split_dataset = Subset(dataset, index_est)

        # return
        if split == "train":
            return train_split_dataset
        elif split == "validation":
            return val_split_dataset
        elif split == "test":
            return est_split_dataset


def tsp_sampler(dataset, train_test_ratio, seed=1234):
    """
    dataset sampler that cloud split dataset with index.

    Parameters
    ----------
    train_test_ratio : float
        split ratio (ex: 0.9 mean split to 90% and 10% )
    seed : int
        seed

    Returns
    -------
    X_train : list[int]
        list of train
    X_test : list[int]
        list of test
    """
    assert 0 <= train_test_ratio <= 1.0
    assert dataset is not None
    # Stratified sampling
    X = [i for i in range(len(dataset))]  # index
    try:
        Y = [i[1] for i in dataset.imgs]  # label
    except AttributeError as e:
        Y = [i for i in dataset.targets]  # label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, train_size=train_test_ratio, random_state=seed)
    
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, stratify=Y_test, train_size=0.5, random_state=seed+1)
    return X_train, X_test, X_val


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate(
        name,
        root,
        tsp_data_info,
        ratio,
        seed
):
    """
        dataset info generate that generate dataset info and store the index in to 'tsp_data_info' in json format

        Parameters
        ----------
        name : string
            name of dataset
        root : string
            path to store the dataset
        tsp_data_info : string
            path get dataset info to generate sub-dataset
        split : string
            dataset type
        ratio : float
            split ratio (ex: 0.9 mean split to 90% and 10% )
        seed : int
            seed
    """
    _logger.info(f"Try to get dataset '{name}'.")
    dataset = get_dataset(name=name, root=root)

    _logger.info(f"Try to sample dataset info.")
    index_train, index_val, index_est = tsp_sampler(dataset=dataset, train_test_ratio=ratio, seed=seed)

    # Check saving path, and save result
    os.makedirs(os.path.dirname(tsp_data_info), exist_ok=True)
    f = open(tsp_data_info, 'w')
    json.dump({
        "index_train": index_train,
        "index_val": index_val,
        "index_est": index_est,
    }, f)
    f.close()
    _logger.info(f"Index save at {tsp_data_info}")
    _logger.info(f"MD5: {md5(tsp_data_info)}")


if __name__ == '__main__':
    _logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s]  %(levelname)s  %(message)s")
    ch.setFormatter(formatter)
    _logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="Dataset path")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--json', type=str, required=True, help="Json file of dataset info")

    parser.add_argument('--lookup', action='store_true', help="look up existed dataset info")

    parser.add_argument('--generate', action='store_true', help="generate dataset info")
    parser.add_argument('--ratio', type=float, default=0.9, help="train_test_split ratio")
    parser.add_argument('--seed', type=int, default=1, help="seed")
    parser.add_argument('-f')
    cf = parser.parse_args()

    # if giving wrong args
    if cf.lookup and cf.generate:
        raise ValueError("Can only selected one, --lookup or --generate")

    # if just want to analysis existed dataset info (json file)
    if cf.lookup:
        _ = create_dataset(name=cf.dataset, root=cf.root, tsp_data_info=cf.json, split="train")

    # if you want to generate new dataset info (json file)
    if cf.generate:
        if os.path.isfile(cf.json):
            _logger.warning(f"File {cf.json} existed. Please remove it and try again if you want to regenerate it.")
        else:
            generate(name=cf.dataset, root=cf.root, tsp_data_info=cf.json, ratio=cf.ratio, seed=cf.seed)
        time.sleep(1)
        _ = create_dataset(name=cf.dataset, root=cf.root, tsp_data_info=cf.json, split="train")
