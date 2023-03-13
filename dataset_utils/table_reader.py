import sys
import torch
import pandas as pd
import yaml
from PIL import Image
import os
import numpy as np
import timm


def table_read(file_paths: list, key_type="index") -> dict:
    result = dict()
    if key_type not in ["index", "file_name"]:
        raise ValueError('key_type should be ["index", "file_name"]')
    for file in file_paths:
        table = pd.read_csv(file, compression="bz2" if file[-3:] == "bz2" else None)
        table_info = table.iloc[:, :4]
        table_predict = table.iloc[:, 4:]

        table_len = table.shape[0]
        for row_index in range(table_len):
            info = table_info.iloc[row_index].to_dict()
            predict = table_predict.iloc[row_index].to_list()
            if key_type == "file_name":
                result[info["filename"]] = info
                result[info["filename"]].update(dict(predict=predict))
            elif key_type == "index":
                result[info["file_index"]] = info
                result[info["file_index"]].update(dict(predict=predict))
    return result
