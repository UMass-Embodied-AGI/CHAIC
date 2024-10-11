import torch
from torch.utils.data import DataLoader, Dataset
import glob
import json
import tqdm
import numpy as np
import os
from functools import partial
import copy

class CustomDataset(Dataset):
    def __init__(self, json_file, max_data = -1):
        lst = json.load(open(json_file, "r"))
        self.data_list = []
        if max_data == -1:
            max_data = len(lst)
        else:
            max_data = min(max_data, len(lst))

        for i in range(max_data):
            self.data_list.append(lst[i])
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ret = self.data_list[idx].copy()
        return ret