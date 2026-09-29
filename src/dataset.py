"""
Dataset and DataLoader definitions for Isolated Sign Language Recognition.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WLASLDataset(Dataset):
    """
    Dataset loader for WLASL landmarks / features.
    """
    def __init__(self, annotations_path, landmarks_dir, split="train", transform=None):
        self.annotations_path = annotations_path
        self.landmarks_dir = landmarks_dir
        self.split = split
        self.transform = transform
        self.samples = []
        # TODO: Load annotations and filter by split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: Load .npy landmark file and return tensor + label
        pass
