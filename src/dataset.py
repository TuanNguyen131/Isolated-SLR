"""
Dataset and DataLoader definitions for Isolated Sign Language Recognition.
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WLASLDataset(Dataset):
    """
    Dataset loader cho các chuỗi đặc trưng landmarks trích xuất từ MediaPipe Holistic.
    """
    def __init__(
        self,
        annotations_path: Union[str, Path],
        landmarks_dir: Union[str, Path] = "data/landmarks",
        split: str = "train",
        num_classes: int = 100,
        max_seq_len: int = 60,
        flatten: bool = True,
        only_available: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.annotations_path = Path(annotations_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.split = split
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.flatten = flatten
        self.transform = transform

        self.gloss_to_id = {}
        mapping_file = self.annotations_path.parent / "class_mapping.json"
        if not mapping_file.is_file() and self.annotations_path.is_dir():
            mapping_file = self.annotations_path / "class_mapping.json"
        if mapping_file.is_file():
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    cmap = json.load(f)
                    self.gloss_to_id = cmap.get("gloss_to_id", cmap)
            except Exception:
                pass

        self.samples: List[Dict[str, Any]] = []
        if self.annotations_path.exists():
            from .utils.filter_wlasl import load_wlasl_split
            all_samples = load_wlasl_split(
                self.annotations_path,
                split=self.split,
                num_classes=self.num_classes,
                only_available=False,
            )

            # Lọc chỉ lấy các mẫu đã có file landmark .npy
            if only_available and self.landmarks_dir.exists():
                for s in all_samples:
                    npy_path = self.landmarks_dir / f"{s['video_id']}.npy"
                    if npy_path.is_file():
                        self.samples.append(s)
            else:
                self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        gloss = sample.get("gloss", "")
        raw_label = sample.get("label")
        if raw_label is not None:
            label = int(raw_label)
        else:
            label = int(self.gloss_to_id.get(gloss, 0))
        npy_path = self.landmarks_dir / f"{video_id}.npy"

        # Tải mảng landmarks: (T, num_keypoints, coord_dim)
        if npy_path.is_file():
            landmarks = np.load(npy_path).astype(np.float32)
        else:
            # Nếu file thiếu -> trả về tensor rỗng
            landmarks = np.zeros((1, 543, 3), dtype=np.float32)

        # Chuẩn hóa độ dài chuỗi theo thời gian (temporal padding / truncation)
        num_frames = landmarks.shape[0]
        if num_frames > self.max_seq_len:
            # Lấy mẫu đều các frame qua thời gian
            indices = np.linspace(0, num_frames - 1, self.max_seq_len, dtype=int)
            landmarks = landmarks[indices]
        elif num_frames < self.max_seq_len:
            # Zero-padding cho các frame còn thiếu ở cuối
            pad_len = self.max_seq_len - num_frames
            padding = np.zeros((pad_len, *landmarks.shape[1:]), dtype=np.float32)
            landmarks = np.concatenate([landmarks, padding], axis=0)

        # Làm phẳng đặc trưng mỗi frame: (max_seq_len, num_keypoints * coord_dim)
        if self.flatten:
            landmarks = landmarks.reshape(self.max_seq_len, -1)

        # Chuyển đổi thành PyTorch Tensors
        features_tensor = torch.from_numpy(landmarks).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            features_tensor = self.transform(features_tensor)

        return features_tensor, label_tensor, video_id


def create_dataloader(
    annotations_path: Union[str, Path],
    landmarks_dir: Union[str, Path] = "data/landmarks",
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    max_seq_len: int = 60,
    flatten: bool = True,
) -> DataLoader:
    """Tạo DataLoader tiện lợi cho việc huấn luyện và kiểm thử."""
    is_train = split == "train"
    shuffle = is_train if shuffle is None else shuffle

    dataset = WLASLDataset(
        annotations_path=annotations_path,
        landmarks_dir=landmarks_dir,
        split=split,
        max_seq_len=max_seq_len,
        flatten=flatten,
        only_available=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
