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
    Hỗ trợ tải trực tiếp từ các file .npy riêng lẻ hoặc file HDF5 (.h5) tổng hợp.
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
        h5_path: Optional[Union[str, Path]] = None,
    ):
        self.annotations_path = Path(annotations_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.split = split
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.flatten = flatten
        self.transform = transform

        if h5_path:
            self.h5_path = Path(h5_path)
        elif (self.landmarks_dir / "wlasl100_landmarks.h5").is_file():
            self.h5_path = self.landmarks_dir / "wlasl100_landmarks.h5"
        else:
            self.h5_path = None

        self._h5_file = None

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

            # Lọc chỉ lấy các mẫu đã có file landmark .npy hoặc trong HDF5
            if only_available:
                h5_keys = set()
                if self.h5_path and self.h5_path.is_file():
                    try:
                        import h5py
                        with h5py.File(str(self.h5_path), "r") as hf:
                            if self.split in hf:
                                h5_keys = set(hf[self.split].keys())
                            else:
                                h5_keys = set(hf.keys())
                    except Exception:
                        pass

                for s in all_samples:
                    vid = s["video_id"]
                    npy_path = self.landmarks_dir / f"{vid}.npy"
                    if npy_path.is_file() or vid in h5_keys:
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

        landmarks = None
        npy_path = self.landmarks_dir / f"{video_id}.npy"

        # 1. Thử tải từ file .npy
        if npy_path.is_file():
            try:
                landmarks = np.load(npy_path).astype(np.float32)
            except Exception:
                landmarks = None

        # 2. Nếu chưa có, thử tải từ file HDF5
        if landmarks is None and self.h5_path and self.h5_path.is_file():
            try:
                if self._h5_file is None:
                    import h5py
                    self._h5_file = h5py.File(str(self.h5_path), "r")
                h5_key = f"{self.split}/{video_id}"
                if h5_key in self._h5_file:
                    landmarks = np.array(self._h5_file[h5_key], dtype=np.float32)
                elif video_id in self._h5_file:
                    landmarks = np.array(self._h5_file[video_id], dtype=np.float32)
            except Exception:
                landmarks = None

        # 3. Nếu không tìm thấy -> tensor rỗng mặc định
        if landmarks is None or landmarks.size == 0:
            landmarks = np.zeros((1, 543, 3), dtype=np.float32)

        # Chuẩn hóa độ dài chuỗi theo thời gian (temporal padding / truncation)
        num_frames = landmarks.shape[0]
        if num_frames > self.max_seq_len:
            indices = np.linspace(0, num_frames - 1, self.max_seq_len, dtype=int)
            landmarks = landmarks[indices]
        elif num_frames < self.max_seq_len:
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

    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


def create_dataloader(
    annotations_path: Union[str, Path],
    landmarks_dir: Union[str, Path] = "data/landmarks",
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    max_seq_len: int = 60,
    flatten: bool = True,
    h5_path: Optional[Union[str, Path]] = None,
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
        h5_path=h5_path,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
