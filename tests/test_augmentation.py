"""
Unit tests cho module Data Augmentation trên vector tọa độ Landmarks (Isolated-SLR).
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.augmentation import (
    add_gaussian_noise,
    rotate_landmarks,
    scale_landmarks,
    translate_landmarks,
    dropout_landmarks,
    CoordinateAugmentor,
    SignAugmentor,
)
from src.dataset import WLASLDataset, create_dataloader


class TestCoordinateAugmentation(unittest.TestCase):

    def test_gaussian_noise_and_zero_masking(self):
        """
        Kiểm tra thêm nhiễu Gaussian và tính năng bảo toàn điểm zero (mask_zeros=True).
        """
        # Tạo mảng (10, 67, 3)
        # Point 0: có giá trị, Point 1: là điểm zero (0, 0, 0)
        coords = np.zeros((10, 67, 3), dtype=np.float32)
        coords[:, 0] = [1.0, 2.0, 3.0]

        noisy = add_gaussian_noise(coords, sigma=0.05, mask_zeros=True, seed=42)

        self.assertEqual(noisy.shape, coords.shape)
        # Point 0 phải bị thay đổi do nhiễu
        self.assertFalse(np.allclose(noisy[:, 0], coords[:, 0]))
        # Point 1 (zero landmark) phải giữ nguyên tuyệt đối 0.0
        np.testing.assert_array_equal(noisy[:, 1:], 0.0)

        # Khi mask_zeros=False, điểm zero cũng bị thêm nhiễu
        noisy_all = add_gaussian_noise(coords, sigma=0.05, mask_zeros=False, seed=42)
        self.assertFalse(np.all(noisy_all[:, 1] == 0.0))

    def test_gaussian_noise_probability(self):
        """
        Kiểm tra xác suất áp dụng prob: khi prob=0.0 thì không có gì thay đổi.
        """
        coords = np.ones((5, 201), dtype=np.float32)
        out = add_gaussian_noise(coords, sigma=0.1, prob=0.0)
        np.testing.assert_array_equal(out, coords)

    def test_rotation_z_axis_geometry(self):
        """
        Kiểm tra xoay 90 độ quanh trục Z:
        (1, 0, 0) -> (0, 1, 0)
        (0, 1, 0) -> (-1, 0, 0)
        (0, 0, 1) -> (0, 0, 1)
        """
        coords = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # zero landmark
        ], dtype=np.float32).reshape(1, 4, 3)

        rotated = rotate_landmarks(
            coords,
            angle_deg=90.0,
            axis="z",
            center="origin",
            mask_zeros=True,
        )

        # Kiểm tra tọa độ điểm 0
        np.testing.assert_allclose(rotated[0, 0], [0.0, 1.0, 0.0], atol=1e-5)
        # Kiểm tra tọa độ điểm 1
        np.testing.assert_allclose(rotated[0, 1], [-1.0, 0.0, 0.0], atol=1e-5)
        # Điểm trên trục Z không đổi
        np.testing.assert_allclose(rotated[0, 2], [0.0, 0.0, 1.0], atol=1e-5)
        # Điểm zero được bảo toàn
        np.testing.assert_array_equal(rotated[0, 3], [0.0, 0.0, 0.0])

        # Khoảng cách đến gốc tọa độ được bảo toàn (phép đẳng cự)
        self.assertAlmostEqual(float(np.linalg.norm(rotated[0, 0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(rotated[0, 1])), 1.0, places=5)

    def test_rotation_center_modes(self):
        """
        Kiểm tra xoay quanh trọng tâm (center='mean') và tâm tùy chọn.
        """
        # Hai điểm đối xứng qua tâm (5, 0, 0): (4, 0, 0) và (6, 0, 0)
        coords = np.array([
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ], dtype=np.float32).reshape(1, 2, 3)

        # Xoay 180 độ quanh mean (5, 0, 0) sẽ hoán đổi vị trí của 2 điểm
        rotated = rotate_landmarks(coords, angle_deg=180.0, axis="z", center="mean")
        np.testing.assert_allclose(rotated[0, 0], [6.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(rotated[0, 1], [4.0, 0.0, 0.0], atol=1e-5)

    def test_scale_landmarks(self):
        """
        Kiểm tra co giãn tỉ lệ (scale).
        """
        coords = np.array([
            [2.0, 4.0, 6.0],
            [0.0, 0.0, 0.0],  # zero
        ], dtype=np.float32).reshape(1, 2, 3)

        scaled = scale_landmarks(coords, scale_factor=1.5, center="origin", mask_zeros=True)

        # Điểm 0 nhân với 1.5
        np.testing.assert_allclose(scaled[0, 0], [3.0, 6.0, 9.0], atol=1e-5)
        # Điểm zero giữ nguyên
        np.testing.assert_array_equal(scaled[0, 1], [0.0, 0.0, 0.0])

    def test_translate_landmarks(self):
        """
        Kiểm tra dịch chuyển tịnh tiến (translation).
        """
        coords = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],  # zero
        ], dtype=np.float32).reshape(1, 2, 3)

        shifted = translate_landmarks(coords, shift=[0.1, -0.2, 0.3], mask_zeros=True)
        np.testing.assert_allclose(shifted[0, 0], [1.1, 1.8, 3.3], atol=1e-5)
        np.testing.assert_array_equal(shifted[0, 1], [0.0, 0.0, 0.0])

    def test_dropout_landmarks(self):
        """
        Kiểm tra ẩn ngẫu nhiên (dropout).
        """
        coords = np.ones((10, 67, 3), dtype=np.float32)
        dropped = dropout_landmarks(coords, drop_prob=0.2, prob=1.0, seed=42)

        self.assertEqual(dropped.shape, coords.shape)
        # Phải có một số điểm bị đưa về 0
        num_zeros = np.sum(np.all(dropped == 0.0, axis=-1))
        self.assertGreater(num_zeros, 0)
        self.assertLess(num_zeros, 10 * 67)

    def test_torch_tensor_support(self):
        """
        Kiểm tra hỗ trợ toàn diện với PyTorch Tensor (giữ nguyên device, dtype, tensor type).
        """
        t = torch.randn(30, 201, dtype=torch.float32)

        # 1. Noise
        noisy_t = add_gaussian_noise(t, sigma=0.02, prob=1.0, seed=123)
        self.assertTrue(torch.is_tensor(noisy_t))
        self.assertEqual(noisy_t.shape, t.shape)
        self.assertEqual(noisy_t.dtype, t.dtype)

        # 2. Rotation
        rot_t = rotate_landmarks(t, angle_deg=12.0, axis="z", prob=1.0)
        self.assertTrue(torch.is_tensor(rot_t))
        self.assertEqual(rot_t.shape, t.shape)

        # 3. Scaling
        scale_t = scale_landmarks(t, scale_factor=1.1, prob=1.0)
        self.assertTrue(torch.is_tensor(scale_t))
        self.assertEqual(scale_t.shape, t.shape)

        # 4. Pipeline
        augmentor = CoordinateAugmentor(seed=42)
        aug_t = augmentor(t)
        self.assertTrue(torch.is_tensor(aug_t))
        self.assertEqual(aug_t.shape, t.shape)

    def test_shape_compatibility(self):
        """
        Kiểm tra độ tương thích với các shape khác nhau:
        - 1D: (201,)
        - 2D: (30, 201)
        - 3D: (30, 543, 3), (30, 67, 3)
        """
        augmentor = CoordinateAugmentor(seed=99)

        # 1D
        vec_1d = np.random.randn(201).astype(np.float32)
        out_1d = augmentor(vec_1d)
        self.assertEqual(out_1d.shape, (201,))

        # 2D
        seq_2d = np.random.randn(30, 201).astype(np.float32)
        out_2d = augmentor(seq_2d)
        self.assertEqual(out_2d.shape, (30, 201))

        # 3D Holistic 543 điểm
        seq_543 = np.random.randn(30, 543, 3).astype(np.float32)
        out_543 = augmentor(seq_543)
        self.assertEqual(out_543.shape, (30, 543, 3))

    def test_reproducibility_with_seed(self):
        """
        Kiểm tra tính tái lập khi truyền cùng random seed.
        """
        data = np.random.randn(30, 201).astype(np.float32)
        aug1 = CoordinateAugmentor(seed=12345)
        aug2 = CoordinateAugmentor(seed=12345)

        res1 = aug1(data)
        res2 = aug2(data)
        np.testing.assert_array_equal(res1, res2)

    def test_dataset_and_dataloader_integration(self):
        """
        Kiểm tra tích hợp trực tiếp vào WLASLDataset và create_dataloader.
        """
        ann_path = Path("data/annotations/wlasl_100/train.json")
        if not ann_path.is_file():
            ann_path = Path("data/annotations/train.json")

        augmentor = CoordinateAugmentor(noise_sigma=0.01, rotation_max_angle=8.0, seed=42)

        if ann_path.is_file():
            dataset = WLASLDataset(
                annotations_path=ann_path,
                landmarks_dir="data/processed_landmarks",
                split="train",
                target_seq_len=30,
                temporal_resampling=True,
                transform=augmentor,
            )

            if len(dataset) > 0:
                features, label, vid_id = dataset[0]
                self.assertEqual(features.shape, torch.Size([30, 201]))
                self.assertTrue(torch.is_tensor(features))

                # DataLoader với cờ augment=True
                loader = create_dataloader(
                    annotations_path=ann_path,
                    landmarks_dir="data/processed_landmarks",
                    split="train",
                    batch_size=4,
                    augment=True,
                )
                batch_features, batch_labels, _ = next(iter(loader))
                self.assertEqual(batch_features.shape[1], 30)
                self.assertEqual(batch_features.shape[2], 201)


if __name__ == "__main__":
    unittest.main()
