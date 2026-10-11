"""
Unit tests cho kỹ thuật Temporal Resampling cố định chuỗi độ dài T = 30 khung hình
qua nội suy tuyến tính (Linear Interpolation).
"""

import sys
import unittest
from pathlib import Path

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.interpolation import (
    temporal_resample,
    pad_or_truncate_sequence,
    HandInterpolationPipeline,
)
from src.dataset import WLASLDataset


class TestTemporalResampling(unittest.TestCase):

    def test_linear_interpolation_exactness(self):
        """
        Kiểm tra tính chính xác của thuật toán nội suy tuyến tính:
        Hàm tuyến tính y(t) = a * t + b phải được bảo toàn tuyệt đối tại mọi điểm mẫu mới.
        """
        # Chuỗi gốc 5 frames: t = 0, 1, 2, 3, 4
        # y = 3.0 * t + 10.0 => [10.0, 13.0, 16.0, 19.0, 22.0]
        t_orig = np.arange(5, dtype=np.float32)
        seq = (3.0 * t_orig + 10.0)[:, np.newaxis]  # shape (5, 1)

        # Resample về T = 30 khung hình
        res = temporal_resample(seq, target_len=30, kind="linear")
        self.assertEqual(res.shape, (30, 1))

        # Tọa độ thời gian mới: t_new chạy từ 0 đến 4 trải đều qua 30 điểm
        t_target = np.linspace(0.0, 4.0, 30, dtype=np.float32)
        expected = (3.0 * t_target + 10.0)[:, np.newaxis]

        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_endpoint_preservation(self):
        """
        Khung hình đầu tiên (t = 0) và khung hình cuối cùng (t = N - 1)
        phải được giữ nguyên vẹn 100% sau khi resample về T = 30.
        """
        seq = np.random.randn(45, 201).astype(np.float32)
        res = temporal_resample(seq, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30, 201))
        np.testing.assert_allclose(res[0], seq[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res[-1], seq[-1], rtol=1e-6, atol=1e-6)

    def test_upsampling_short_sequence(self):
        """
        Video ngắn hơn 30 frames (N < 30): Nội suy làm dày (Upsampling) thành đúng 30 frames
        mượt mà, không đệm số 0 đột ngột.
        """
        short_seq = np.random.randn(12, 63).astype(np.float32)
        res = temporal_resample(short_seq, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30, 63))
        # Không có frame nào rơi về 0 giả tạo do đệm
        self.assertFalse(np.all(res == 0.0))
        np.testing.assert_allclose(res[0], short_seq[0], rtol=1e-5)
        np.testing.assert_allclose(res[-1], short_seq[-1], rtol=1e-5)

    def test_downsampling_long_sequence(self):
        """
        Video dài hơn 30 frames (N > 30): Giảm mẫu liên tục (Downsampling) thành đúng 30 frames
        bảo toàn hình dạng chuyển động.
        """
        long_seq = np.random.randn(80, 201).astype(np.float32)
        res = temporal_resample(long_seq, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30, 201))
        np.testing.assert_allclose(res[0], long_seq[0], rtol=1e-5)
        np.testing.assert_allclose(res[-1], long_seq[-1], rtol=1e-5)

    def test_exact_sequence_length(self):
        """
        Video đã có sẵn độ dài N = 30: Trả về bản sao giữ nguyên 100%.
        """
        exact_seq = np.random.randn(30, 201).astype(np.float32)
        res = temporal_resample(exact_seq, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30, 201))
        np.testing.assert_array_equal(res, exact_seq)

    def test_holistic_landmarks_3d(self):
        """
        Kiểm tra hoạt động trên mảng 3D Holistic landmarks (T, 543, 3) hoặc (T, 67, 3).
        """
        seq_3d = np.random.randn(55, 543, 3).astype(np.float32)
        res = temporal_resample(seq_3d, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30, 543, 3))
        np.testing.assert_allclose(res[0], seq_3d[0], rtol=1e-5)
        np.testing.assert_allclose(res[-1], seq_3d[-1], rtol=1e-5)

    def test_1d_array(self):
        """
        Kiểm tra mảng 1D (T,).
        """
        seq_1d = np.linspace(0.0, 100.0, 15, dtype=np.float32)
        res = temporal_resample(seq_1d, target_len=30, kind="linear")

        self.assertEqual(res.shape, (30,))
        self.assertAlmostEqual(res[0], 0.0, places=5)
        self.assertAlmostEqual(res[-1], 100.0, places=5)

    def test_torch_tensor_support(self):
        """
        Kiểm tra hỗ trợ đầu vào dạng PyTorch Tensor (CPU/GPU, giữ nguyên device và dtype).
        """
        tensor_2d = torch.randn(40, 201, dtype=torch.float32)
        res_tensor = temporal_resample(tensor_2d, target_len=30, kind="linear")

        self.assertTrue(torch.is_tensor(res_tensor))
        self.assertEqual(res_tensor.shape, torch.Size([30, 201]))
        self.assertEqual(res_tensor.device, tensor_2d.device)
        self.assertEqual(res_tensor.dtype, tensor_2d.dtype)
        self.assertTrue(torch.allclose(res_tensor[0], tensor_2d[0], atol=1e-5))
        self.assertTrue(torch.allclose(res_tensor[-1], tensor_2d[-1], atol=1e-5))

        # So sánh đối chiếu kết quả Tensor với NumPy
        numpy_res = temporal_resample(tensor_2d.numpy(), target_len=30, kind="linear")
        np.testing.assert_allclose(res_tensor.numpy(), numpy_res, rtol=1e-5, atol=1e-5)

    def test_nearest_resampling(self):
        """
        Kiểm tra phương thức kind='nearest'.
        """
        seq = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        res = temporal_resample(seq, target_len=5, kind="nearest")

        self.assertEqual(res.shape, (5, 1))
        expected = np.array([[10.0], [10.0], [20.0], [30.0], [30.0]], dtype=np.float32)
        np.testing.assert_array_equal(res, expected)

    def test_edge_cases(self):
        """
        Kiểm tra các trường hợp biên đặc biệt:
        - N = 0: Trả về zeros shape (30, D)
        - N = 1: Lặp lại frame đó 30 lần
        - target_len <= 0: Báo lỗi ValueError
        - kind không hợp lệ: Báo lỗi ValueError
        """
        # N = 0
        empty_seq = np.empty((0, 201), dtype=np.float32)
        res_empty = temporal_resample(empty_seq, target_len=30)
        self.assertEqual(res_empty.shape, (30, 201))
        self.assertTrue(np.all(res_empty == 0.0))

        # N = 1
        single_frame = np.full((1, 201), 7.5, dtype=np.float32)
        res_single = temporal_resample(single_frame, target_len=30)
        self.assertEqual(res_single.shape, (30, 201))
        self.assertTrue(np.all(res_single == 7.5))

        # Tham số không hợp lệ
        with self.assertRaises(ValueError):
            temporal_resample(np.ones((10, 5)), target_len=0)

        with self.assertRaises(ValueError):
            temporal_resample(np.ones((10, 5)), target_len=30, kind="invalid_kind")

        with self.assertRaises(TypeError):
            temporal_resample([1, 2, 3], target_len=30)

    def test_custom_axis(self):
        """
        Kiểm tra nội suy dọc theo trục thời gian tùy chọn (ví dụ axis=1 trong (Batch, T, D)).
        """
        batch_seq = np.random.randn(4, 50, 16).astype(np.float32)
        res_batch = temporal_resample(batch_seq, target_len=30, axis=1)

        self.assertEqual(res_batch.shape, (4, 30, 16))
        np.testing.assert_allclose(res_batch[:, 0, :], batch_seq[:, 0, :], rtol=1e-5)
        np.testing.assert_allclose(res_batch[:, -1, :], batch_seq[:, -1, :], rtol=1e-5)

    def test_pad_or_truncate_sequence_resample_mode(self):
        """
        Kiểm tra pad_or_truncate_sequence tích hợp method='resample'.
        """
        seq = np.random.randn(42, 201).astype(np.float32)
        res = pad_or_truncate_sequence(seq, max_seq_len=30, method="resample")
        self.assertEqual(res.shape, (30, 201))
        np.testing.assert_allclose(res[0], seq[0], rtol=1e-5)
        np.testing.assert_allclose(res[-1], seq[-1], rtol=1e-5)

    def test_pipeline_integration(self):
        """
        Kiểm tra tích hợp HandInterpolationPipeline với temporal_resampling=True.
        """
        pipeline = HandInterpolationPipeline(
            kind="linear",
            boundary_mode="zeros",
            temporal_resampling=True,
            target_seq_len=30,
        )
        seq = np.zeros((15, 201), dtype=np.float32)
        seq[3, 63:126] = 2.0
        seq[6, 63:126] = 8.0

        processed, stats = pipeline.process(seq)
        self.assertEqual(processed.shape, (30, 201))
        self.assertEqual(stats["temporal_resampled_len"], 30)

    def test_dataset_integration(self):
        """
        Kiểm tra tích hợp WLASLDataset với cấu hình temporal_resampling=True, target_seq_len=30.
        """
        annotations_dir = Path("data/annotations")
        dataset = WLASLDataset(
            annotations_path=annotations_dir / "train.json",
            landmarks_dir="data/processed_landmarks",
            split="train",
            target_seq_len=30,
            temporal_resampling=True,
            flatten=True,
        )
        if len(dataset) > 0:
            features, label, vid_id = dataset[0]
            # features shape phải là (30, 201)
            self.assertEqual(features.shape[0], 30)
            self.assertEqual(features.shape[1], 201)
            self.assertEqual(features.dim(), 2)


if __name__ == "__main__":
    unittest.main()
