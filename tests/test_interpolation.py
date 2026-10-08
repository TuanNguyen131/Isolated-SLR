"""
Unit tests cho module nội suy (Interpolation) và Zero-Padding các frame mất dấu bàn tay.
"""

import sys
import unittest
from pathlib import Path

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.interpolation import (
    detect_missing_frames,
    interpolate_sequence_2d,
    interpolate_hand_landmarks,
    interpolate_vector201,
    interpolate_holistic_landmarks,
    pad_or_truncate_sequence,
    HandInterpolationPipeline,
)


class TestHandInterpolation(unittest.TestCase):

    def test_detect_missing_frames(self):
        # Tạo mảng (5, 21, 3): frame 0, 4 hợp lệ; frame 1, 2 là zero; frame 3 là nan
        arr = np.zeros((5, 21, 3), dtype=np.float32)
        arr[0] = 1.0
        arr[4] = 2.0
        arr[3, 0, 0] = np.nan

        mask = detect_missing_frames(arr)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_interpolate_internal_gap(self):
        # T = 5, D = 2
        # Frame 0: [0, 0]
        # Frame 1: [10, 20] (hợp lệ)
        # Frame 2: [0, 0]   (mất dấu nội tại)
        # Frame 3: [30, 40] (hợp lệ)
        # Frame 4: [0, 0]   (vùng biên sau)
        seq = np.zeros((5, 2), dtype=np.float32)
        seq[1] = [10.0, 20.0]
        seq[3] = [30.0, 40.0]

        res, stats = interpolate_sequence_2d(seq, boundary_mode="zeros")

        # Frame 0 (leading) phải là 0.0 (Zero-Padding)
        np.testing.assert_array_equal(res[0], [0.0, 0.0])
        # Frame 1 giữ nguyên
        np.testing.assert_array_equal(res[1], [10.0, 20.0])
        # Frame 2 (nội suy giữa frame 1 và 3): trung bình tuyến tính = [20.0, 30.0]
        np.testing.assert_allclose(res[2], [20.0, 30.0], rtol=1e-5)
        # Frame 3 giữ nguyên
        np.testing.assert_array_equal(res[3], [30.0, 40.0])
        # Frame 4 (trailing) phải là 0.0 (Zero-Padding)
        np.testing.assert_array_equal(res[4], [0.0, 0.0])

        self.assertEqual(stats["interpolated_frames"], 1)
        self.assertEqual(stats["zero_padded_frames"], 2)

    def test_completely_missing_hand(self):
        # Bàn tay hoàn toàn không xuất hiện (cử chỉ 1 tay)
        seq = np.zeros((10, 63), dtype=np.float32)
        res, stats = interpolate_hand_landmarks(seq, boundary_mode="zeros")

        self.assertTrue(np.all(res == 0.0))
        self.assertEqual(stats["valid_frames"], 0)
        self.assertEqual(stats["zero_padded_frames"], 10)
        self.assertEqual(stats["interpolated_frames"], 0)

    def test_single_valid_frame(self):
        # Chỉ có 1 frame hợp lệ duy nhất
        seq = np.zeros((5, 63), dtype=np.float32)
        seq[2] = 5.0

        res, stats = interpolate_hand_landmarks(seq, boundary_mode="zeros")
        # Frame 2 giữ nguyên 5.0, các frame khác là 0.0
        np.testing.assert_array_equal(res[2], np.full(63, 5.0))
        self.assertTrue(np.all(res[:2] == 0.0))
        self.assertTrue(np.all(res[3:] == 0.0))
        self.assertEqual(stats["zero_padded_frames"], 4)

    def test_max_gap_size_limit(self):
        # Frame 0: 0, Frame 1: 10, Frame 2, 3, 4: missing (gap=3), Frame 5: 50
        seq = np.zeros((6, 1), dtype=np.float32)
        seq[1] = 10.0
        seq[5] = 50.0

        # max_gap_size = 2 -> gap 3 frame sẽ KHÔNG được nội suy, để nguyên 0.0
        res, stats = interpolate_sequence_2d(seq, max_gap_size=2, boundary_mode="zeros")
        self.assertTrue(np.all(res[2:5] == 0.0))
        self.assertEqual(stats["interpolated_frames"], 0)

        # max_gap_size = 3 (hoặc None) -> gap 3 frame ĐƯỢC nội suy
        res2, stats2 = interpolate_sequence_2d(seq, max_gap_size=3, boundary_mode="zeros")
        self.assertEqual(stats2["interpolated_frames"], 3)
        np.testing.assert_allclose(res2[2:5, 0], [20.0, 30.0, 40.0], rtol=1e-5)

    def test_vector201_interpolation(self):
        # Mảng shape (10, 201)
        # Left hand: 0:63, Right hand: 63:126, Pose: 126:201
        seq = np.zeros((10, 201), dtype=np.float32)

        # LH: frame 2 và 4 có giá trị -> frame 3 bị missing
        seq[2, 0:63] = 1.0
        seq[4, 0:63] = 3.0

        # RH: frame 1 và 5 có giá trị -> frame 2, 3, 4 bị missing
        seq[1, 63:126] = 10.0
        seq[5, 63:126] = 50.0

        # Pose: có giá trị suốt từ frame 0 đến 9
        seq[:, 126:201] = 0.5

        out, stats = interpolate_vector201(seq, boundary_mode="zeros")

        self.assertEqual(out.shape, (10, 201))
        # Kiểm tra LH frame 3 được nội suy thành 2.0
        np.testing.assert_allclose(out[3, 0:63], np.full(63, 2.0), rtol=1e-5)
        # Kiểm tra LH frame 0, 1 là zero-padded
        self.assertTrue(np.all(out[0:2, 0:63] == 0.0))
        # Kiểm tra RH frame 3 được nội suy thành 30.0
        np.testing.assert_allclose(out[3, 63:126], np.full(63, 30.0), rtol=1e-5)
        # Pose không bị thay đổi
        np.testing.assert_allclose(out[:, 126:201], np.full((10, 75), 0.5), rtol=1e-5)

    def test_holistic_interpolation(self):
        # Mảng 543 điểm: (8, 543, 3)
        seq = np.zeros((8, 543, 3), dtype=np.float32)
        # LH: 501..522. Đặt frame 1 và frame 3
        seq[1, 501:522] = 2.0
        seq[3, 501:522] = 4.0

        out, stats = interpolate_holistic_landmarks(seq, keypoint_mode="full", boundary_mode="zeros")
        self.assertEqual(out.shape, (8, 543, 3))
        # Frame 2 của LH được nội suy thành 3.0
        np.testing.assert_allclose(out[2, 501:522], np.full((21, 3), 3.0), rtol=1e-5)
        # Frame 0 của LH là 0.0
        self.assertTrue(np.all(out[0, 501:522] == 0.0))

    def test_pad_or_truncate_sequence(self):
        # Case 1: T < max_seq_len (Cần zero-padding)
        short_seq = np.ones((20, 201), dtype=np.float32)
        padded = pad_or_truncate_sequence(short_seq, max_seq_len=60, padding_mode="zeros")
        self.assertEqual(padded.shape, (60, 201))
        np.testing.assert_array_equal(padded[:20], short_seq)
        self.assertTrue(np.all(padded[20:] == 0.0))

        # Case 2: T > max_seq_len (Uniform resampling)
        long_seq = np.ones((100, 201), dtype=np.float32)
        truncated = pad_or_truncate_sequence(long_seq, max_seq_len=60)
        self.assertEqual(truncated.shape, (60, 201))

        # Case 3: T == max_seq_len (Không đổi)
        exact_seq = np.ones((60, 201), dtype=np.float32)
        res = pad_or_truncate_sequence(exact_seq, max_seq_len=60)
        self.assertEqual(res.shape, (60, 201))

    def test_pipeline_class(self):
        pipeline = HandInterpolationPipeline(
            kind="linear",
            boundary_mode="zeros",
            max_seq_len=30,
        )
        seq = np.zeros((15, 201), dtype=np.float32)
        seq[5, 63:126] = 1.0
        seq[7, 63:126] = 3.0

        processed, stats = pipeline.process(seq)
        # Kết quả phải có shape (30, 201) do đã pad_sequence
        self.assertEqual(processed.shape, (30, 201))
        # Frame 6 của RH phải là 2.0
        np.testing.assert_allclose(processed[6, 63:126], np.full(63, 2.0), rtol=1e-5)
        # Frame 15..29 là zero padding độ dài chuỗi
        self.assertTrue(np.all(processed[15:] == 0.0))


if __name__ == "__main__":
    unittest.main()
