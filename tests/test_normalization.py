"""
Unit tests cho module chuẩn hóa tọa độ (dời gốc về cổ tay / khoảng cách hai vai).
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.normalization import (
    compute_shoulder_reference,
    compute_sequence_shoulder_reference,
    normalize_hand_landmarks,
    normalize_pose_landmarks,
    normalize_landmarks_wrist_shoulder,
    normalize_vector201,
    normalize_holistic_landmarks,
    CoordinateNormalizer,
)


class TestCoordinateNormalization(unittest.TestCase):

    def test_compute_shoulder_reference_ideal(self):
        # Vai trái (11) tại (0.6, 0.4, 0.0), Vai phải (12) tại (0.2, 0.4, 0.0)
        pose = np.zeros((25, 3), dtype=np.float32)
        pose[11] = [0.6, 0.4, 0.1]
        pose[12] = [0.2, 0.4, -0.1]

        center, scale = compute_shoulder_reference(pose)

        # Trung điểm: ((0.6+0.2)/2, (0.4+0.4)/2) = (0.4, 0.4, 0.0)
        np.testing.assert_allclose(center[:2], [0.4, 0.4], rtol=1e-5)
        self.assertAlmostEqual(center[2], 0.0, places=5)
        # Khoảng cách: 0.6 - 0.2 = 0.4
        self.assertAlmostEqual(scale, 0.4, places=5)

    def test_compute_shoulder_reference_fallback(self):
        # Mất cả 2 vai
        pose = np.zeros((25, 3), dtype=np.float32)
        center, scale = compute_shoulder_reference(pose)
        # Fallback về tâm (0.5, 0.5) và scale = 1.0
        np.testing.assert_allclose(center[:2], [0.5, 0.5], rtol=1e-5)
        self.assertEqual(scale, 1.0)

    def test_normalize_hand_landmarks_wrist_origin(self):
        # Bàn tay 21 điểm: Cổ tay (point 0) tại (0.8, 0.6, 0.1)
        hand = np.zeros((21, 3), dtype=np.float32)
        wrist = np.array([0.8, 0.6, 0.1], dtype=np.float32)
        thumb_tip = np.array([0.9, 0.65, 0.15], dtype=np.float32)

        hand[0] = wrist
        hand[4] = thumb_tip  # Ngón cái
        # Các điểm khác cũng có giá trị để hand hợp lệ
        hand[1:] = thumb_tip

        scale = 0.5  # Khoảng cách 2 vai

        norm_hand = normalize_hand_landmarks(
            hand, scale=scale, hand_origin="wrist", store_wrist_origin=False
        )

        # Điểm cổ tay (0) phải là (0, 0, 0)
        np.testing.assert_allclose(norm_hand[0], [0.0, 0.0, 0.0], atol=1e-6)

        # Điểm ngón cái (4): (thumb - wrist) / scale
        expected_thumb = (thumb_tip - wrist) / scale
        np.testing.assert_allclose(norm_hand[4], expected_thumb, rtol=1e-5)

    def test_normalize_hand_landmarks_missing_hand(self):
        # Bàn tay mất dấu (toàn bộ là 0.0)
        hand = np.zeros((21, 3), dtype=np.float32)
        norm_hand = normalize_hand_landmarks(hand, scale=0.5, hand_origin="wrist")

        # Phải bảo toàn toàn bộ 0.0 (không chia hay dịch chuyển ảo)
        self.assertTrue(np.all(norm_hand == 0.0))

    def test_normalize_pose_landmarks(self):
        pose = np.zeros((25, 3), dtype=np.float32)
        pose[11] = [0.6, 0.4, 0.0]  # Vai trái
        pose[12] = [0.2, 0.4, 0.0]  # Vai phải

        center, scale = compute_shoulder_reference(pose)
        norm_pose = normalize_pose_landmarks(pose, center=center, scale=scale)

        # Sau chuẩn hóa:
        # Trung điểm 2 vai phải ở gốc (0, 0)
        mid_norm = (norm_pose[11, :2] + norm_pose[12, :2]) / 2.0
        np.testing.assert_allclose(mid_norm, [0.0, 0.0], atol=1e-6)
        # Khoảng cách giữa 2 vai sau chuẩn hóa phải chính xác bằng 1.0
        dist_norm = np.linalg.norm(norm_pose[11, :2] - norm_pose[12, :2])
        self.assertAlmostEqual(dist_norm, 1.0, places=5)

    def test_normalize_vector201_sequence(self):
        T = 5
        seq = np.zeros((T, 201), dtype=np.float32)

        # Pose: vai trái (11) tại (0.6, 0.4), vai phải (12) tại (0.2, 0.4)
        for t in range(T):
            # pose slice: [126:201], mỗi điểm 3 tọa độ
            seq[t, 126 + 11*3 : 126 + 11*3 + 3] = [0.6, 0.4, 0.0]
            seq[t, 126 + 12*3 : 126 + 12*3 + 3] = [0.2, 0.4, 0.0]
            # Left Hand slice: [0:63]
            # Cổ tay trái tại (0.7, 0.8, 0.1)
            seq[t, 0:3] = [0.7, 0.8, 0.1]
            seq[t, 3:63] = np.tile([0.75, 0.85, 0.12], 20)
            # Right Hand slice: [63:126]
            # Cổ tay phải tại (0.1, 0.8, -0.1)
            seq[t, 63:66] = [0.1, 0.8, -0.1]
            seq[t, 66:126] = np.tile([0.15, 0.85, -0.08], 20)


        norm_seq = normalize_vector201(seq, sequence_level=True, hand_origin="wrist")

        self.assertEqual(norm_seq.shape, (T, 201))

        # Điểm cổ tay trái (0:3) của LH phải là (0, 0, 0)
        for t in range(T):
            np.testing.assert_allclose(norm_seq[t, 0:3], [0.0, 0.0, 0.0], atol=1e-6)
            # Điểm cổ tay phải (63:66) của RH phải là (0, 0, 0)
            np.testing.assert_allclose(norm_seq[t, 63:66], [0.0, 0.0, 0.0], atol=1e-6)

            # Khoảng cách 2 vai trong pose phải là 1.0
            ls = norm_seq[t, 126 + 11*3 : 126 + 11*3 + 2]
            rs = norm_seq[t, 126 + 12*3 : 126 + 12*3 + 2]
            self.assertAlmostEqual(np.linalg.norm(ls - rs), 1.0, places=5)

    def test_normalize_holistic_landmarks(self):
        seq = np.zeros((4, 543, 3), dtype=np.float32)
        # Vai trái (11) và vai phải (12)
        seq[:, 11] = [0.7, 0.3, 0.0]
        seq[:, 12] = [0.3, 0.3, 0.0]
        # Tay trái: 501..522
        seq[:, 501] = [0.8, 0.7, 0.0]  # Left wrist
        seq[:, 502:522] = [0.85, 0.75, 0.0]
        # Tay phải: 522..543
        seq[:, 522] = [0.2, 0.7, 0.0]  # Right wrist
        seq[:, 523:543] = [0.25, 0.75, 0.0]

        norm_seq = normalize_holistic_landmarks(seq, keypoint_mode="full", hand_origin="wrist")

        self.assertEqual(norm_seq.shape, (4, 543, 3))
        # Cổ tay trái (501) và cổ tay phải (522) phải là (0, 0, 0)
        np.testing.assert_allclose(norm_seq[:, 501], np.zeros((4, 3)), atol=1e-6)
        np.testing.assert_allclose(norm_seq[:, 522], np.zeros((4, 3)), atol=1e-6)
        # Khoảng cách vai = 1.0
        dists = np.linalg.norm(norm_seq[:, 11, :2] - norm_seq[:, 12, :2], axis=1)
        np.testing.assert_allclose(dists, np.ones(4), rtol=1e-5)

    def test_coordinate_normalizer_pipeline(self):
        normalizer = CoordinateNormalizer(hand_origin="wrist")
        vec = np.zeros((201,), dtype=np.float32)
        vec[126 + 11*3 : 126 + 11*3 + 2] = [0.6, 0.3]
        vec[126 + 12*3 : 126 + 12*3 + 2] = [0.2, 0.3]
        vec[0:3] = [0.65, 0.7, 0.0]
        vec[3:6] = [0.70, 0.75, 0.0]

        res = normalizer.normalize(vec)
        self.assertEqual(res.shape, (201,))
        # Điểm cổ tay trái là (0, 0, 0)
        np.testing.assert_allclose(res[0:3], [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
