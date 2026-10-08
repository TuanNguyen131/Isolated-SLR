"""
Module trích xuất đặc trưng tư thế (Pose), bàn tay (Hands), và khuôn mặt (Face)
sử dụng MediaPipe Holistic cho bài toán nhận dạng ngôn ngữ ký hiệu (Isolated-SLR).
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

from .holistic_config import HolisticConfig, UNIQUE_LIPS_INDICES
from src.data.interpolation import interpolate_holistic_landmarks


class MediaPipeHolisticExtractor:
    """
    Bộ trích xuất landmark toàn diện từ ảnh hoặc video bằng MediaPipe Holistic.
    Hỗ trợ chuẩn hóa bất biến theo tỉ lệ và vị trí người ra dấu (scale & position invariant).
    """

    def __init__(self, config: Optional[HolisticConfig] = None):
        self.config = config or HolisticConfig()
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Khởi tạo đối tượng Holistic của MediaPipe
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            refine_face_landmarks=self.config.refine_face_landmarks,
            enable_segmentation=self.config.enable_segmentation,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        if hasattr(self, "holistic") and self.holistic:
            self.holistic.close()

    def _extract_landmarks_from_result(self, results: Any) -> np.ndarray:
        """
        Trích xuất và ghép nối các landmark thành mảng numpy kích thước (K, C).
        K = num_keypoints, C = coord_dim (2, 3 hoặc 4).
        """
        coord_dim = self.config.coord_dim
        include_z = self.config.include_z
        include_vis = self.config.include_visibility
        fill_val = np.nan if self.config.fill_missing == "nan" else 0.0

        # 1. Pose landmarks (33 điểm)
        if results.pose_landmarks:
            pose_pts = []
            for lm in results.pose_landmarks.landmark:
                pt = [lm.x, lm.y]
                if include_z:
                    pt.append(lm.z)
                if include_vis:
                    pt.append(lm.visibility)
                pose_pts.append(pt)
            pose_arr = np.array(pose_pts, dtype=np.float32)
        else:
            pose_arr = np.full((33, coord_dim), fill_val, dtype=np.float32)

        # 2. Left Hand landmarks (21 điểm)
        if results.left_hand_landmarks:
            lh_pts = []
            for lm in results.left_hand_landmarks.landmark:
                pt = [lm.x, lm.y]
                if include_z:
                    pt.append(lm.z)
                if include_vis:
                    pt.append(1.0)  # Tay phát hiện được -> visibility = 1.0
                lh_pts.append(pt)
            lh_arr = np.array(lh_pts, dtype=np.float32)
        else:
            lh_arr = np.full((21, coord_dim), fill_val, dtype=np.float32)

        # 3. Right Hand landmarks (21 điểm)
        if results.right_hand_landmarks:
            rh_pts = []
            for lm in results.right_hand_landmarks.landmark:
                pt = [lm.x, lm.y]
                if include_z:
                    pt.append(lm.z)
                if include_vis:
                    pt.append(1.0)
                rh_pts.append(pt)
            rh_arr = np.array(rh_pts, dtype=np.float32)
        else:
            rh_arr = np.full((21, coord_dim), fill_val, dtype=np.float32)

        # 4. Face landmarks
        mode = self.config.keypoint_mode
        if mode == "full":
            if results.face_landmarks:
                face_pts = []
                for lm in results.face_landmarks.landmark:
                    pt = [lm.x, lm.y]
                    if include_z:
                        pt.append(lm.z)
                    if include_vis:
                        pt.append(1.0)
                    face_pts.append(pt)
                face_arr = np.array(face_pts, dtype=np.float32)
            else:
                face_arr = np.full((468, coord_dim), fill_val, dtype=np.float32)

            # Ghép: Pose (33) + Face (468) + LeftHand (21) + RightHand (21) = 543
            return np.concatenate([pose_arr, face_arr, lh_arr, rh_arr], axis=0)

        elif mode == "hands_pose":
            # Ghép: Pose (33) + LeftHand (21) + RightHand (21) = 75
            return np.concatenate([pose_arr, lh_arr, rh_arr], axis=0)

        elif mode == "hands_pose_lips":
            if results.face_landmarks:
                lms = results.face_landmarks.landmark
                lips_pts = []
                for idx in UNIQUE_LIPS_INDICES:
                    lm = lms[idx]
                    pt = [lm.x, lm.y]
                    if include_z:
                        pt.append(lm.z)
                    if include_vis:
                        pt.append(1.0)
                    lips_pts.append(pt)
                lips_arr = np.array(lips_pts, dtype=np.float32)
            else:
                lips_arr = np.full((len(UNIQUE_LIPS_INDICES), coord_dim), fill_val, dtype=np.float32)

            # Ghép: Pose (33) + LeftHand (21) + RightHand (21) + Lips (40) = 115
            return np.concatenate([pose_arr, lh_arr, rh_arr, lips_arr], axis=0)

        else:
            raise ValueError(f"keypoint_mode không hợp lệ: {mode}")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Xử lý 1 khung hình BGR duy nhất.
        Trả về:
            landmarks_array: Mảng numpy (num_keypoints, coord_dim)
            results: Đối tượng kết quả gốc của MediaPipe (dùng cho vẽ hoặc debug)
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        landmarks = self._extract_landmarks_from_result(results)
        return landmarks, results

    def normalize_landmarks(self, landmarks_seq: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa chuỗi landmarks theo khung tọa độ chuẩn:
        - Tịnh tiến (Centering): Đưa trung điểm của 2 vai (landmark 11 và 12 của Pose) về gốc (0, 0, 0).
        - Co giãn (Scaling): Chia cho khoảng cách Euclidean giữa 2 vai để bất biến khoảng cách máy quay.
        
        Input shape: (num_frames, num_keypoints, coord_dim)
        """
        if not self.config.normalize or landmarks_seq.shape[0] == 0:
            return landmarks_seq

        normed = landmarks_seq.copy()
        # Trong mảng landmarks, 33 điểm đầu tiên luôn là Pose landmarks:
        # Landmark 11: left_shoulder, Landmark 12: right_shoulder
        left_shoulders = normed[:, 11, :2]
        right_shoulders = normed[:, 12, :2]

        # Kiểm tra tính hợp lệ của vai
        valid_mask = (np.abs(left_shoulders).sum(axis=-1) > 0) & (np.abs(right_shoulders).sum(axis=-1) > 0)

        if np.any(valid_mask):
            # Tính trung điểm vai trung bình qua các frame hợp lệ
            mid_shoulders = (left_shoulders[valid_mask] + right_shoulders[valid_mask]) / 2.0
            center = np.mean(mid_shoulders, axis=0)  # (2,)

            # Tính khoảng cách 2 vai
            dists = np.linalg.norm(left_shoulders[valid_mask] - right_shoulders[valid_mask], axis=1)
            scale = np.mean(dists)
            if scale < 1e-4:
                scale = 1.0
        else:
            center = np.array([0.5, 0.5], dtype=np.float32)
            scale = 1.0

        # Áp dụng chuẩn hóa cho x và y
        coord_dim = self.config.coord_dim
        dim_to_norm = min(2, coord_dim)

        # Trừ tâm và chia tỉ lệ (chỉ trừ vào các điểm khác 0/hợp lệ)
        nonzero_mask = np.abs(normed[:, :, :dim_to_norm]).sum(axis=-1) > 0
        normed[:, :, :dim_to_norm][nonzero_mask] -= center
        normed[:, :, :dim_to_norm][nonzero_mask] /= scale

        # Nếu có tọa độ z (cột 2), chia tỉ lệ cùng scale
        if coord_dim >= 3 and self.config.include_z:
            z_mask = np.abs(normed[:, :, 2]) > 0
            normed[:, :, 2][z_mask] /= scale

        return normed

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """
        Trích xuất landmarks từ toàn bộ video.
        
        Args:
            video_path: Đường dẫn tới file video (.mp4).
            max_frames: Số lượng frame tối đa cần lấy (mặc định lấy theo config).
            stride: Bước nhảy khung hình (mặc định lấy theo config).

        Returns:
            np.ndarray có shape: (T, num_keypoints, coord_dim)
        """
        v_path = Path(video_path)
        if not v_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video tại: {v_path.resolve()}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Không thể mở video bằng OpenCV: {v_path.resolve()}")

        stride = stride or self.config.frame_stride or 1
        max_f = max_frames if max_frames is not None else self.config.max_frames

        frames_landmarks = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                lms, _ = self.process_frame(frame)
                frames_landmarks.append(lms)

                if max_f > 0 and len(frames_landmarks) >= max_f:
                    break

            frame_idx += 1

        cap.release()

        if len(frames_landmarks) == 0:
            # Video rỗng hoặc không đọc được frame nào
            return np.empty((0, self.config.num_keypoints, self.config.coord_dim), dtype=np.float32)

        seq = np.array(frames_landmarks, dtype=np.float32)

        # Áp dụng chuẩn hóa bất biến vị trí và tỉ lệ nếu bật cấu hình
        if self.config.normalize:
            seq = self.normalize_landmarks(seq)

        # Áp dụng thuật toán nội suy và zero-padding cho frame mất dấu bàn tay
        if getattr(self.config, "interpolate_hands", False) and seq.shape[0] > 1:
            seq, _ = interpolate_holistic_landmarks(
                seq,
                keypoint_mode=self.config.keypoint_mode,
                max_gap_size=getattr(self.config, "max_gap_size", None),
                boundary_mode=getattr(self.config, "boundary_mode", "zeros"),
            )

        return seq

    def draw_landmarks(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        """
        Vẽ khung xương và các điểm landmark lên khung hình BGR phục vụ trực quan hóa.
        """
        annotated = frame_bgr.copy()

        # 1. Vẽ Face Mesh
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.face_landmarks,
                connections=self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        # 2. Vẽ Pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.pose_landmarks,
                connections=self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # 3. Vẽ Left Hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.left_hand_landmarks,
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        # 4. Vẽ Right Hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.right_hand_landmarks,
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated
