"""
Module chuẩn hóa tọa độ (Coordinate Normalization) cho Isolated Sign Language Recognition (Isolated-SLR).

Thuật toán kết hợp chuẩn hóa 2 tầng phân cấp:
1. Tầng toàn cục thân người (Body / Global Pose Normalization):
   - Dời gốc tọa độ (Centering) về trung điểm của hai vai (Midpoint between Left & Right Shoulders).
   - Co giãn (Scaling) theo khoảng cách Euclidean giữa hai vai (Shoulder Distance / Shoulder Width).
   - Giúp vector đặc trưng bất biến trước khoảng cách người đứng xa/gần camera và vị trí lệch tâm khung hình.

2. Tầng cục bộ bàn tay (Handshape / Local Wrist Normalization - Dời gốc về cổ tay):
   - Với mỗi bàn tay (Left Hand, Right Hand: 21 điểm mốc), điểm mốc số 0 chính là Cổ tay (WRIST).
   - Dời gốc tọa độ của tất cả 21 điểm bàn tay về chính điểm Cổ tay của bàn tay đó:
       p_i_rel = p_i - p_wrist,  với i = 0..20 (Khi đó p_wrist = (0, 0, 0)).
   - Co giãn theo khoảng cách hai vai (Shoulder Distance):
       p_i_norm = p_i_rel / shoulder_dist.
   - Ý nghĩa: Tách biệt hoàn toàn hình thái ngón tay (Handshape) khỏi vị trí của bàn tay trong không gian,
     giúp mô hình học cấu trúc cử chỉ ngón tay ổn định vượt trội, không bị phụ thuộc vào vị trí giơ tay.
   - Vị trí toàn cục của cổ tay trong không gian vẫn được bảo toàn trọn vẹn thông qua các điểm mốc
     cổ tay trong thân trên (Upper Pose Landmark 15: Cổ tay trái, Landmark 16: Cổ tay phải).
"""

from typing import Optional, Tuple, Union, Dict, Any
import numpy as np


def compute_shoulder_reference(
    pose: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """
    Tính trung điểm hai vai (center) và khoảng cách hai vai (scale) từ mảng pose của 1 khung hình.

    Trong MediaPipe Pose / Upper Body:
    - Index 11: Vai trái (Left Shoulder)
    - Index 12: Vai phải (Right Shoulder)

    Args:
        pose: Mảng pose shape (25, C) hoặc (33, C) với C >= 2 (x, y, z...).
        eps: Ngưỡng kiểm tra điểm hợp lệ.

    Returns:
        center: Tọa độ trung điểm 2 vai shape (C,), mặc định xét (x, y).
        scale: Khoảng cách Euclidean giữa 2 vai (float).
    """
    coord_dim = pose.shape[-1]
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]

    has_left = np.abs(left_shoulder).sum() > eps
    has_right = np.abs(right_shoulder).sum() > eps

    if has_left and has_right:
        center_xy = (left_shoulder + right_shoulder) / 2.0
        scale = float(np.linalg.norm(left_shoulder - right_shoulder))
        if scale < 1e-4:
            scale = 1.0
    elif has_left or has_right:
        center_xy = left_shoulder if has_left else right_shoulder
        scale = 1.0
    else:
        # Fallback: dùng mũi (index 0) hoặc tâm khung hình (0.5, 0.5)
        nose = pose[0, :2]
        if np.abs(nose).sum() > eps:
            center_xy = nose
        else:
            center_xy = np.array([0.5, 0.5], dtype=np.float32)
        scale = 1.0

    # Ghép thêm các chiều z nếu có (cho center z = 0.0 để giữ nguyên độ sâu tương đối)
    center = np.zeros(coord_dim, dtype=np.float32)
    center[:2] = center_xy

    # Nếu cả 2 vai có tọa độ z hợp lệ, có thể lấy trung điểm z
    if coord_dim >= 3 and has_left and has_right:
        center[2] = (pose[11, 2] + pose[12, 2]) / 2.0

    return center, scale


def compute_sequence_shoulder_reference(
    pose_seq: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """
    Tính toán trung điểm 2 vai và khoảng cách 2 vai ổn định trung bình qua toàn bộ chuỗi video.
    Giúp chuỗi video bất biến tỉ lệ mà không bị rung giật (jitter) giữa các frame liên tiếp.

    Args:
        pose_seq: Mảng pose chuỗi video shape (T, num_pose_landmarks, C).
        eps: Ngưỡng kiểm tra điểm hợp lệ.

    Returns:
        center: Trung điểm 2 vai trung bình qua các frame hợp lệ.
        scale: Khoảng cách 2 vai trung bình qua các frame hợp lệ.
    """
    if pose_seq.ndim != 3 or pose_seq.shape[0] == 0:
        return np.array([0.5, 0.5, 0.0], dtype=np.float32), 1.0

    num_frames = pose_seq.shape[0]
    coord_dim = pose_seq.shape[2]

    left_shoulders = pose_seq[:, 11, :2]
    right_shoulders = pose_seq[:, 12, :2]

    valid_mask = (np.abs(left_shoulders).sum(axis=-1) > eps) & (np.abs(right_shoulders).sum(axis=-1) > eps)

    if np.any(valid_mask):
        mid_shoulders = (left_shoulders[valid_mask] + right_shoulders[valid_mask]) / 2.0
        center_xy = np.mean(mid_shoulders, axis=0)

        dists = np.linalg.norm(left_shoulders[valid_mask] - right_shoulders[valid_mask], axis=1)
        scale = float(np.mean(dists))
        if scale < 1e-4:
            scale = 1.0

        center = np.zeros(coord_dim, dtype=np.float32)
        center[:2] = center_xy
        if coord_dim >= 3:
            z_left = pose_seq[valid_mask, 11, 2]
            z_right = pose_seq[valid_mask, 12, 2]
            center[2] = float(np.mean((z_left + z_right) / 2.0))
        return center, scale
    else:
        # Nếu không có frame nào đủ cả 2 vai, thử lấy từng frame đơn lẻ
        centers = []
        scales = []
        for t in range(num_frames):
            c, s = compute_shoulder_reference(pose_seq[t], eps=eps)
            centers.append(c)
            scales.append(s)
        return np.mean(centers, axis=0), float(np.mean(scales))


def normalize_hand_landmarks(
    hand: np.ndarray,
    scale: float = 1.0,
    hand_origin: str = "wrist",
    store_wrist_origin: bool = False,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Chuẩn hóa tọa độ một bàn tay (21 điểm):
    - Dời gốc về cổ tay (Landmark 0) nếu hand_origin == "wrist".
    - Co giãn theo khoảng cách hai vai (scale).
    - Các frame mất dấu (zero-padded) được giữ nguyên 0.

    Hỗ trợ input 2D: (21, C) cho 1 frame, hoặc 3D: (T, 21, C) cho chuỗi video.

    Args:
        hand: Mảng numpy shape (21, C) hoặc (T, 21, C).
        scale: Hệ số tỉ lệ khoảng cách 2 vai (shoulder distance).
        hand_origin: "wrist" (dời gốc về cổ tay) hoặc "none" (chỉ chia tỉ lệ).
        store_wrist_origin: Nếu True, điểm 0 giữ lại tọa độ cổ tay ban đầu chia scale,
                           chỉ các điểm 1..20 là tương đối so với cổ tay.
                           Nếu False (mặc định), điểm 0 chính xác là (0, 0, 0).
        eps: Ngưỡng kiểm tra điểm rỗng.

    Returns:
        norm_hand: Mảng cùng shape với hand sau khi chuẩn hóa.
    """
    norm_hand = np.array(hand, dtype=np.float32, copy=True)
    if scale < 1e-4:
        scale = 1.0

    if norm_hand.ndim == 2:
        # Một khung hình đơn lẻ: shape (21, C)
        # Kiểm tra xem bàn tay có được phát hiện hợp lệ không
        is_valid = np.abs(norm_hand[:, :2]).sum() > eps
        if not is_valid:
            norm_hand.fill(0.0)
            return norm_hand

        if hand_origin == "wrist":
            wrist_pos = norm_hand[0].copy()
            # Dời toàn bộ 21 điểm về gốc cổ tay
            norm_hand = (norm_hand - wrist_pos) / scale
            if store_wrist_origin:
                norm_hand[0] = wrist_pos / scale
            else:
                norm_hand[0] = 0.0
        else:
            norm_hand = norm_hand / scale

        return norm_hand

    elif norm_hand.ndim == 3:
        # Chuỗi video: shape (T, 21, C)
        num_frames = norm_hand.shape[0]
        for t in range(num_frames):
            frame_hand = norm_hand[t]
            is_valid = np.abs(frame_hand[:, :2]).sum() > eps
            if not is_valid:
                norm_hand[t].fill(0.0)
                continue

            if hand_origin == "wrist":
                wrist_pos = frame_hand[0].copy()
                norm_hand[t] = (frame_hand - wrist_pos) / scale
                if store_wrist_origin:
                    norm_hand[t, 0] = wrist_pos / scale
                else:
                    norm_hand[t, 0] = 0.0
            else:
                norm_hand[t] = frame_hand / scale

        return norm_hand
    else:
        raise ValueError(f"Shape không hợp lệ cho hand landmarks: {hand.shape}")


def normalize_pose_landmarks(
    pose: np.ndarray,
    center: np.ndarray,
    scale: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Chuẩn hóa tọa độ tư thế thân trên (Pose):
    - Dời gốc tọa độ về trung điểm 2 vai (center).
    - Co giãn theo khoảng cách hai vai (scale).

    Hỗ trợ input 2D: (num_pose, C) hoặc 3D: (T, num_pose, C).

    Args:
        pose: Mảng pose shape (num_pose, C) hoặc (T, num_pose, C).
        center: Tọa độ trung điểm 2 vai shape (C,) hoặc (2,).
        scale: Khoảng cách 2 vai.
        eps: Ngưỡng kiểm tra điểm rỗng.

    Returns:
        norm_pose: Mảng pose sau khi chuẩn hóa.
    """
    norm_pose = np.array(pose, dtype=np.float32, copy=True)
    if scale < 1e-4:
        scale = 1.0

    dim_to_norm = min(center.shape[0], norm_pose.shape[-1])

    if norm_pose.ndim == 2:
        # Đơn frame: (num_pose, C)
        nonzero = np.abs(norm_pose[:, :2]).sum(axis=-1) > eps
        norm_pose[nonzero, :dim_to_norm] = (norm_pose[nonzero, :dim_to_norm] - center[:dim_to_norm]) / scale
        if norm_pose.shape[-1] >= 3 and dim_to_norm < norm_pose.shape[-1]:
            z_nonzero = np.abs(norm_pose[:, 2]) > eps
            norm_pose[z_nonzero, 2] = norm_pose[z_nonzero, 2] / scale
        return norm_pose

    elif norm_pose.ndim == 3:
        # Chuỗi video: (T, num_pose, C)
        nonzero = np.abs(norm_pose[:, :, :2]).sum(axis=-1) > eps
        norm_pose[:, :, :dim_to_norm][nonzero] = (norm_pose[:, :, :dim_to_norm][nonzero] - center[:dim_to_norm]) / scale
        if norm_pose.shape[-1] >= 3 and dim_to_norm < norm_pose.shape[-1]:
            z_nonzero = np.abs(norm_pose[:, :, 2]) > eps
            norm_pose[:, :, 2][z_nonzero] = norm_pose[:, :, 2][z_nonzero] / scale
        return norm_pose

    else:
        raise ValueError(f"Shape không hợp lệ cho pose landmarks: {pose.shape}")


def normalize_landmarks_wrist_shoulder(
    lh: np.ndarray,
    rh: np.ndarray,
    pose: np.ndarray,
    sequence_level: bool = True,
    hand_origin: str = "wrist",
    store_wrist_origin: bool = False,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm chuẩn hóa tổng hợp cho 3 thành phần: Tay trái, Tay phải, và Thân trên.

    Quy chuẩn:
    - Pose: Gốc tọa độ tại trung điểm hai vai, co giãn theo khoảng cách hai vai.
    - Left Hand: Gốc tọa độ tại cổ tay trái (Landmark 0), co giãn theo khoảng cách hai vai.
    - Right Hand: Gốc tọa độ tại cổ tay phải (Landmark 0), co giãn theo khoảng cách hai vai.

    Tự động hỗ trợ cả 1 khung hình (Realtime inference) lẫn toàn bộ chuỗi clip (Offline preprocessing).

    Args:
        lh: Tay trái shape (21, C) hoặc (T, 21, C).
        rh: Tay phải shape (21, C) hoặc (T, 21, C).
        pose: Thân trên shape (num_pose, C) hoặc (T, num_pose, C).
        sequence_level: Nếu True và input là chuỗi 3D, dùng trung bình vai qua cả clip.
        hand_origin: "wrist" (dời gốc về cổ tay) hoặc "shoulder" (dời về vai như chuẩn cũ).
        store_wrist_origin: Lưu vị trí cổ tay tại landmark 0 thay vì đặt thành (0, 0, 0).
        eps: Ngưỡng kiểm tra điểm hợp lệ.

    Returns:
        norm_lh, norm_rh, norm_pose: 3 mảng numpy đã được chuẩn hóa.
    """
    is_sequence = (pose.ndim == 3)

    if is_sequence:
        if sequence_level:
            center, scale = compute_sequence_shoulder_reference(pose, eps=eps)
        else:
            # Sẽ tính theo từng frame trong loop
            center, scale = None, None
    else:
        center, scale = compute_shoulder_reference(pose, eps=eps)

    if is_sequence and not sequence_level:
        # Chuẩn hóa từng frame riêng biệt cho video
        num_frames = pose.shape[0]
        norm_lh = np.empty_like(lh)
        norm_rh = np.empty_like(rh)
        norm_pose = np.empty_like(pose)

        for t in range(num_frames):
            c_t, s_t = compute_shoulder_reference(pose[t], eps=eps)
            norm_pose[t] = normalize_pose_landmarks(pose[t], center=c_t, scale=s_t, eps=eps)
            norm_lh[t] = normalize_hand_landmarks(
                lh[t], scale=s_t, hand_origin=hand_origin, store_wrist_origin=store_wrist_origin, eps=eps
            )
            norm_rh[t] = normalize_hand_landmarks(
                rh[t], scale=s_t, hand_origin=hand_origin, store_wrist_origin=store_wrist_origin, eps=eps
            )
        return norm_lh, norm_rh, norm_pose

    # Chuẩn hóa chung với center và scale cố định
    norm_pose = normalize_pose_landmarks(pose, center=center, scale=scale, eps=eps)

    if hand_origin == "wrist":
        norm_lh = normalize_hand_landmarks(
            lh, scale=scale, hand_origin="wrist", store_wrist_origin=store_wrist_origin, eps=eps
        )
        norm_rh = normalize_hand_landmarks(
            rh, scale=scale, hand_origin="wrist", store_wrist_origin=store_wrist_origin, eps=eps
        )
    else:
        # Chuẩn hóa cả bàn tay theo trung điểm vai
        norm_lh = normalize_pose_landmarks(lh, center=center, scale=scale, eps=eps)
        norm_rh = normalize_pose_landmarks(rh, center=center, scale=scale, eps=eps)

    return norm_lh, norm_rh, norm_pose


def normalize_vector201(
    feature_seq: np.ndarray,
    sequence_level: bool = True,
    hand_origin: str = "wrist",
    store_wrist_origin: bool = False,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Chuẩn hóa vector 201 đặc trưng (chuỗi T frames hoặc 1 frame đơn lẻ).

    Cấu trúc vector 201:
    - [0 : 63]   -> Tay trái (Left Hand: 21 x 3)
    - [63 : 126] -> Tay phải (Right Hand: 21 x 3)
    - [126 : 201]-> Thân trên (Upper Pose: 25 x 3)

    Args:
        feature_seq: Mảng shape (T, 201) hoặc (201,).
        sequence_level: Tính trung bình vai qua chuỗi clip.
        hand_origin: "wrist" (dời về cổ tay) hoặc "shoulder" (dời về vai).
        store_wrist_origin: Lưu vị trí cổ tay tại index 0 của bàn tay.
        eps: Ngưỡng kiểm tra điểm hợp lệ.

    Returns:
        norm_seq: Mảng cùng shape (T, 201) hoặc (201,).
    """
    is_1d = (feature_seq.ndim == 1)
    if is_1d:
        seq = feature_seq.reshape(1, 201)
    else:
        seq = feature_seq

    num_frames = seq.shape[0]
    lh = seq[:, 0:63].reshape(num_frames, 21, 3)
    rh = seq[:, 63:126].reshape(num_frames, 21, 3)
    pose = seq[:, 126:201].reshape(num_frames, 25, 3)

    norm_lh, norm_rh, norm_pose = normalize_landmarks_wrist_shoulder(
        lh=lh,
        rh=rh,
        pose=pose,
        sequence_level=sequence_level,
        hand_origin=hand_origin,
        store_wrist_origin=store_wrist_origin,
        eps=eps,
    )

    out_seq = np.concatenate(
        [
            norm_lh.reshape(num_frames, 63),
            norm_rh.reshape(num_frames, 63),
            norm_pose.reshape(num_frames, 75),
        ],
        axis=-1,
    ).astype(np.float32)

    return out_seq[0] if is_1d else out_seq


def normalize_holistic_landmarks(
    landmarks_seq: np.ndarray,
    keypoint_mode: str = "full",
    sequence_level: bool = True,
    hand_origin: str = "wrist",
    store_wrist_origin: bool = False,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Chuẩn hóa mảng landmarks MediaPipe Holistic (T, num_keypoints, C).

    Hỗ trợ keypoint_mode:
    - "full": 543 điểm (Pose: 0..33, Face: 33..501, LH: 501..522, RH: 522..543)
    - "hands_pose": 75 điểm (Pose: 0..33, LH: 33..54, RH: 54..75)
    - "hands_pose_lips": 115 điểm (Pose: 0..33, LH: 33..54, RH: 54..75, Lips: 75..115)

    Args:
        landmarks_seq: Mảng shape (T, K, C) hoặc (K, C).
        keypoint_mode: "full", "hands_pose", hoặc "hands_pose_lips".
        sequence_level: Dùng trung bình vai qua cả video clip.
        hand_origin: "wrist" hoặc "shoulder".
        store_wrist_origin: Lưu tọa độ cổ tay tại landmark 0.
        eps: Ngưỡng kiểm tra điểm hợp lệ.

    Returns:
        norm_seq: Mảng cùng shape (T, K, C) hoặc (K, C).
    """
    is_2d = (landmarks_seq.ndim == 2)
    if is_2d:
        seq = landmarks_seq[np.newaxis, ...]
    else:
        seq = landmarks_seq.copy()

    if keypoint_mode == "full":
        pose_slice = slice(0, 33)
        face_slice = slice(33, 501)
        lh_slice = slice(501, 522)
        rh_slice = slice(522, 543)
    elif keypoint_mode == "hands_pose":
        pose_slice = slice(0, 33)
        lh_slice = slice(33, 54)
        rh_slice = slice(54, 75)
        face_slice = None
    elif keypoint_mode == "hands_pose_lips":
        pose_slice = slice(0, 33)
        lh_slice = slice(33, 54)
        rh_slice = slice(54, 75)
        face_slice = slice(75, 115)  # lips
    else:
        raise ValueError(f"Không hỗ trợ keypoint_mode: {keypoint_mode}")

    # Tính center và scale vai từ Pose
    if sequence_level and seq.shape[0] > 1:
        center, scale = compute_sequence_shoulder_reference(seq[:, pose_slice, :], eps=eps)
    else:
        center, scale = compute_shoulder_reference(seq[0, pose_slice, :], eps=eps)

    # 1. Chuẩn hóa Pose
    seq[:, pose_slice, :] = normalize_pose_landmarks(seq[:, pose_slice, :], center=center, scale=scale, eps=eps)

    # 2. Chuẩn hóa Bàn tay (dời gốc về cổ tay / scale theo vai)
    if hand_origin == "wrist":
        seq[:, lh_slice, :] = normalize_hand_landmarks(
            seq[:, lh_slice, :], scale=scale, hand_origin="wrist", store_wrist_origin=store_wrist_origin, eps=eps
        )
        seq[:, rh_slice, :] = normalize_hand_landmarks(
            seq[:, rh_slice, :], scale=scale, hand_origin="wrist", store_wrist_origin=store_wrist_origin, eps=eps
        )
    else:
        seq[:, lh_slice, :] = normalize_pose_landmarks(seq[:, lh_slice, :], center=center, scale=scale, eps=eps)
        seq[:, rh_slice, :] = normalize_pose_landmarks(seq[:, rh_slice, :], center=center, scale=scale, eps=eps)

    # 3. Chuẩn hóa Face / Lips (theo trung điểm vai)
    if face_slice is not None:
        seq[:, face_slice, :] = normalize_pose_landmarks(seq[:, face_slice, :], center=center, scale=scale, eps=eps)

    return seq[0] if is_2d else seq


class CoordinateNormalizer:
    """
    Lớp điều phối chuẩn hóa tọa độ toàn diện cho dự án Isolated-SLR.
    """

    def __init__(
        self,
        hand_origin: str = "wrist",
        sequence_level: bool = True,
        store_wrist_origin: bool = False,
        eps: float = 1e-5,
    ):
        self.hand_origin = hand_origin
        self.sequence_level = sequence_level
        self.store_wrist_origin = store_wrist_origin
        self.eps = eps

    def normalize(
        self,
        landmarks: np.ndarray,
        keypoint_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Tự động chuẩn hóa theo cấu trúc của mảng đầu vào (Vector 201 hoặc Holistic).
        """
        if landmarks.ndim in (1, 2) and landmarks.shape[-1] == 201:
            return normalize_vector201(
                landmarks,
                sequence_level=self.sequence_level,
                hand_origin=self.hand_origin,
                store_wrist_origin=self.store_wrist_origin,
                eps=self.eps,
            )
        elif landmarks.ndim in (2, 3):
            mode = keypoint_mode or ("full" if landmarks.shape[-2] == 543 else "hands_pose")
            return normalize_holistic_landmarks(
                landmarks,
                keypoint_mode=mode,
                sequence_level=self.sequence_level,
                hand_origin=self.hand_origin,
                store_wrist_origin=self.store_wrist_origin,
                eps=self.eps,
            )
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: {landmarks.shape}")
