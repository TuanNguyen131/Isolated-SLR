"""
Module xử lý các frame bị mất dấu bàn tay bằng thuật toán nội suy (Interpolation)
và đệm số 0 (Zero-Padding) cho bài toán nhận dạng ngôn ngữ ký hiệu (Isolated-SLR).

Trong video ngôn ngữ ký hiệu, việc bàn tay bị che khuất (occlusion), chuyển động quá nhanh,
hoặc ra khỏi tầm quan sát của camera thường khiến MediaPipe Holistic bị mất dấu (drop-out).
Hiện tượng này tạo ra các bước nhảy đột ngột về (0, 0, 0), phá vỡ tính liên tục của quỹ đạo
chuyển động và gây nhiễu nghiêm trọng cho các mạng học sâu tuần tự (RNN, LSTM, GRU, Transformer).

Quy trình xử lý chuẩn hóa kết hợp:
1. Nhận diện frame mất dấu (Missing Detection): Xác định chính xác các frame thiếu tọa độ bàn tay.
2. Nội suy thời gian (Temporal Interpolation): Áp dụng nội suy tuyến tính (Linear Interpolation)
   cho các khoảng trống cục bộ (internal gaps) giữa hai thời điểm phát hiện được.
3. Đệm số 0 (Zero-Padding):
   - Giữ nguyên giá trị 0 cho các frame ở đầu (leading frames) khi tay chưa vào vị trí cử chỉ.
   - Giữ nguyên giá trị 0 cho các frame ở cuối (trailing frames) khi tay đã hạ xuống.
   - Giữ nguyên giá trị 0 cho bàn tay hoàn toàn không xuất hiện (ví dụ cử chỉ 1 tay).
   - Đệm số 0 theo chiều thời gian để cố định chiều dài chuỗi (Temporal sequence padding).
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np


def detect_missing_frames(
    landmark_arr: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Xác định các khung hình (frames) bị mất dấu landmark.

    Một frame được coi là mất dấu nếu:
    - Chứa giá trị NaN / Inf.
    - Hoặc toàn bộ tọa độ đều xấp xỉ bằng 0 (chưa phát hiện hoặc bị reset về 0).

    Args:
        landmark_arr: Mảng numpy có shape (T, K, C) hoặc (T, D) hoặc (T,).
        eps: Ngưỡng kiểm tra xấp xỉ 0.

    Returns:
        missing_mask: Mảng boolean shape (T,), True nghĩa là frame bị mất dấu.
    """
    if landmark_arr.ndim == 1:
        # Trường hợp mảng 1D: (T,)
        is_nan = np.isnan(landmark_arr) | np.isinf(landmark_arr)
        is_zero = np.abs(landmark_arr) < eps
        return is_nan | is_zero

    # Tính theo các trục còn lại ngoài trục thời gian (axis 0)
    reduce_axes = tuple(range(1, landmark_arr.ndim))
    is_nan = np.any(np.isnan(landmark_arr) | np.isinf(landmark_arr), axis=reduce_axes)
    is_zero = np.all(np.abs(landmark_arr) < eps, axis=reduce_axes)

    return is_nan | is_zero


def interpolate_sequence_2d(
    seq_2d: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    kind: str = "linear",
    max_gap_size: Optional[int] = None,
    boundary_mode: str = "zeros",
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Nội suy và zero-padding trên mảng 2D (T, D) theo trục thời gian T.

    Quy tắc:
    - Các khoảng trống nội tại (internal gaps) giữa 2 frame hợp lệ:
        + Nếu khoảng trống <= max_gap_size (hoặc max_gap_size is None): Nội suy tuyến tính.
        + Nếu khoảng trống > max_gap_size: Giữ nguyên đệm số 0 (Zero-padding).
    - Vùng biên trước frame hợp lệ đầu tiên (leading frames):
        + boundary_mode == "zeros": Đệm số 0 (Zero-padding).
        + boundary_mode == "nearest": Lặp lại giá trị frame hợp lệ đầu tiên.
    - Vùng biên sau frame hợp lệ cuối cùng (trailing frames):
        + boundary_mode == "zeros": Đệm số 0 (Zero-padding).
        + boundary_mode == "nearest": Lặp lại giá trị frame hợp lệ cuối cùng.
    - Bàn tay hoàn toàn không xuất hiện (0 frame hợp lệ):
        + Giữ nguyên toàn bộ 0 (Zero-padding).

    Args:
        seq_2d: Mảng numpy shape (T, D).
        valid_mask: Mảng boolean shape (T,), True nếu frame hợp lệ. Nếu None, tự động tính.
        kind: Phương pháp nội suy ("linear" hoặc "nearest").
        max_gap_size: Số frame mất dấu liên tiếp tối đa được phép nội suy. None là không giới hạn.
        boundary_mode: Cách xử lý 2 đầu video ("zeros" hoặc "nearest"). Mặc định "zeros".
        eps: Ngưỡng kiểm tra 0.

    Returns:
        interpolated_seq: Mảng numpy shape (T, D) sau khi nội suy và padding.
        stats: Dictionary thống kê số frame ban đầu, số frame nội suy, số frame zero-padded.
    """
    seq = np.array(seq_2d, dtype=np.float32, copy=True)
    num_frames, feat_dim = seq.shape

    if num_frames <= 1:
        return seq, {
            "total_frames": num_frames,
            "valid_frames": num_frames,
            "interpolated_frames": 0,
            "zero_padded_frames": 0,
        }

    if valid_mask is None:
        missing_mask = detect_missing_frames(seq, eps=eps)
        valid_mask = ~missing_mask
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    num_valid = len(valid_indices)

    # Thống kê
    interpolated_cnt = 0
    zero_padded_cnt = 0

    # 1. Trường hợp không có frame nào hợp lệ (Toàn bộ chuỗi mất dấu)
    if num_valid == 0:
        seq.fill(0.0)
        return seq, {
            "total_frames": num_frames,
            "valid_frames": 0,
            "interpolated_frames": 0,
            "zero_padded_frames": num_frames,
        }

    # 2. Trường hợp chỉ có đúng 1 frame hợp lệ
    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])

    if num_valid == 1:
        if boundary_mode == "nearest":
            seq[:] = seq[first_valid]
        else:
            # Zero-padding cho tất cả các frame khác
            val = seq[first_valid].copy()
            seq.fill(0.0)
            seq[first_valid] = val
            zero_padded_cnt = num_frames - 1

        return seq, {
            "total_frames": num_frames,
            "valid_frames": 1,
            "interpolated_frames": 0,
            "zero_padded_frames": zero_padded_cnt,
        }

    # 3. Xử lý vùng biên trước (Leading missing frames: 0 -> first_valid - 1)
    if first_valid > 0:
        if boundary_mode == "nearest":
            seq[:first_valid] = seq[first_valid]
        else:
            seq[:first_valid] = 0.0
            zero_padded_cnt += first_valid

    # 4. Xử lý vùng biên sau (Trailing missing frames: last_valid + 1 -> num_frames - 1)
    if last_valid < num_frames - 1:
        if boundary_mode == "nearest":
            seq[last_valid + 1:] = seq[last_valid]
        else:
            seq[last_valid + 1:] = 0.0
            zero_padded_cnt += (num_frames - 1 - last_valid)

    # 5. Xử lý các khoảng trống nội tại (Internal gaps: first_valid -> last_valid)
    t = first_valid
    while t < last_valid:
        if not valid_mask[t]:
            gap_start = t
            while t <= last_valid and not valid_mask[t]:
                t += 1
            gap_end = t - 1
            gap_len = gap_end - gap_start + 1

            prev_idx = gap_start - 1
            next_idx = gap_end + 1

            # Kiểm tra giới hạn max_gap_size
            if max_gap_size is not None and gap_len > max_gap_size:
                # Quá ngưỡng gap cho phép -> zero-padding
                seq[gap_start : gap_end + 1] = 0.0
                zero_padded_cnt += gap_len
            else:
                # Tiến hành nội suy
                prev_val = seq[prev_idx]
                next_val = seq[next_idx]

                if kind == "linear":
                    steps = gap_len + 1
                    # Vector hóa nội suy cho toàn bộ feat_dim:
                    # alpha chạy từ 1/(steps) đến gap_len/(steps)
                    alphas = np.linspace(0.0, 1.0, steps + 1)[1:-1, np.newaxis]
                    seq[gap_start : gap_end + 1] = (1.0 - alphas) * prev_val + alphas * next_val
                elif kind == "nearest":
                    mid = gap_start + gap_len // 2
                    seq[gap_start : mid] = prev_val
                    seq[mid : gap_end + 1] = next_val
                else:
                    raise ValueError(f"Không hỗ trợ phương pháp nội suy kind='{kind}'")

                interpolated_cnt += gap_len
        else:
            t += 1

    stats = {
        "total_frames": num_frames,
        "valid_frames": num_valid,
        "interpolated_frames": interpolated_cnt,
        "zero_padded_frames": zero_padded_cnt,
    }
    return seq, stats


def interpolate_hand_landmarks(
    hand_landmarks: np.ndarray,
    kind: str = "linear",
    max_gap_size: Optional[int] = None,
    boundary_mode: str = "zeros",
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Nội suy và zero-padding cho một bàn tay (Left Hand hoặc Right Hand).
    Hỗ trợ input 3D (T, 21, 3) hoặc 2D (T, 63).

    Args:
        hand_landmarks: Mảng tọa độ bàn tay shape (T, 21, 3) hoặc (T, K, C) hoặc (T, D).
        kind: Phương pháp nội suy ("linear" hoặc "nearest").
        max_gap_size: Độ dài khoảng mất dấu tối đa được nội suy (None = toàn bộ).
        boundary_mode: Xử lý frame đầu/cuối ("zeros" hoặc "nearest").
        eps: Ngưỡng phát hiện số 0.

    Returns:
        processed_landmarks: Mảng có cùng shape với input sau khi nội suy và padding.
        stats: Thống kê chi tiết quá trình xử lý.
    """
    orig_shape = hand_landmarks.shape
    if hand_landmarks.ndim == 3:
        num_frames = orig_shape[0]
        feat_dim = orig_shape[1] * orig_shape[2]
        reshaped = hand_landmarks.reshape(num_frames, feat_dim)
        result_2d, stats = interpolate_sequence_2d(
            reshaped,
            kind=kind,
            max_gap_size=max_gap_size,
            boundary_mode=boundary_mode,
            eps=eps,
        )
        return result_2d.reshape(orig_shape), stats
    elif hand_landmarks.ndim == 2:
        return interpolate_sequence_2d(
            hand_landmarks,
            kind=kind,
            max_gap_size=max_gap_size,
            boundary_mode=boundary_mode,
            eps=eps,
        )
    else:
        raise ValueError(f"Shape không hợp lệ cho hand landmarks: {orig_shape}")


def interpolate_vector201(
    feature_seq: np.ndarray,
    kind: str = "linear",
    max_gap_size: Optional[int] = None,
    boundary_mode: str = "zeros",
    interpolate_pose: bool = False,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Nội suy và zero-padding cho chuỗi vector 201 đặc trưng (Feature Extractor chuẩn của dự án).

    Cấu trúc vector 201 chiều:
    - [0 : 63]   -> Tay trái (Left Hand): 21 điểm x 3 chiều
    - [63 : 126] -> Tay phải (Right Hand): 21 điểm x 3 chiều
    - [126 : 201]-> Thân trên (Upper Pose): 25 điểm x 3 chiều

    Args:
        feature_seq: Mảng numpy shape (T, 201).
        kind: Phương pháp nội suy ("linear" hoặc "nearest").
        max_gap_size: Độ dài khoảng mất dấu tối đa được nội suy (None = không giới hạn).
        boundary_mode: Cách xử lý frame ở 2 đầu video ("zeros" hoặc "nearest").
        interpolate_pose: Nếu True, nội suy thêm phần pose thân trên nếu có frame khuyết.
        eps: Ngưỡng phát hiện số 0.

    Returns:
        processed_seq: Mảng numpy shape (T, 201) đã nội suy và zero-pad.
        stats: Dictionary tổng hợp thống kê cho cả 2 bàn tay.
    """
    if feature_seq.ndim != 2 or feature_seq.shape[1] != 201:
        raise ValueError(f"Kỳ vọng mảng (T, 201), nhận được: {feature_seq.shape}")

    num_frames = feature_seq.shape[0]
    out_seq = feature_seq.copy()

    # 1. Nội suy tay trái: [0 : 63]
    lh_part = out_seq[:, 0:63]
    interp_lh, lh_stats = interpolate_sequence_2d(
        lh_part,
        kind=kind,
        max_gap_size=max_gap_size,
        boundary_mode=boundary_mode,
        eps=eps,
    )
    out_seq[:, 0:63] = interp_lh

    # 2. Nội suy tay phải: [63 : 126]
    rh_part = out_seq[:, 63:126]
    interp_rh, rh_stats = interpolate_sequence_2d(
        rh_part,
        kind=kind,
        max_gap_size=max_gap_size,
        boundary_mode=boundary_mode,
        eps=eps,
    )
    out_seq[:, 63:126] = interp_rh

    # 3. (Tùy chọn) Nội suy thân trên: [126 : 201]
    pose_stats = {}
    if interpolate_pose:
        pose_part = out_seq[:, 126:201]
        interp_pose, pose_stats = interpolate_sequence_2d(
            pose_part,
            kind=kind,
            max_gap_size=max_gap_size,
            boundary_mode=boundary_mode,
            eps=eps,
        )
        out_seq[:, 126:201] = interp_pose

    combined_stats = {
        "num_frames": num_frames,
        "left_hand": lh_stats,
        "right_hand": rh_stats,
    }
    if interpolate_pose:
        combined_stats["pose"] = pose_stats

    return out_seq, combined_stats


def interpolate_holistic_landmarks(
    landmarks_seq: np.ndarray,
    keypoint_mode: str = "full",
    kind: str = "linear",
    max_gap_size: Optional[int] = None,
    boundary_mode: str = "zeros",
    interpolate_pose: bool = False,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Nội suy và zero-padding cho mảng landmark MediaPipe Holistic dạng (T, num_keypoints, coord_dim).

    Hỗ trợ các keypoint_mode:
    - "full": 543 điểm (Pose: 0..33, Face: 33..501, LH: 501..522, RH: 522..543).
    - "hands_pose": 75 điểm (Pose: 0..33, LH: 33..54, RH: 54..75).
    - "hands_pose_lips": 115 điểm (Pose: 0..33, LH: 33..54, RH: 54..75, Lips: 75..115).

    Args:
        landmarks_seq: Mảng numpy shape (T, num_keypoints, coord_dim).
        keypoint_mode: Chế độ landmark ("full", "hands_pose", "hands_pose_lips").
        kind: Phương pháp nội suy ("linear" hoặc "nearest").
        max_gap_size: Số frame mất dấu liên tiếp tối đa được nội suy.
        boundary_mode: Xử lý 2 đầu video ("zeros" hoặc "nearest").
        interpolate_pose: Có nội suy pose hay không.
        eps: Ngưỡng phát hiện 0.

    Returns:
        processed_landmarks: Mảng cùng shape (T, num_keypoints, coord_dim).
        stats: Thống kê quá trình xử lý.
    """
    if landmarks_seq.ndim != 3:
        raise ValueError(f"Kỳ vọng mảng 3D (T, num_keypoints, coord_dim), nhận được: {landmarks_seq.shape}")

    num_frames, num_kpts, coord_dim = landmarks_seq.shape
    out_seq = landmarks_seq.copy()

    # Xác định vị trí lát cắt (slice) của hai bàn tay
    if keypoint_mode == "full":
        lh_slice = slice(501, 522)
        rh_slice = slice(522, 543)
        pose_slice = slice(0, 33)
    elif keypoint_mode in ("hands_pose", "hands_pose_lips"):
        lh_slice = slice(33, 54)
        rh_slice = slice(54, 75)
        pose_slice = slice(0, 33)
    else:
        raise ValueError(f"Không hỗ trợ keypoint_mode: {keypoint_mode}")

    # 1. Nội suy tay trái
    lh_arr = out_seq[:, lh_slice, :]
    interp_lh, lh_stats = interpolate_hand_landmarks(
        lh_arr,
        kind=kind,
        max_gap_size=max_gap_size,
        boundary_mode=boundary_mode,
        eps=eps,
    )
    out_seq[:, lh_slice, :] = interp_lh

    # 2. Nội suy tay phải
    rh_arr = out_seq[:, rh_slice, :]
    interp_rh, rh_stats = interpolate_hand_landmarks(
        rh_arr,
        kind=kind,
        max_gap_size=max_gap_size,
        boundary_mode=boundary_mode,
        eps=eps,
    )
    out_seq[:, rh_slice, :] = interp_rh

    # 3. (Tùy chọn) Nội suy Pose
    pose_stats = {}
    if interpolate_pose:
        pose_arr = out_seq[:, pose_slice, :]
        interp_pose, pose_stats = interpolate_hand_landmarks(
            pose_arr,
            kind=kind,
            max_gap_size=max_gap_size,
            boundary_mode=boundary_mode,
            eps=eps,
        )
        out_seq[:, pose_slice, :] = interp_pose

    stats = {
        "num_frames": num_frames,
        "left_hand": lh_stats,
        "right_hand": rh_stats,
    }
    if interpolate_pose:
        stats["pose"] = pose_stats

    return out_seq, stats


def pad_or_truncate_sequence(
    sequence: np.ndarray,
    max_seq_len: int = 60,
    padding_mode: str = "zeros",
) -> np.ndarray:
    """
    Chuẩn hóa chiều dài chuỗi thời gian về độ dài cố định max_seq_len.
    Áp dụng đệm số 0 (Zero-Padding) khi thiếu frame hoặc nội suy thời gian đều (uniform resampling).

    Args:
        sequence: Mảng numpy shape (T, ...) với T là số frames hiện tại.
        max_seq_len: Độ dài frame cố định mong muốn (ví dụ: 60).
        padding_mode: Chế độ đệm ("zeros" để pad đuôi bằng mảng 0, "edge" lặp frame cuối).

    Returns:
        padded_sequence: Mảng numpy shape (max_seq_len, ...).
    """
    num_frames = sequence.shape[0]

    if num_frames == max_seq_len:
        return sequence

    if num_frames > max_seq_len:
        # Lấy mẫu đều theo trục thời gian (uniform sampling)
        indices = np.linspace(0, num_frames - 1, max_seq_len, dtype=int)
        return sequence[indices]

    # num_frames < max_seq_len: cần đệm thêm (padding)
    pad_len = max_seq_len - num_frames
    pad_shape = (pad_len, *sequence.shape[1:])

    if padding_mode == "zeros":
        padding = np.zeros(pad_shape, dtype=sequence.dtype)
    elif padding_mode == "edge":
        padding = np.repeat(sequence[-1:], pad_len, axis=0)
    else:
        raise ValueError(f"Không hỗ trợ padding_mode: {padding_mode}")

    return np.concatenate([sequence, padding], axis=0)


class HandInterpolationPipeline:
    """
    Lớp pipeline đóng gói toàn bộ quy trình tiền xử lý:
    Nội suy frame mất dấu (Interpolation) -> Đệm số 0 vùng biên & bàn tay vắng mặt (Zero-Padding) ->
    Cố định chiều dài chuỗi (Temporal sequence padding).
    """

    def __init__(
        self,
        kind: str = "linear",
        max_gap_size: Optional[int] = None,
        boundary_mode: str = "zeros",
        max_seq_len: Optional[int] = 60,
        interpolate_pose: bool = False,
        eps: float = 1e-5,
    ):
        self.kind = kind
        self.max_gap_size = max_gap_size
        self.boundary_mode = boundary_mode
        self.max_seq_len = max_seq_len
        self.interpolate_pose = interpolate_pose
        self.eps = eps

    def process(
        self,
        landmarks: np.ndarray,
        keypoint_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Xử lý tự động theo định dạng dữ liệu đầu vào.

        Args:
            landmarks: Mảng numpy dạng vector 201 (T, 201) hoặc Holistic (T, K, C).
            keypoint_mode: Chế độ landmark nếu đầu vào là 3D ("full", "hands_pose", ...).

        Returns:
            processed_seq: Mảng numpy đã xử lý nội suy và zero-padding.
            stats: Thống kê quá trình xử lý.
        """
        if landmarks.ndim == 2 and landmarks.shape[1] == 201:
            processed, stats = interpolate_vector201(
                landmarks,
                kind=self.kind,
                max_gap_size=self.max_gap_size,
                boundary_mode=self.boundary_mode,
                interpolate_pose=self.interpolate_pose,
                eps=self.eps,
            )
        elif landmarks.ndim == 3:
            mode = keypoint_mode or ("full" if landmarks.shape[1] == 543 else "hands_pose")
            processed, stats = interpolate_holistic_landmarks(
                landmarks,
                keypoint_mode=mode,
                kind=self.kind,
                max_gap_size=self.max_gap_size,
                boundary_mode=self.boundary_mode,
                interpolate_pose=self.interpolate_pose,
                eps=self.eps,
            )
        else:
            # Xử lý tổng quát cho mảng (T, D)
            processed, stats = interpolate_sequence_2d(
                landmarks,
                kind=self.kind,
                max_gap_size=self.max_gap_size,
                boundary_mode=self.boundary_mode,
                eps=self.eps,
            )

        if self.max_seq_len is not None and self.max_seq_len > 0:
            processed = pad_or_truncate_sequence(
                processed,
                max_seq_len=self.max_seq_len,
                padding_mode="zeros",
            )
            stats["temporal_padded_len"] = self.max_seq_len

        return processed, stats
