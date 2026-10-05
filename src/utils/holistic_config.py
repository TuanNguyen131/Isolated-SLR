"""
Lớp cấu hình cho MediaPipe Holistic trong bài toán Isolated Sign Language Recognition (Isolated-SLR).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml


# Chỉ số các điểm môi (lips) từ 468 điểm khuôn mặt của MediaPipe Face Mesh
# Chuẩn 40 điểm viền môi ngoài và môi trong phục vụ nhận dạng khẩu hình cử chỉ
LIPS_LANDMARK_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78,
    191, 80, 81, 82, 13, 312
]
# Khử trùng lặp và giữ thứ tự ổn định
UNIQUE_LIPS_INDICES = sorted(list(set(LIPS_LANDMARK_INDICES)))


@dataclass
class HolisticConfig:
    """
    Cấu hình cho bộ giải pháp MediaPipe Holistic và quy trình trích xuất landmark.
    """
    # MediaPipe Solutions parameters
    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = False
    enable_segmentation: bool = False

    # Keypoint selection & representation
    keypoint_mode: str = "full"  # "full" (543), "hands_pose" (75), "hands_pose_lips" (115)
    include_z: bool = True
    include_visibility: bool = False
    normalize: bool = True
    fill_missing: str = "zeros"  # "zeros" hoặc "nan"
    frame_stride: int = 1
    max_frames: int = 0  # 0 nghĩa là lấy toàn bộ frame

    @property
    def num_keypoints(self) -> int:
        """Tổng số điểm landmark theo keypoint_mode."""
        if self.keypoint_mode == "full":
            # 33 pose + 468 face + 21 left_hand + 21 right_hand
            return 543
        elif self.keypoint_mode == "hands_pose":
            # 33 pose + 21 left_hand + 21 right_hand
            return 75
        elif self.keypoint_mode == "hands_pose_lips":
            # 33 pose + 21 left_hand + 21 right_hand + 40 lips
            return 75 + len(UNIQUE_LIPS_INDICES)
        else:
            raise ValueError(f"Không hỗ trợ keypoint_mode: {self.keypoint_mode}")

    @property
    def coord_dim(self) -> int:
        """Số chiều tọa độ cho mỗi điểm (2, 3 hoặc 4)."""
        if self.include_visibility:
            return 4
        return 3 if self.include_z else 2

    @property
    def feature_dim(self) -> int:
        """Kích thước vector đặc trưng cho mỗi khung hình (num_keypoints * coord_dim)."""
        return self.num_keypoints * self.coord_dim

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển lồng nhau."""
        return {
            "mediapipe": {
                "static_image_mode": self.static_image_mode,
                "model_complexity": self.model_complexity,
                "smooth_landmarks": self.smooth_landmarks,
                "min_detection_confidence": self.min_detection_confidence,
                "min_tracking_confidence": self.min_tracking_confidence,
                "refine_face_landmarks": self.refine_face_landmarks,
                "enable_segmentation": self.enable_segmentation,
            },
            "extraction": {
                "keypoint_mode": self.keypoint_mode,
                "include_z": self.include_z,
                "include_visibility": self.include_visibility,
                "normalize": self.normalize,
                "fill_missing": self.fill_missing,
                "frame_stride": self.frame_stride,
                "max_frames": self.max_frames,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HolisticConfig":
        """Khởi tạo từ dictionary."""
        mp_cfg = data.get("mediapipe", {})
        ext_cfg = data.get("extraction", {})
        merged = {}
        merged.update(mp_cfg)
        merged.update(ext_cfg)
        # Giữ lại các key thuộc dataclass
        field_names = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in merged.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> "HolisticConfig":
        """Đọc cấu hình từ file YAML."""
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file config tại: {yaml_path.resolve()}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save_to_yaml(self, path: Union[str, Path]) -> None:
        """Lưu cấu hình ra file YAML."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
