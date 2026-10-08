"""
Module trích xuất và chuẩn hóa vector 201 đặc trưng (Feature Extractor)
sử dụng MediaPipe Holistic cho bài toán Isolated Sign Language Recognition (Isolated-SLR).

Cấu trúc vector 201 đặc trưng cho mỗi khung hình (Frame):
- 21 điểm tay trái (Left Hand):   21 x 3 (x, y, z) = 63 giá trị (chỉ số 0 -> 62)
- 21 điểm tay phải (Right Hand):  21 x 3 (x, y, z) = 63 giá trị (chỉ số 63 -> 125)
- 25 điểm tư thế thân trên (Pose): 25 x 3 (x, y, z) = 75 giá trị (chỉ số 126 -> 200)
Tổng cộng: 63 + 63 + 75 = 201 đặc trưng / frame.

Module được thiết kế dùng chung cho cả:
1. Khâu tiền xử lý ngoại tuyến (Offline batch preprocessing) -> lưu .npy
2. Khâu suy luận thời gian thực (Real-time inference loop từ Webcam)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union, Any, List, Dict

import cv2
import numpy as np
import mediapipe as mp

from .interpolation import interpolate_vector201
from .normalization import normalize_landmarks_wrist_shoulder

# Đảm bảo UTF-8 hoạt động chuẩn trên Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 25 chỉ số điểm thân trên của MediaPipe Pose (bỏ các điểm chân từ 25 -> 32)
# 0: Mũi, 1-10: Vùng mắt/tai/miệng, 11-12: Vai, 13-14: Khuỷu tay, 15-16: Cổ tay, 17-22: Bàn tay phụ, 23-24: Hông
POSE_UPPER_BODY_INDICES = list(range(25))

NUM_POSE_LANDMARKS = 25
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_HAND_LANDMARKS + NUM_HAND_LANDMARKS + NUM_POSE_LANDMARKS  # 21 + 21 + 25 = 67
COORD_DIM = 3
FEATURE_DIM = TOTAL_LANDMARKS * COORD_DIM  # 67 * 3 = 201

# Slices phân đoạn tiện lợi cho downstream models
LEFT_HAND_SLICE = slice(0, 63)
RIGHT_HAND_SLICE = slice(63, 126)
POSE_SLICE = slice(126, 201)

# Các đường nối khung xương phần thân trên (Upper-body Pose Connections)
UPPER_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # Mặt & tai
    (9, 10),  # Miệng
    (11, 12),  # Hai vai
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Cánh tay & bàn tay trái
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Cánh tay & bàn tay phải
    (11, 23), (12, 24), (23, 24),  # Thân & hai bên hông
]


class SignFeatureExtractor:
    """
    Bộ trích xuất và chuẩn hóa vector 201 đặc trưng từ ảnh/video sử dụng MediaPipe Holistic.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        normalize: bool = True,
        interpolate_hands: bool = True,
        max_gap_size: Optional[int] = None,
        boundary_mode: str = "zeros",
        hand_origin: str = "wrist",
        store_wrist_origin: bool = False,
    ):
        """
        Khởi tạo MediaPipe Holistic.
        
        Args:
            static_image_mode: False cho video/webcam (bật optical tracking), True cho ảnh rời rạc.
            model_complexity: 0 (nhẹ), 1 (cân bằng), 2 (chính xác cao nhất).
            smooth_landmarks: Làm mượt quỹ đạo chuyển động giữa các frame.
            min_detection_confidence: Ngưỡng tin cậy phát hiện ban đầu.
            min_tracking_confidence: Ngưỡng tin cậy bám vết khung hình.
            normalize: Bật chuẩn hóa vị trí (trung điểm vai) và tỉ lệ (khoảng cách vai).
            interpolate_hands: Bật thuật toán nội suy và zero-padding cho frame mất dấu bàn tay.
            max_gap_size: Số frame mất dấu liên tiếp tối đa được nội suy (None là toàn bộ gap nội tại).
            boundary_mode: Cách xử lý vùng biên đầu/cuối video ("zeros" hoặc "nearest").
            hand_origin: "wrist" (dời gốc về cổ tay) hoặc "shoulder" (dời gốc về trung điểm 2 vai).
            store_wrist_origin: Lưu vị trí cổ tay tại landmark 0 thay vì đặt thành (0, 0, 0).
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.normalize = normalize
        self.interpolate_hands = interpolate_hands
        self.max_gap_size = max_gap_size
        self.boundary_mode = boundary_mode
        self.hand_origin = hand_origin
        self.store_wrist_origin = store_wrist_origin

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            refine_face_landmarks=False,
            enable_segmentation=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        if hasattr(self, "holistic") and self.holistic:
            self.holistic.close()

    def _parse_raw_landmarks(self, results: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tách raw landmarks thành 3 mảng numpy riêng biệt:
        - left_hand: (21, 3)
        - right_hand: (21, 3)
        - pose: (25, 3)
        """
        # 1. Tay trái (21 điểm)
        lh = np.zeros((NUM_HAND_LANDMARKS, COORD_DIM), dtype=np.float32)
        if results.left_hand_landmarks:
            for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                lh[idx] = [lm.x, lm.y, lm.z]

        # 2. Tay phải (21 điểm)
        rh = np.zeros((NUM_HAND_LANDMARKS, COORD_DIM), dtype=np.float32)
        if results.right_hand_landmarks:
            for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                rh[idx] = [lm.x, lm.y, lm.z]

        # 3. Thân trên (25 điểm)
        pose = np.zeros((NUM_POSE_LANDMARKS, COORD_DIM), dtype=np.float32)
        if results.pose_landmarks:
            for i, p_idx in enumerate(POSE_UPPER_BODY_INDICES):
                lm = results.pose_landmarks.landmark[p_idx]
                pose[i] = [lm.x, lm.y, lm.z]

        return lh, rh, pose

    def normalize_landmarks(
        self, lh: np.ndarray, rh: np.ndarray, pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn hóa tọa độ theo thuật toán chuẩn:
        - Thân trên (Pose): Gốc tọa độ tại trung điểm hai vai, co giãn theo khoảng cách hai vai.
        - Bàn tay (Hands): Nếu hand_origin == "wrist", dời gốc tọa độ của 21 điểm về cổ tay (Landmark 0)
          và co giãn theo khoảng cách hai vai. Nếu "shoulder", dời gốc về trung điểm hai vai.
        """
        return normalize_landmarks_wrist_shoulder(
            lh=lh,
            rh=rh,
            pose=pose,
            sequence_level=False,
            hand_origin=self.hand_origin,
            store_wrist_origin=self.store_wrist_origin,
        )


    def extract_frame_with_results(
        self, frame: np.ndarray, is_bgr: bool = True
    ) -> Tuple[np.ndarray, Any]:
        """
        Trích xuất vector 201 đặc trưng từ 1 khung hình cùng với đối tượng MediaPipe results.
        Hàm lý tưởng cho suy luận thời gian thực (Real-time inference) để vừa lấy vector vừa vẽ khung xương.

        Returns:
            feature_vector: np.ndarray shape (201,), dtype float32
            results: MediaPipe Holistic results object
        """
        if is_bgr:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = frame

        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)

        lh, rh, pose = self._parse_raw_landmarks(results)

        if self.normalize:
            lh, rh, pose = self.normalize_landmarks(lh, rh, pose)

        # Ghép thành vector 1D: 63 (LH) + 63 (RH) + 75 (Pose) = 201
        feature_vector = np.concatenate([lh.flatten(), rh.flatten(), pose.flatten()]).astype(np.float32)
        return feature_vector, results

    def extract_frame(self, frame: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """
        Trích xuất trực tiếp vector 201 đặc trưng từ 1 khung hình.
        
        Returns:
            np.ndarray shape (201,), dtype float32
        """
        feature_vector, _ = self.extract_frame_with_results(frame, is_bgr=is_bgr)
        return feature_vector

    def extract_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        stride: int = 1,
    ) -> np.ndarray:
        """
        Trích xuất chuỗi vector 201 đặc trưng từ toàn bộ video clip ngoại tuyến.

        Args:
            video_path: Đường dẫn tới file video (.mp4).
            max_frames: Số khung hình tối đa cần trích xuất (None là lấy hết).
            stride: Bước nhảy khung hình (1 = lấy mọi frame).

        Returns:
            np.ndarray shape (T, 201), dtype float32
        """
        v_path = Path(video_path)
        if not v_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {v_path.resolve()}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Không thể mở video bằng OpenCV: {v_path.resolve()}")

        sequence = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                vector = self.extract_frame(frame, is_bgr=True)
                sequence.append(vector)

                if max_frames is not None and max_frames > 0 and len(sequence) >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        if len(sequence) == 0:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)

        seq_arr = np.array(sequence, dtype=np.float32)

        # Áp dụng thuật toán nội suy và zero-padding cho frame mất dấu bàn tay
        if self.interpolate_hands and seq_arr.shape[0] > 1:
            seq_arr, _ = interpolate_vector201(
                seq_arr,
                max_gap_size=self.max_gap_size,
                boundary_mode=self.boundary_mode,
            )

        return seq_arr

    def draw_landmarks(self, frame_bgr: np.ndarray, results: Any) -> np.ndarray:
        """
        Vẽ trực quan 25 điểm thân trên và 2 bàn tay lên khung hình BGR (phục vụ Webcam / Video demo).
        """
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        # 1. Vẽ Pose thân trên (25 điểm) và các đường nối
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Vẽ các đường kết nối thân trên
            for p1_idx, p2_idx in UPPER_POSE_CONNECTIONS:
                if p1_idx < len(landmarks) and p2_idx < len(landmarks):
                    lm1 = landmarks[p1_idx]
                    lm2 = landmarks[p2_idx]
                    if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                        pt1 = (int(lm1.x * w), int(lm1.y * h))
                        pt2 = (int(lm2.x * w), int(lm2.y * h))
                        cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

            # Vẽ các điểm khớp thân trên
            for p_idx in POSE_UPPER_BODY_INDICES:
                if p_idx < len(landmarks):
                    lm = landmarks[p_idx]
                    if lm.visibility > 0.3:
                        pt = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(annotated, pt, 4, (0, 0, 255), -1)

        # 2. Vẽ bàn tay trái
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        # 3. Vẽ bàn tay phải
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated

    @staticmethod
    def save_landmarks(landmarks_seq: np.ndarray, output_path: Union[str, Path]) -> Path:
        """Lưu chuỗi đặc trưng ra file nhị phân .npy."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_p, landmarks_seq.astype(np.float32))
        return out_p

    @staticmethod
    def load_landmarks(file_path: Union[str, Path]) -> np.ndarray:
        """Đọc chuỗi đặc trưng từ file .npy."""
        return np.load(file_path).astype(np.float32)


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    limit: int = 0,
    skip_existing: bool = True,
    save_preview: bool = True,
    interpolate_hands: bool = True,
    max_gap_size: Optional[int] = None,
    boundary_mode: str = "zeros",
    hand_origin: str = "wrist",
    store_wrist_origin: bool = False,
) -> Dict[str, Any]:
    """
    Xử lý tiền xử lý hàng loạt: đọc toàn bộ video từ input_dir và lưu file .npy vào output_dir.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(in_dir.glob("*.mp4")))
    total_videos = len(video_files)
    if limit > 0:
        video_files = video_files[:limit]

    print("=" * 70)
    print("      TIEN XU LY TRICH XUAT VECTOR 201 DAC TRUNG (ISOLATED-SLR)")
    print("=" * 70)
    print(f"[*] Thu muc video goc:      {in_dir.resolve()}")
    print(f"[*] Thu muc processed npy:  {out_dir.resolve()}")
    print(f"[*] Tong so video tim thay: {total_videos} (Xu ly: {len(video_files)})")
    print(f"[*] Kich thuoc vector:      {FEATURE_DIM} (21 LH x 3 + 21 RH x 3 + 25 Pose x 3)")
    print(f"[*] Chuan hoa toa do:       hand_origin='{hand_origin}' (khoang cach 2 vai)")
    print(f"[*] Noi suy mat dau ban tay:{interpolate_hands} (boundary_mode='{boundary_mode}')")
    print("-" * 70)

    success_cnt = 0
    skipped_cnt = 0
    error_cnt = 0
    total_frames = 0
    preview_saved = False
    start_time = time.time()

    with SignFeatureExtractor(
        static_image_mode=False,
        model_complexity=1,
        normalize=True,
        interpolate_hands=interpolate_hands,
        max_gap_size=max_gap_size,
        boundary_mode=boundary_mode,
        hand_origin=hand_origin,
        store_wrist_origin=store_wrist_origin,
    ) as extractor:
        for idx, v_file in enumerate(video_files, 1):
            vid_id = v_file.stem
            out_file = out_dir / f"{vid_id}.npy"

            # Kiểm tra tồn tại nếu skip_existing=True
            if skip_existing and out_file.is_file():
                try:
                    arr = np.load(out_file)
                    if arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
                        skipped_cnt += 1
                        continue
                except Exception:
                    pass

            try:
                seq = extractor.extract_video(v_file)
                if seq.shape[0] == 0:
                    error_cnt += 1
                    continue

                SignFeatureExtractor.save_landmarks(seq, out_file)
                success_cnt += 1
                total_frames += seq.shape[0]

                # Lưu ảnh minh họa khung xương đầu tiên
                if save_preview and not preview_saved:
                    cap = cv2.VideoCapture(str(v_file))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        _, results = extractor.extract_frame_with_results(frame)
                        annotated = extractor.draw_landmarks(frame, results)
                        preview_p = out_dir / "sample_201_preview.jpg"
                        cv2.imwrite(str(preview_p), annotated)
                        preview_saved = True

            except Exception:
                error_cnt += 1

            if idx % 10 == 0 or idx == len(video_files):
                elapsed = time.time() - start_time
                print(
                    f"  [{idx}/{len(video_files)}] Trich xuat moi: {success_cnt} | "
                    f"Bo qua: {skipped_cnt} | Loi: {error_cnt} ({elapsed:.1f}s)"
                )

    elapsed = time.time() - start_time
    fps = (total_frames / elapsed) if elapsed > 0 else 0

    print("\n" + "=" * 70)
    print("                    HOAN TAT TIEN XU LY")
    print("=" * 70)
    print(f"  * Thoi gian thuc hien:       {elapsed:.1f}s")
    print(f"  * Video da trich xuat moi:   {success_cnt}")
    print(f"  * Video da co san (bo qua):  {skipped_cnt}")
    print(f"  * Video loi:                 {error_cnt}")
    print(f"  * Tong so frame dac trung:   {total_frames}")
    print(f"  * Toc do xu ly trung binh:   {fps:.1f} frames/sec")
    print(f"  * Thu muc dau ra:            {out_dir.resolve()}")
    print("=" * 70 + "\n")

    return {
        "success": success_cnt,
        "skipped": skipped_cnt,
        "error": error_cnt,
        "total_frames": total_frames,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Trich xuat vector 201 dac trung tu tap video.")
    parser.add_argument("--input_dir", "-i", type=str, default="data/raw_videos", help="Thu muc video mp4")
    parser.add_argument("--output_dir", "-o", type=str, default="data/processed_landmarks", help="Thu muc luu npy")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Gioi han so video (0 la toan bo)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Bo qua file npy da co")
    parser.add_argument("--no_interpolate", action="store_true", help="Tat tinh nang noi suy frame mat dau ban tay")
    parser.add_argument("--max_gap_size", type=int, default=0, help="Gioi han so frame mat dau lien tiep de noi suy (0 la khong gioi han)")
    parser.add_argument("--boundary_mode", type=str, default="zeros", choices=["zeros", "nearest"], help="Che do xu ly bien dau/cuoi (zeros hoac nearest)")
    parser.add_argument("--hand_origin", type=str, default="wrist", choices=["wrist", "shoulder"], help="Chuan hoa ban tay: dời gốc về cổ tay (wrist) hay vai (shoulder)")
    args = parser.parse_args()

    max_gap = args.max_gap_size if args.max_gap_size > 0 else None
    interpolate = not args.no_interpolate

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        skip_existing=args.skip_existing,
        interpolate_hands=interpolate,
        max_gap_size=max_gap,
        boundary_mode=args.boundary_mode,
        hand_origin=args.hand_origin,
    )




if __name__ == "__main__":
    main()
