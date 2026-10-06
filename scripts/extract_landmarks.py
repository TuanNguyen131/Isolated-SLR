"""
Script trích xuất toàn bộ landmark từ video tập dữ liệu WLASL-100 sử dụng MediaPipe Holistic.
Lưu các chuỗi vector đặc trưng ra định dạng numpy .npy tại data/landmarks/{video_id}.npy
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Thêm thư mục gốc của dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo UTF-8 hoạt động chuẩn trên Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from tqdm import tqdm


from src.utils.holistic_config import HolisticConfig
from src.utils.mediapipe_holistic import MediaPipeHolisticExtractor



def parse_args():
    parser = argparse.ArgumentParser(
        description="Trích xuất landmarks bằng MediaPipe Holistic cho tập dữ liệu WLASL-100."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/holistic_config.yaml",
        help="Đường dẫn file cấu hình MediaPipe (mặc định: configs/holistic_config.yaml)",
    )
    parser.add_argument(
        "--videos_dir",
        "-v",
        type=str,
        default="data/raw_videos",
        help="Thư mục chứa video (.mp4) (mặc định: data/raw_videos)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/landmarks",
        help="Thư mục lưu trữ các file landmark .npy (mặc định: data/landmarks)",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default="data/annotations/wlasl_100/splits_available.json",
        help="File annotation splits (mặc định: data/annotations/wlasl_100/splits_available.json)",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Chọn split cần trích xuất ('all', 'train', 'val', 'test')",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Giới hạn số lượng video cần trích xuất (0 là trích xuất toàn bộ)",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=0,
        help="Số worker đa tiến trình (0: tự động chạy đa tiến trình song song, 1: đơn tiến trình)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="both",
        choices=["both", "npy", "hdf5", "h5"],
        help="Định dạng lưu trữ ('both', 'npy', 'hdf5')",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Bỏ qua các file .npy đã tồn tại và hợp lệ (mặc định: True)",
    )
    parser.add_argument(
        "--save_preview",
        action="store_true",
        default=True,
        help="Lưu ảnh demo vẽ khung xương landmark cho video đầu tiên (mặc định: True)",
    )
    return parser.parse_args()


def load_target_videos(annotations_path: Path, videos_dir: Path, split_filter: str = "all") -> List[Dict[str, Any]]:
    """
    Tải danh sách các video cần xử lý từ file annotation và thư mục video.
    """
    targets = []

    if annotations_path.is_file():
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and any(k in data for k in ["train", "val", "test"]):
            splits_to_check = ["train", "val", "test"] if split_filter == "all" else [split_filter]
            for sp in splits_to_check:
                for item in data.get(sp, []):
                    vid_id = item.get("video_id")
                    v_file = videos_dir / f"{vid_id}.mp4"
                    if v_file.is_file():
                        targets.append({
                            "video_id": vid_id,
                            "gloss": item.get("gloss", ""),
                            "split": sp,
                            "video_path": v_file,
                        })
        elif isinstance(data, list):
            for item in data:
                vid_id = item.get("video_id")
                sp = item.get("split", "unknown")
                if split_filter == "all" or sp == split_filter:
                    v_file = videos_dir / f"{vid_id}.mp4"
                    if v_file.is_file():
                        targets.append({
                            "video_id": vid_id,
                            "gloss": item.get("gloss", ""),
                            "split": sp,
                            "video_path": v_file,
                        })

    # Nếu không tìm thấy qua annotation, quét trực tiếp thư mục video
    if not targets:
        for v_file in videos_dir.glob("*.mp4"):
            targets.append({
                "video_id": v_file.stem,
                "gloss": "unknown",
                "split": "unknown",
                "video_path": v_file,
            })

    return targets


def main():
    args = parse_args()

    # Nếu num_workers != 1 (mặc định là 0 = tự động đa tiến trình), chuyển tiếp sang extract_multiprocess
    if args.num_workers != 1:
        from scripts.extract_multiprocess import main as mp_main
        mp_main()
        return

    config_path = Path(args.config)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("       MEDIAPIPE HOLISTIC LANDMARK EXTRACTION PIPELINE")
    print("=" * 70)

    # Đọc cấu hình
    if config_path.is_file():
        config = HolisticConfig.load_from_yaml(config_path)
        print(f"[*] Cấu hình nạp từ: {config_path.resolve()}")
    else:
        config = HolisticConfig()
        print("[!] Không tìm thấy file config, sử dụng cấu hình mặc định.")

    print(f"[*] Chế độ landmark:   {config.keypoint_mode} ({config.num_keypoints} điểm, feature_dim={config.feature_dim})")
    print(f"[*] Chuẩn hóa tọa độ:  {config.normalize}")
    print(f"[*] Thư mục video:     {videos_dir.resolve()}")
    print(f"[*] Thư mục lưu .npy:  {out_dir.resolve()}")

    ann_path = Path(args.annotations)
    targets = load_target_videos(ann_path, videos_dir, split_filter=args.split)
    print(f"[*] Tổng số video khả dụng: {len(targets)}")

    if args.limit > 0:
        targets = targets[:args.limit]
        print(f"[*] Giới hạn xử lý: {len(targets)} video")
    print("-" * 70)

    # Khởi tạo extractor
    start_time = time.time()
    success_count = 0
    skipped_count = 0
    error_count = 0
    total_frames_extracted = 0

    preview_saved = False

    with MediaPipeHolisticExtractor(config) as extractor:
        pbar = tqdm(targets, desc="Đang trích xuất landmarks", unit="video")

        for item in pbar:
            vid_id = item["video_id"]
            v_path = item["video_path"]
            npy_path = out_dir / f"{vid_id}.npy"

            # Bỏ qua nếu đã tồn tại và hợp lệ
            if args.skip_existing and npy_path.is_file():
                try:
                    arr = np.load(npy_path)
                    if arr.ndim == 3 and arr.shape[1] == config.num_keypoints:
                        skipped_count += 1
                        continue
                except Exception:
                    pass

            try:
                # 1. Trích xuất landmarks chuỗi video
                landmarks_seq = extractor.extract_from_video(v_path)
                
                if landmarks_seq.shape[0] == 0:
                    error_count += 1
                    continue

                # 2. Lưu file .npy
                np.save(npy_path, landmarks_seq)
                success_count += 1
                total_frames_extracted += landmarks_seq.shape[0]

                # 3. Lưu ảnh demo vẽ khung xương nếu được yêu cầu
                if args.save_preview and not preview_saved:
                    cap = cv2.VideoCapture(str(v_path))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        _, results = extractor.process_frame(frame)
                        annotated = extractor.draw_landmarks(frame, results)
                        preview_path = out_dir / "sample_preview.jpg"
                        cv2.imwrite(str(preview_path), annotated)
                        preview_saved = True

            except Exception as e:
                error_count += 1

            pbar.set_postfix({
                "Done": success_count,
                "Skip": skipped_count,
                "Err": error_count
            })

    elapsed = time.time() - start_time
    fps_throughput = (total_frames_extracted / elapsed) if elapsed > 0 else 0

    print("\n" + "=" * 70)
    print("                    HOAN TAT TRICH XUAT")
    print("=" * 70)
    print(f"  * Tong thoi gian:           {elapsed:.1f}s")
    print(f"  * So video trich xuat moi:  {success_count}")
    print(f"  * So video da co (Bo qua):  {skipped_count}")
    print(f"  * So video loi:             {error_count}")
    print(f"  * Tong so khung hinh xu ly: {total_frames_extracted}")
    print(f"  * Toc do xu ly trung binh:  {fps_throughput:.1f} frames/sec")
    print(f"  * Dinh dang file:           .npy shape (T, {config.num_keypoints}, {config.coord_dim})")
    print(f"  * Thu muc landmarks:        {out_dir.resolve()}")
    if preview_saved:
        print(f"  * Anh xem truoc khung xuong: {out_dir / 'sample_preview.jpg'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
