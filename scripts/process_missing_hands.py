"""
Script tiền xử lý hàng loạt: Nội suy và Zero-Padding cho các frame bị mất dấu bàn tay
trong toàn bộ kho dữ liệu landmarks của dự án Isolated-SLR.

Tính năng:
- Quét toàn bộ file .npy (vector 201 hoặc Holistic 543/75/115).
- Thống kê chi tiết tỷ lệ frame mất dấu bàn tay (LH/RH) trước và sau xử lý.
- Áp dụng thuật toán nội suy tuyến tính (Linear Interpolation) cho các gap nội tại.
- Áp dụng Zero-Padding cho các vùng biên (leading/trailing) và bàn tay không hoạt động.
- Xuất báo cáo tổng kết và lưu dữ liệu sạch ra thư mục đích hoặc cập nhật trực tiếp.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from tqdm import tqdm

from src.data.interpolation import (
    interpolate_vector201,
    interpolate_holistic_landmarks,
    detect_missing_frames,
)
from src.data.normalization import (
    normalize_vector201,
    normalize_holistic_landmarks,
)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    kind: str = "linear",
    max_gap_size: int = 0,
    boundary_mode: str = "zeros",
    normalize: bool = False,
    hand_origin: str = "wrist",
    limit: int = 0,
) -> Dict[str, Any]:
    """
    Xử lý hàng loạt toàn bộ file trong input_dir và lưu ra output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(list(input_dir.glob("*.npy")))
    if limit > 0:
        npy_files = npy_files[:limit]

    print("=" * 75)
    print("   XỬ LÝ MẤT DẤU BÀN TAY BẰNG THUẬT TOÁN NỘI SUY (INTERPOLATION) & ZERO-PADDING")
    print("=" * 75)
    print(f"[*] Thư mục nguồn:       {input_dir.resolve()}")
    print(f"[*] Thư mục đích:        {output_dir.resolve()}")
    print(f"[*] Số lượng file:       {len(npy_files)}")
    print(f"[*] Phương pháp nội suy: {kind}")
    print(f"[*] Giới hạn gap:        {'Không giới hạn' if max_gap_size <= 0 else f'{max_gap_size} frames'}")
    print(f"[*] Chế độ vùng biên:    {boundary_mode} (Zero-Padding)")
    print(f"[*] Chuẩn hóa tọa độ:    {normalize} (hand_origin='{hand_origin}')")
    print("-" * 75)


    gap_limit = max_gap_size if max_gap_size > 0 else None

    # Thống kê tổng hợp
    total_videos = len(npy_files)
    total_frames = 0
    total_lh_missing_before = 0
    total_rh_missing_before = 0
    total_lh_interpolated = 0
    total_rh_interpolated = 0
    total_lh_padded = 0
    total_rh_padded = 0
    videos_with_lh_gap = 0
    videos_with_rh_gap = 0

    start_time = time.time()
    pbar = tqdm(npy_files, desc="Đang nội suy & zero-padding", unit="video")

    for f_path in pbar:
        arr = np.load(f_path)
        out_path = output_dir / f_path.name

        if arr.ndim == 2 and arr.shape[1] == 201:
            # Vector 201
            lh_raw = arr[:, 0:63]
            rh_raw = arr[:, 63:126]
            lh_miss = detect_missing_frames(lh_raw).sum()
            rh_miss = detect_missing_frames(rh_raw).sum()

            processed, stats = interpolate_vector201(
                arr,
                kind=kind,
                max_gap_size=gap_limit,
                boundary_mode=boundary_mode,
            )

            lh_interp = stats["left_hand"]["interpolated_frames"]
            rh_interp = stats["right_hand"]["interpolated_frames"]
            lh_pad = stats["left_hand"]["zero_padded_frames"]
            rh_pad = stats["right_hand"]["zero_padded_frames"]

        elif arr.ndim == 3 and arr.shape[1] in (543, 75, 115):
            # Holistic
            k_mode = "full" if arr.shape[1] == 543 else "hands_pose"
            lh_slice = slice(501, 522) if k_mode == "full" else slice(33, 54)
            rh_slice = slice(522, 543) if k_mode == "full" else slice(54, 75)

            lh_miss = detect_missing_frames(arr[:, lh_slice, :]).sum()
            rh_miss = detect_missing_frames(arr[:, rh_slice, :]).sum()

            processed, stats = interpolate_holistic_landmarks(
                arr,
                keypoint_mode=k_mode,
                kind=kind,
                max_gap_size=gap_limit,
                boundary_mode=boundary_mode,
            )

            lh_interp = stats["left_hand"]["interpolated_frames"]
            rh_interp = stats["right_hand"]["interpolated_frames"]
            lh_pad = stats["left_hand"]["zero_padded_frames"]
            rh_pad = stats["right_hand"]["zero_padded_frames"]
        else:
            # Bỏ qua file không đúng định dạng
            np.save(out_path, arr)
            continue

        if normalize:
            if arr.ndim == 2 and arr.shape[1] == 201:
                processed = normalize_vector201(processed, hand_origin=hand_origin)
            elif arr.ndim == 3 and arr.shape[1] in (543, 75, 115):
                processed = normalize_holistic_landmarks(processed, keypoint_mode=k_mode, hand_origin=hand_origin)

        np.save(out_path, processed.astype(np.float32))

        n_f = arr.shape[0]
        total_frames += n_f
        total_lh_missing_before += lh_miss
        total_rh_missing_before += rh_miss
        total_lh_interpolated += lh_interp
        total_rh_interpolated += rh_interp
        total_lh_padded += lh_pad
        total_rh_padded += rh_pad

        if lh_interp > 0:
            videos_with_lh_gap += 1
        if rh_interp > 0:
            videos_with_rh_gap += 1

    elapsed = time.time() - start_time
    fps = (total_frames / elapsed) if elapsed > 0 else 0

    print("\n" + "=" * 75)
    print("                     BÁO CÁO KẾT QUẢ XỬ LÝ")
    print("=" * 75)
    print(f"  * Tổng số video xử lý:            {total_videos}")
    print(f"  * Tổng số frame dữ liệu:          {total_frames:,}")
    print(f"  * Thời gian thực thi:             {elapsed:.2f}s ({fps:.1f} frames/sec)")
    print("-" * 75)
    print(f"  [TAY TRÁI (LEFT HAND)]:")
    print(f"    - Frame mất dấu ban đầu:        {total_lh_missing_before:,} ({total_lh_missing_before/max(1, total_frames)*100:.1f}%)")
    print(f"    - Frame đã được nội suy mượt:   {total_lh_interpolated:,} (tại {videos_with_lh_gap} clips)")
    print(f"    - Frame zero-padded (vùng biên):{total_lh_padded:,}")
    print(f"  [TAY PHẢI (RIGHT HAND)]:")
    print(f"    - Frame mất dấu ban đầu:        {total_rh_missing_before:,} ({total_rh_missing_before/max(1, total_frames)*100:.1f}%)")
    print(f"    - Frame đã được nội suy mượt:   {total_rh_interpolated:,} (tại {videos_with_rh_gap} clips)")
    print(f"    - Frame zero-padded (vùng biên):{total_rh_padded:,}")
    if normalize:
        print(f"  [CHUẨN HÓA TỌA ĐỘ]:")
        print(f"    - Đã chuẩn hóa dời gốc về '{hand_origin}' và co giãn theo khoảng cách 2 vai cho toàn bộ {total_videos} videos.")
    print("=" * 75 + "\n")

    return {
        "total_videos": total_videos,
        "total_frames": total_frames,
        "total_interpolated": total_lh_interpolated + total_rh_interpolated,
        "total_padded": total_lh_padded + total_rh_padded,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý nội suy và zero-padding cho kho landmarks.")
    parser.add_argument("--input_dir", "-i", type=str, default="data/processed_landmarks", help="Thư mục npy nguồn")
    parser.add_argument("--output_dir", "-o", type=str, default="data/processed_landmarks_interpolated", help="Thư mục npy đích")
    parser.add_argument("--kind", type=str, default="linear", choices=["linear", "nearest"], help="Thuật toán nội suy")
    parser.add_argument("--max_gap_size", type=int, default=0, help="Giới hạn frame mất dấu tối đa để nội suy")
    parser.add_argument("--boundary_mode", type=str, default="zeros", choices=["zeros", "nearest"], help="Chế độ vùng biên (mặc định: zeros)")
    parser.add_argument("--normalize", action="store_true", help="Chuẩn hóa tọa độ (dời gốc về cổ tay/khoảng cách hai vai)")
    parser.add_argument("--hand_origin", type=str, default="wrist", choices=["wrist", "shoulder"], help="Gốc tọa độ bàn tay: 'wrist' hoặc 'shoulder'")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Giới hạn số file xử lý (0 là toàn bộ)")
    args = parser.parse_args()

    process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        kind=args.kind,
        max_gap_size=args.max_gap_size,
        boundary_mode=args.boundary_mode,
        normalize=args.normalize,
        hand_origin=args.hand_origin,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
