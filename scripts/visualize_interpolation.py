"""
Script trực quan hóa (Visualization) thuật toán Nội suy (Interpolation)
và Zero-Padding cho các frame bị mất dấu bàn tay trong Isolated-SLR.

Vẽ biểu đồ so sánh quỹ đạo chuyển động trước và sau khi xử lý:
1. Trước xử lý: Các điểm sụt giảm đột ngột về (0, 0, 0) do mất dấu bàn tay.
2. Sau xử lý: Quỹ đạo nội suy liên tục, mượt mà giữa các frame hợp lệ.
3. Vùng biên (leading/trailing): Duy trì zero-padding chuẩn xác.
"""

import sys
from pathlib import Path

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.interpolation import (
    interpolate_vector201,
    interpolate_hand_landmarks,
    detect_missing_frames,
)


def visualize_sample(
    file_path: Path,
    output_image: Path,
    hand_choice: str = "right",
    landmark_idx: int = 0,  # 0 là cổ tay (wrist)
):
    """
    Trực quan hóa so sánh quỹ đạo tọa độ 1 landmark trước và sau nội suy.
    """
    print(f"[*] Nạp file dữ liệu: {file_path}")
    raw_data = np.load(file_path)

    is_vector201 = (raw_data.ndim == 2 and raw_data.shape[1] == 201)
    is_holistic = (raw_data.ndim == 3 and raw_data.shape[1] in (543, 75, 115))

    if is_vector201:
        if hand_choice == "left":
            raw_hand = raw_data[:, 0:63].reshape(-1, 21, 3)
            hand_label = "Tay trái (Left Hand)"
        else:
            raw_hand = raw_data[:, 63:126].reshape(-1, 21, 3)
            hand_label = "Tay phải (Right Hand)"

        interp_data, stats = interpolate_vector201(raw_data, boundary_mode="zeros")
        if hand_choice == "left":
            interp_hand = interp_data[:, 0:63].reshape(-1, 21, 3)
        else:
            interp_hand = interp_data[:, 63:126].reshape(-1, 21, 3)

    elif is_holistic:
        k_mode = "full" if raw_data.shape[1] == 543 else "hands_pose"
        if k_mode == "full":
            h_slice = slice(501, 522) if hand_choice == "left" else slice(522, 543)
        else:
            h_slice = slice(33, 54) if hand_choice == "left" else slice(54, 75)

        raw_hand = raw_data[:, h_slice, :]
        hand_label = f"Tay {'trái' if hand_choice == 'left' else 'phải'} (Holistic)"

        interp_hand, stats = interpolate_hand_landmarks(raw_hand, boundary_mode="zeros")
    else:
        raise ValueError(f"Không nhận diện được định dạng dữ liệu: shape={raw_data.shape}")

    num_frames = raw_hand.shape[0]
    frames = np.arange(num_frames)

    # Missing mask ban đầu
    missing_mask = detect_missing_frames(raw_hand)
    valid_mask = ~missing_mask
    valid_indices = np.where(valid_mask)[0]

    first_valid = int(valid_indices[0]) if len(valid_indices) > 0 else 0
    last_valid = int(valid_indices[-1]) if len(valid_indices) > 0 else num_frames - 1

    # Trích xuất tọa độ X và Y của landmark được chọn
    # Landmark 0: Wrist, 4: Thumb tip, 8: Index tip
    lm_names = {0: "Cổ tay (Wrist)", 4: "Đầu ngón cái (Thumb tip)", 8: "Đầu ngón trỏ (Index tip)"}
    lm_name = lm_names.get(landmark_idx, f"Điểm mốc {landmark_idx}")

    raw_x = raw_hand[:, landmark_idx, 0]
    raw_y = raw_hand[:, landmark_idx, 1]
    interp_x = interp_hand[:, landmark_idx, 0]
    interp_y = interp_hand[:, landmark_idx, 1]

    # Cài đặt giao diện đồ thị phong cách chuyên nghiệp
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(14, 8), dpi=200)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # 1. Subplot 1: Tọa độ X - Trước và sau nội suy
    ax1 = fig.add_subplot(gs[0, :])
    # Đánh dấu vùng Zero-padding biên
    if first_valid > 0:
        ax1.axvspan(0, first_valid - 1, color="gray", alpha=0.15, label="Zero-Padding (Biên đầu)")
    if last_valid < num_frames - 1:
        ax1.axvspan(last_valid + 1, num_frames - 1, color="gray", alpha=0.15, label="Zero-Padding (Biên cuối)")

    # Vẽ đường gốc bị drop
    ax1.plot(frames, raw_x, "r--o", markersize=4, alpha=0.7, label=f"Dữ liệu gốc (Bị mất dấu -> rơi về 0)")
    # Vẽ đường nội suy mượt
    ax1.plot(frames, interp_x, "b-d", markersize=4, linewidth=2, alpha=0.85, label=f"Sau Nội suy & Zero-Padding")

    # Đánh dấu các frame được nội suy (nằm giữa first và last nhưng gốc bị missing)
    internal_missing = [t for t in range(first_valid, last_valid + 1) if missing_mask[t]]
    if internal_missing:
        ax1.scatter(
            internal_missing,
            interp_x[internal_missing],
            color="limegreen",
            s=80,
            zorder=5,
            edgecolor="darkgreen",
            label=f"Frame được nội suy ({len(internal_missing)} frames)",
        )

    ax1.set_title(f"Quỹ đạo tọa độ X theo thời gian - {hand_label} - {lm_name}", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Chỉ số khung hình (Frame index)", fontsize=11)
    ax1.set_ylabel("Tọa độ X (Chuẩn hóa)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Subplot 2: Tọa độ Y - So sánh chi tiết
    ax2 = fig.add_subplot(gs[1, 0])
    if first_valid > 0:
        ax2.axvspan(0, first_valid - 1, color="gray", alpha=0.15)
    if last_valid < num_frames - 1:
        ax2.axvspan(last_valid + 1, num_frames - 1, color="gray", alpha=0.15)

    ax2.plot(frames, raw_y, "r--o", markersize=3, alpha=0.7, label="Gốc (Có mất dấu)")
    ax2.plot(frames, interp_y, "b-", linewidth=2, alpha=0.85, label="Đã nội suy")
    if internal_missing:
        ax2.scatter(internal_missing, interp_y[internal_missing], color="limegreen", s=60, zorder=5)

    ax2.set_title(f"Tọa độ Y - {lm_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frame", fontsize=10)
    ax2.set_ylabel("Tọa độ Y", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Subplot 3: Quỹ đạo chuyển động trong không gian 2D (X vs Y)
    ax3 = fig.add_subplot(gs[1, 1])
    # Chỉ vẽ các frame hợp lệ để thấy chuyển động cử chỉ thực tế
    ax3.plot(interp_x[first_valid:last_valid + 1], interp_y[first_valid:last_valid + 1], "b-", linewidth=2, alpha=0.7, label="Quỹ đạo nội suy liên tục")
    ax3.scatter(interp_x[first_valid:last_valid + 1], interp_y[first_valid:last_valid + 1], c=np.arange(last_valid - first_valid + 1), cmap="viridis", s=30, zorder=4)
    if internal_missing:
        ax3.scatter(interp_x[internal_missing], interp_y[internal_missing], color="limegreen", s=80, marker="*", zorder=5, label="Điểm nội suy")

    ax3.set_title(f"Quỹ đạo không gian (X vs Y) của {lm_name}", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Tọa độ X", fontsize=10)
    ax3.set_ylabel("Tọa độ Y", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[+] Đã lưu biểu đồ trực quan hóa thành công tại: {output_image.resolve()}")
    print(f"    * Tổng frame:           {num_frames}")
    print(f"    * Frame hợp lệ ban đầu: {len(valid_indices)}")
    print(f"    * Frame mất dấu nội tại:{len(internal_missing)} (Đã nội suy tuyến tính)")
    print(f"    * Frame vùng biên:      {first_valid} đầu, {num_frames - 1 - last_valid} cuối (Zero-padded)")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa so sánh trước và sau khi nội suy landmark bàn tay.")
    parser.add_argument("--file", "-f", type=str, default="data/processed_landmarks/00626.npy", help="File npy mẫu")
    parser.add_argument("--output", "-o", type=str, default="data/sample_interpolation_comparison.png", help="Đường dẫn file ảnh đầu ra")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"], help="Bàn tay cần xem")
    parser.add_argument("--landmark", "-l", type=int, default=0, help="Chỉ số landmark (0 là cổ tay, 4 là ngón cái, 8 là ngón trỏ)")
    args = parser.parse_args()

    file_p = Path(args.file)
    if not file_p.is_file():
        # Thử tìm file thay thế
        candidates = list(Path("data/processed_landmarks").glob("*.npy"))
        if candidates:
            file_p = candidates[0]
            print(f"[!] Không thấy {args.file}, dùng file thay thế: {file_p}")
        else:
            print(f"[x] Không tìm thấy file dữ liệu: {file_p}")
            return

    visualize_sample(file_p, Path(args.output), hand_choice=args.hand, landmark_idx=args.landmark)


if __name__ == "__main__":
    main()
