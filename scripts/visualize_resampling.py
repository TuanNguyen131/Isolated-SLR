"""
Script trực quan hóa (Visualization) kỹ thuật Temporal Resampling cố định chuỗi độ dài
T = 30 khung hình qua nội suy tuyến tính (Linear Interpolation) cho Isolated-SLR.

Vẽ biểu đồ so sánh:
1. Quỹ đạo tọa độ X và Y theo dòng thời gian chuẩn hóa t in [0, 1].
2. So sánh mật độ phân bố frame trước (N frames) và sau khi resample (T = 30 frames).
3. Quỹ đạo không gian 2D (X vs Y) chứng minh cử chỉ được bảo toàn mượt mà.
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
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.interpolation import temporal_resample, interpolate_vector201


def visualize_resampling(
    file_path: Path,
    output_image: Path,
    target_len: int = 30,
    hand_choice: str = "right",
    landmark_idx: int = 0,  # 0: Wrist, 4: Thumb, 8: Index tip
):
    print(f"[*] Nạp file dữ liệu: {file_path}")
    raw_data = np.load(file_path)

    # 1. Trích xuất mảng bàn tay
    if raw_data.ndim == 2 and raw_data.shape[1] == 201:
        # Nội suy các frame mất dấu nội tại trước nếu có
        clean_data, _ = interpolate_vector201(raw_data, boundary_mode="zeros")
        if hand_choice == "left":
            hand_seq = clean_data[:, 0:63].reshape(-1, 21, 3)
            hand_label = "Tay trái (Left Hand)"
        else:
            hand_seq = clean_data[:, 63:126].reshape(-1, 21, 3)
            hand_label = "Tay phải (Right Hand)"
    elif raw_data.ndim == 3:
        hand_seq = raw_data
        hand_label = f"Bàn tay ({hand_choice})"
    else:
        raise ValueError(f"Shape dữ liệu không hợp lệ: {raw_data.shape}")

    orig_num_frames = hand_seq.shape[0]

    # 2. Áp dụng kỹ thuật Temporal Resampling cố định T = target_len khung hình
    resampled_seq = temporal_resample(hand_seq, target_len=target_len, kind="linear")

    # 3. Lấy tọa độ landmark quan sát
    lm_names = {
        0: "Cổ tay (Wrist)",
        4: "Đầu ngón cái (Thumb tip)",
        8: "Đầu ngón trỏ (Index tip)",
        12: "Đầu ngón giữa (Middle tip)",
    }
    lm_name = lm_names.get(landmark_idx, f"Khớp {landmark_idx}")

    orig_x = hand_seq[:, landmark_idx, 0]
    orig_y = hand_seq[:, landmark_idx, 1]
    res_x = resampled_seq[:, landmark_idx, 0]
    res_y = resampled_seq[:, landmark_idx, 1]

    # Trục thời gian chuẩn hóa [0, 1] để đối chiếu trực quan
    t_norm_orig = np.linspace(0.0, 1.0, orig_num_frames)
    t_norm_res = np.linspace(0.0, 1.0, target_len)

    # Cấu hình giao diện biểu đồ
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 9), dpi=200)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.22)

    # ---------------- Subplot 1: Tọa độ X theo thời gian chuẩn hóa ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        t_norm_orig, orig_x,
        color="#ef4444", linestyle="--", marker="o", markersize=4, alpha=0.65,
        label=f"Gốc (N = {orig_num_frames} frames)",
    )
    ax1.plot(
        t_norm_res, res_x,
        color="#2563eb", linestyle="-", marker="s", markersize=4, linewidth=2.0, alpha=0.85,
        label=f"Temporal Resampling (T = {target_len} frames)",
    )
    ax1.set_title(f"Quỹ đạo tọa độ X - {hand_label} - {lm_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Thời gian chuẩn hóa (Normalized Time t in [0, 1])", fontsize=10)
    ax1.set_ylabel("Tọa độ X", fontsize=10)
    ax1.legend(loc="best", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---------------- Subplot 2: Tọa độ Y theo thời gian chuẩn hóa ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        t_norm_orig, orig_y,
        color="#ef4444", linestyle="--", marker="o", markersize=4, alpha=0.65,
        label=f"Gốc (N = {orig_num_frames} frames)",
    )
    ax2.plot(
        t_norm_res, res_y,
        color="#16a34a", linestyle="-", marker="^", markersize=4, linewidth=2.0, alpha=0.85,
        label=f"Temporal Resampling (T = {target_len} frames)",
    )
    ax2.set_title(f"Quỹ đạo tọa độ Y - {hand_label} - {lm_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Thời gian chuẩn hóa (Normalized Time t in [0, 1])", fontsize=10)
    ax2.set_ylabel("Tọa độ Y", fontsize=10)
    ax2.legend(loc="best", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # ---------------- Subplot 3: Mật độ khung hình rời rạc (Frame Indices) ----------------
    ax3 = fig.add_subplot(gs[1, 0])
    # Vẽ các vạch thời gian rời rạc
    ax3.vlines(
        np.arange(orig_num_frames), 0.55, 0.95,
        colors="#ef4444", linewidth=1.5, alpha=0.8,
        label=f"Chuỗi gốc (N = {orig_num_frames} frames rời rạc)",
    )
    ax3.vlines(
        np.linspace(0, orig_num_frames - 1, target_len), 0.05, 0.45,
        colors="#2563eb", linewidth=2.0, alpha=0.85,
        label=f"Resampled (T = {target_len} frames phân bố đều)",
    )
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels([f"T = {target_len}\n(Resampled)", f"N = {orig_num_frames}\n(Gốc)"], fontsize=10, fontweight="bold")
    ax3.set_xlabel("Thang đo khung hình gốc (Original Frame Index)", fontsize=10)
    ax3.set_title("So sánh mật độ lấy mẫu khung hình theo thời gian", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # ---------------- Subplot 4: Quỹ đạo không gian 2D (X vs Y) ----------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        orig_x, orig_y,
        color="#94a3b8", linestyle=":", linewidth=2.5, alpha=0.8,
        label="Quỹ đạo gốc",
    )
    ax4.plot(
        res_x, res_y,
        color="#8b5cf6", linestyle="-", linewidth=2.0, alpha=0.9,
        label=f"Quỹ đạo Resampled (T = {target_len})",
    )
    # Scatter điểm bắt đầu và kết thúc
    ax4.scatter([res_x[0]], [res_y[0]], color="#16a34a", s=90, zorder=5, marker="o", label="Điểm bắt đầu (t=0)")
    ax4.scatter([res_x[-1]], [res_y[-1]], color="#dc2626", s=90, zorder=5, marker="X", label=f"Điểm kết thúc (t={target_len-1})")
    ax4.set_title(f"Quỹ đạo không gian 2D (X vs Y) của {lm_name}", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Tọa độ X", fontsize=10)
    ax4.set_ylabel("Tọa độ Y", fontsize=10)
    ax4.legend(loc="best", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.6)

    # Tiêu đề tổng quát
    scale_type = "Downsampling (Nén chuỗi)" if orig_num_frames > target_len else ("Upsampling (Kéo dãn chuỗi)" if orig_num_frames < target_len else "Equal (Độ dài bằng nhau)")
    fig.suptitle(
        f"KỸ THUẬT TEMPORAL RESAMPLING (NỘI SUY TUYẾN TÍNH CỐ ĐỊNH T = {target_len} KHUNG HÌNH)\n"
        f"File: {file_path.name} | Gốc: {orig_num_frames} frames -> Mục tiêu: {target_len} frames ({scale_type})",
        fontsize=13, fontweight="bold", y=0.98,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[+] Đã tạo thành công ảnh biểu đồ trực quan hóa tại: {output_image.resolve()}")
    print(f"    * Số frame gốc (N):       {orig_num_frames}")
    print(f"    * Số frame resampled (T): {target_len}")
    print(f"    * Tỉ lệ thời gian:        {target_len / orig_num_frames:.3f}x")
    print(f"    * Chế độ biến đổi:        {scale_type}")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa kỹ thuật Temporal Resampling T = 30 khung hình.")
    parser.add_argument("--file", "-f", type=str, default="data/processed_landmarks/00626.npy", help="File npy mẫu")
    parser.add_argument("--output", "-o", type=str, default="data/sample_resampling_comparison.png", help="Đường dẫn file ảnh đầu ra")
    parser.add_argument("--target_len", "-t", type=int, default=30, help="Số khung hình mục tiêu T (mặc định: 30)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"], help="Bàn tay cần xem")
    parser.add_argument("--landmark", "-l", type=int, default=0, help="Chỉ số landmark (0: Wrist, 4: Thumb, 8: Index)")
    args = parser.parse_args()

    file_p = Path(args.file)
    if not file_p.is_file():
        candidates = list(Path("data/processed_landmarks").glob("*.npy"))
        if candidates:
            file_p = candidates[0]
            print(f"[!] Không thấy {args.file}, dùng file thay thế: {file_p}")
        else:
            print(f"[x] Không tìm thấy file dữ liệu: {file_p}")
            return

    visualize_resampling(
        file_path=file_p,
        output_image=Path(args.output),
        target_len=args.target_len,
        hand_choice=args.hand,
        landmark_idx=args.landmark,
    )


if __name__ == "__main__":
    main()
