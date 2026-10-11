"""
Script trực quan hóa (Visualization) các kỹ thuật Data Augmentation trên vector tọa độ:
1. Thêm nhiễu Gaussian ngẫu nhiên (Gaussian Noise)
2. Xoay góc nhẹ trong không gian 2D (Slight Rotation)
3. Co giãn kích thước tỉ lệ (Scaling)
4. Phối hợp tổng hợp (Composite Pipeline: Scale + Rotate + Noise)
"""

import sys
from pathlib import Path

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

from src.data.augmentation import (
    add_gaussian_noise,
    rotate_landmarks,
    scale_landmarks,
    CoordinateAugmentor,
)

# 21 kết nối khớp xương bàn tay MediaPipe Hands
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand_wireframe(ax, hand_pts, color="#2563eb", alpha=0.8, label="", linestyle="-", marker="o", markersize=4):
    """Vẽ wireframe khung xương bàn tay."""
    # Lật trục Y vì MediaPipe y hướng xuống
    xs = hand_pts[:, 0]
    ys = -hand_pts[:, 1]

    # Vẽ các đoạn xương nối
    for p1, p2 in HAND_CONNECTIONS:
        ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], color=color, alpha=alpha * 0.7, linestyle=linestyle, linewidth=1.5)

    # Vẽ các khớp điểm mốc
    ax.scatter(xs, ys, color=color, alpha=alpha, s=markersize**2 * 4, marker=marker, label=label, zorder=5)


def visualize_augmentations(
    file_path: Path,
    output_image: Path,
    frame_idx: int = -1,
    hand_choice: str = "right",
):
    print(f"[*] Nạp file dữ liệu: {file_path}")
    raw_data = np.load(file_path)

    if raw_data.ndim == 2 and raw_data.shape[1] == 201:
        if hand_choice == "left":
            all_hand = raw_data[:, 0:63].reshape(-1, 21, 3)
            hand_label = "Tay trái (Left Hand)"
        else:
            all_hand = raw_data[:, 63:126].reshape(-1, 21, 3)
            hand_label = "Tay phải (Right Hand)"
    elif raw_data.ndim == 3:
        all_hand = raw_data
        hand_label = "Bàn tay"
    else:
        raise ValueError(f"Shape không hợp lệ: {raw_data.shape}")

    # Tìm khung hình có cử chỉ hoạt động tốt nhất (tổng độ lệch khác 0 lớn nhất)
    if frame_idx < 0:
        activity = np.linalg.norm(all_hand, axis=(1, 2))
        best_frame = int(np.argmax(activity))
    else:
        best_frame = frame_idx

    hand_orig = all_hand[best_frame]  # (21, 3)
    if np.all(np.abs(hand_orig) < 1e-4):
        print("[!] Khung hình được chọn có bàn tay bằng 0, đang tìm frame hợp lệ khác...")
        for i in range(len(all_hand)):
            if np.any(np.abs(all_hand[i]) > 1e-3):
                hand_orig = all_hand[i]
                best_frame = i
                break

    # 1. Thêm Gaussian noise
    hand_noise = add_gaussian_noise(hand_orig, sigma=0.03, prob=1.0, seed=42)

    # 2. Xoay góc nhẹ (+12 độ quanh tâm)
    hand_rot = rotate_landmarks(hand_orig, angle_deg=12.0, axis="z", center="mean", prob=1.0)

    # 3. Co giãn tỉ lệ (Scale 1.2x quanh tâm)
    hand_scaled = scale_landmarks(hand_orig, scale_factor=1.2, center="mean", prob=1.0)

    # 4. Pipeline phối hợp
    pipeline = CoordinateAugmentor(
        apply_noise=True, noise_sigma=0.02, noise_prob=1.0,
        apply_rotation=True, rotation_max_angle=10.0, rotation_prob=1.0,
        apply_scale=True, scale_range=(1.15, 1.15), scale_prob=1.0,
        center_mode="mean",
        seed=101,
    )
    hand_combined = pipeline(hand_orig)

    # Cấu hình biểu đồ
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=200)

    # Subplot 1: Gaussian Noise
    ax1 = axes[0]
    draw_hand_wireframe(ax1, hand_orig, color="#94a3b8", alpha=0.5, label="Gốc (Original)", linestyle="--")
    draw_hand_wireframe(ax1, hand_noise, color="#ef4444", alpha=0.85, label="Gaussian Noise (σ=0.03)")
    ax1.set_title("1. Thêm Nhiễu Gaussian (Noise Injection)", fontsize=11, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Random Rotation
    ax2 = axes[1]
    draw_hand_wireframe(ax2, hand_orig, color="#94a3b8", alpha=0.5, label="Gốc (Original)", linestyle="--")
    draw_hand_wireframe(ax2, hand_rot, color="#2563eb", alpha=0.85, label="Xoay góc nhẹ (+12°)")
    ax2.set_title("2. Xoay Góc Nhẹ (Random Rotation)", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.axis("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Random Scaling
    ax3 = axes[2]
    draw_hand_wireframe(ax3, hand_orig, color="#94a3b8", alpha=0.5, label="Gốc (Original)", linestyle="--")
    draw_hand_wireframe(ax3, hand_scaled, color="#16a34a", alpha=0.85, label="Co giãn (Scale x1.2)")
    ax3.set_title("3. Co Giãn Kích Thước (Random Scaling)", fontsize=11, fontweight="bold", pad=8)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.axis("equal")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Combined Pipeline
    ax4 = axes[3]
    draw_hand_wireframe(ax4, hand_orig, color="#94a3b8", alpha=0.5, label="Gốc (Original)", linestyle="--")
    draw_hand_wireframe(ax4, hand_combined, color="#9333ea", alpha=0.85, label="Phối Hợp (Composite)")
    ax4.set_title("4. Pipeline Tổng Hợp (Composite Augmentor)", fontsize=11, fontweight="bold", pad=8)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.axis("equal")
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"KỸ THUẬT DATA AUGMENTATION TRÊN VECTOR TỌA ĐỘ KHUNG XƯƠNG ({hand_label.upper()})\n"
        f"Mẫu: {file_path.name} | Khung hình phân tích: #{best_frame}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[+] Đã lưu ảnh so sánh trực quan hóa thành công tại: {output_image.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa các kỹ thuật Data Augmentation trên tọa độ landmark.")
    parser.add_argument("--file", "-f", type=str, default="data/processed_landmarks/00626.npy", help="File npy mẫu")
    parser.add_argument("--output", "-o", type=str, default="data/sample_augmentation_comparison.png", help="Đường dẫn file ảnh đầu ra")
    parser.add_argument("--frame", type=int, default=-1, help="Chỉ số frame cần xem (-1 là tự động chọn frame hoạt động)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"], help="Bàn tay cần xem")
    args = parser.parse_args()

    file_p = Path(args.file)
    if not file_p.is_file():
        candidates = list(Path("data/processed_landmarks").glob("*.npy"))
        if candidates:
            file_p = candidates[0]
            print(f"[!] Dùng file thay thế: {file_p}")
        else:
            print(f"[x] Không tìm thấy file dữ liệu: {file_p}")
            return

    visualize_augmentations(
        file_path=file_p,
        output_image=Path(args.output),
        frame_idx=args.frame,
        hand_choice=args.hand,
    )


if __name__ == "__main__":
    main()
