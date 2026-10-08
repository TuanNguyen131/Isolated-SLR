"""
Script trực quan hóa (Visualization) thuật toán Chuẩn hóa tọa độ:
1. Thân trên: Dời gốc về trung điểm hai vai, co giãn theo khoảng cách hai vai.
2. Bàn tay: Dời gốc về cổ tay (Wrist Landmark 0), co giãn theo khoảng cách hai vai.

Vẽ biểu đồ so sánh:
- Trước chuẩn hóa (Hệ tọa độ vai toàn cục)
- Sau chuẩn hóa (Bàn tay dời gốc về cổ tay, hình thái ngón tay bất biến vị trí)
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
    except Exception:
        pass

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.normalization import (
    compute_shoulder_reference,
    normalize_hand_landmarks,
    normalize_pose_landmarks,
)

# 21 kết nối khớp xương bàn tay MediaPipe Hands
HAND_CONNECTIONS = [
    # Cổ tay tới các gốc ngón
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # Ngón cái
    (1, 2), (2, 3), (3, 4),
    # Ngón trỏ
    (5, 6), (6, 7), (7, 8),
    # Ngón giữa
    (9, 10), (10, 11), (11, 12),
    # Ngón áp út
    (13, 14), (14, 15), (15, 16),
    # Ngón út
    (17, 18), (18, 19), (19, 20),
    # Khớp bàn tay nối ngang
    (5, 9), (9, 13), (13, 17),
]

FINGER_COLORS = {
    "thumb": "#e11d48",   # đỏ hồng
    "index": "#2563eb",   # xanh dương
    "middle": "#16a34a",  # xanh lá
    "ring": "#d97706",    # cam
    "pinky": "#9333ea",   # tím
    "palm": "#475569",    # xám đen
}


def draw_hand_skeleton(ax, hand_pts, title="", is_centered=False):
    """Vẽ khung xương 21 điểm của bàn tay lên trục ax."""
    # Lật trục Y vì MediaPipe y hướng xuống
    xs = hand_pts[:, 0]
    ys = -hand_pts[:, 1]

    # Vẽ các đoạn xương nối
    for p1, p2 in HAND_CONNECTIONS:
        # Chọn màu theo ngón
        if p1 in (1, 2, 3) or p2 in (2, 3, 4):
            color = FINGER_COLORS["thumb"]
        elif p1 in (5, 6, 7) or p2 in (6, 7, 8):
            color = FINGER_COLORS["index"]
        elif p1 in (9, 10, 11) or p2 in (10, 11, 12):
            color = FINGER_COLORS["middle"]
        elif p1 in (13, 14, 15) or p2 in (14, 15, 16):
            color = FINGER_COLORS["ring"]
        elif p1 in (17, 18, 19) or p2 in (18, 19, 20):
            color = FINGER_COLORS["pinky"]
        else:
            color = FINGER_COLORS["palm"]

        ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], color=color, linewidth=2, alpha=0.85)

    # Vẽ các điểm khớp
    ax.scatter(xs, ys, color="#1e293b", s=35, zorder=5, edgecolor="white", linewidth=1.2)

    # Đánh dấu cổ tay (Landmark 0) nổi bật
    ax.scatter([xs[0]], [ys[0]], color="#dc2626", s=100, zorder=6, edgecolor="black", linewidth=1.5, label="Cổ tay (Wrist Landmark 0)")

    # Đánh dấu các đầu ngón tay
    fingertips = [4, 8, 12, 16, 20]
    ax.scatter(xs[fingertips], ys[fingertips], color="#0284c7", s=60, zorder=6, label="Đầu 5 ngón tay")

    if is_centered:
        # Vẽ dấu chữ thập tại gốc (0, 0)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax.plot(0, 0, marker="+", markersize=14, color="crimson", markeredgewidth=2)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)


def visualize_normalization_sample(
    file_path: Path,
    output_image: Path,
    frame_idx: int = 15,
):
    """
    Trực quan hóa so sánh chuẩn hóa cho 1 khung hình mẫu.
    """
    print(f"[*] Nạp file dữ liệu: {file_path}")
    raw_data = np.load(file_path)

    if raw_data.ndim == 2 and raw_data.shape[1] == 201:
        rh_raw = raw_data[:, 63:126].reshape(-1, 21, 3)
        pose_raw = raw_data[:, 126:201].reshape(-1, 25, 3)
    elif raw_data.ndim == 3 and raw_data.shape[1] == 543:
        rh_raw = raw_data[:, 522:543, :]
        pose_raw = raw_data[:, 0:25, :]
    else:
        raise ValueError(f"Định dạng không được hỗ trợ: {raw_data.shape}")

    # Tìm frame hợp lệ có cả bàn tay và vai
    num_frames = raw_data.shape[0]
    valid_frames = []
    for t in range(num_frames):
        if np.abs(rh_raw[t, :, :2]).sum() > 0.05 and np.abs(pose_raw[t, 11:13, :2]).sum() > 0.05:
            valid_frames.append(t)

    if not valid_frames:
        target_f = 0
    elif frame_idx in valid_frames:
        target_f = frame_idx
    else:
        target_f = valid_frames[len(valid_frames) // 2]

    print(f"[*] Sử dụng Frame index: {target_f} / {num_frames}")

    pose_f = pose_raw[target_f]
    hand_f = rh_raw[target_f]

    # Tính center và scale vai
    center, shoulder_dist = compute_shoulder_reference(pose_f)
    print(f"    * Trung điểm 2 vai: ({center[0]:.4f}, {center[1]:.4f})")
    print(f"    * Khoảng cách 2 vai: {shoulder_dist:.4f}")

    # 1. Trạng thái cũ: Chuẩn hóa theo vai (Shoulder-centered hand)
    hand_shoulder_norm = hand_f.copy()
    hand_wrist_norm = normalize_hand_landmarks(hand_f, scale=shoulder_dist, hand_origin="wrist")

    # Tạo figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 6.5), dpi=200)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)

    # Panel 1: Toàn cảnh Thân trên & Vị trí Bàn tay (Body Context)
    ax1 = fig.add_subplot(gs[0, 0])
    # Vẽ Pose thân trên
    p_x = pose_f[:, 0]
    p_y = -pose_f[:, 1]
    ax1.scatter(p_x, p_y, color="#64748b", s=25, label="Thân trên (Pose)")
    # Vai
    ax1.plot([p_x[11], p_x[12]], [p_y[11], p_y[12]], "g-", linewidth=3, label=f"Khoảng cách 2 vai ({shoulder_dist:.2f})")
    ax1.plot(center[0], -center[1], "r*", markersize=14, label="Gốc trung điểm vai (0,0)")
    # Vẽ bàn tay tại vị trí thực trong khung hình
    h_x = hand_f[:, 0]
    h_y = -hand_f[:, 1]
    ax1.scatter(h_x, h_y, color="#2563eb", s=30, label="Bàn tay phải (Vị trí thực)")
    for p1, p2 in HAND_CONNECTIONS:
        ax1.plot([h_x[p1], h_x[p2]], [h_y[p1], h_y[p2]], color="#3b82f6", linewidth=1.5, alpha=0.7)
    ax1.set_title("1. Khung hình toàn cảnh (Hệ tọa độ Thân người)", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)

    # Panel 2: Bàn tay khi CHƯA dời gốc về cổ tay (Bị phụ thuộc vị trí giơ tay)
    ax2 = fig.add_subplot(gs[0, 1])
    wrist_xy = hand_shoulder_norm[0, :2]
    draw_hand_skeleton(
        ax2,
        hand_shoulder_norm,
        title=f"2. Trước dời gốc: Cổ tay tại ({wrist_xy[0]:.2f}, {wrist_xy[1]:.2f})\n(Bị phụ thuộc vị trí vung tay)",
        is_centered=False,
    )

    # Panel 3: Bàn tay SAU KHI DỜI GỐC VỀ CỔ TAY (Wrist-Centered & Shoulder-Scaled)
    ax3 = fig.add_subplot(gs[0, 2])
    draw_hand_skeleton(
        ax3,
        hand_wrist_norm,
        title="3. Sau dời gốc: Cổ tay tại Gốc (0, 0)\n(Hình thái ngón tay bất biến vị trí)",
        is_centered=True,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[+] Đã lưu biểu đồ chuẩn hóa tọa độ tại: {output_image.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa thuật toán chuẩn hóa tọa độ (cổ tay / khoảng cách 2 vai).")
    parser.add_argument("--file", "-f", type=str, default="data/processed_landmarks/00618.npy", help="File npy mẫu")
    parser.add_argument("--output", "-o", type=str, default="data/sample_normalization_comparison.png", help="Đường dẫn ảnh xuất")
    parser.add_argument("--frame", type=int, default=15, help="Index frame cần trực quan hóa")
    args = parser.parse_args()

    file_p = Path(args.file)
    if not file_p.is_file():
        candidates = list(Path("data/processed_landmarks").glob("*.npy"))
        if candidates:
            file_p = candidates[0]
        else:
            print(f"[x] Không tìm thấy file: {file_p}")
            return

    visualize_normalization_sample(file_p, Path(args.output), frame_idx=args.frame)


if __name__ == "__main__":
    main()
