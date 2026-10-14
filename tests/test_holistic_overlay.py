"""
Script test MediaPipe Holistic: doc tung khung hinh, trich xuat landmark va ve khung xuong (Overlay) truc tiep.
Ho tro ca Webcam va Video tap tin (.mp4) voi bo dieu khien tuong tac truc quan.

Cach chay:
    # 1. Test truc tiep voi Webcam mac dinh (index 0):
    python tests/test_holistic_overlay.py --source 0

    # 2. Test voi mot video mau trong tap du lieu WLASL-100:
    python tests/test_holistic_overlay.py --source data/raw_videos/00618.mp4

Dieu khien tren cua so hien thi:
    [Space] : Tam dung (Pause) / Tiep tuc (Resume) de soi ky toa do
    [D]     : Tien 1 khung hinh (khi dang Pause)
    [R]     : Phat lai tu dau (khi xem video tap tin)
    [S]     : Luu anh chup overlay (Snapshot) ra thu muc tests/snapshots/
    [Q]/ESC : Thoat chuong trinh
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# Dam bao UTF-8 hoat dong chuan tren Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    total_frames: int,
    results: any,
    is_paused: bool,
    source_name: str,
) -> np.ndarray:
    """
    Ve bang thong tin (HUD Dashboard) len goc tren ben trai cua khung hinh.
    """
    h, w = frame.shape[:2]

    # Tao vung banner ban trong suot
    banner_w = min(360, w - 20)
    banner_h = 165
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + banner_w, 10 + banner_h), (80, 80, 80), 1)

    # Thong tin nguon & FPS
    frame_info = f"Frame: {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else "")
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {frame_info}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Nguon video
    cv2.putText(
        frame,
        f"Source: {source_name[:28]}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Trang thai Pose
    has_pose = results.pose_landmarks is not None
    pose_str = "OK (33 pts)" if has_pose else "LOST"
    pose_color = (0, 255, 0) if has_pose else (0, 0, 255)
    cv2.putText(
        frame,
        f"Pose:       {pose_str}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        pose_color,
        1,
        cv2.LINE_AA,
    )

    # Trang thai Left Hand
    has_lh = results.left_hand_landmarks is not None
    lh_str = "OK (21 pts)" if has_lh else "LOST"
    lh_color = (0, 255, 0) if has_lh else (0, 0, 255)
    cv2.putText(
        frame,
        f"Left Hand:  {lh_str}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        lh_color,
        1,
        cv2.LINE_AA,
    )

    # Trang thai Right Hand
    has_rh = results.right_hand_landmarks is not None
    rh_str = "OK (21 pts)" if has_rh else "LOST"
    rh_color = (0, 255, 0) if has_rh else (0, 0, 255)
    cv2.putText(
        frame,
        f"Right Hand: {rh_str}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        rh_color,
        1,
        cv2.LINE_AA,
    )

    # Thong bao trang thai Pause / Guide
    if is_paused:
        cv2.rectangle(frame, (w - 290, 10), (w - 10, 50), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "PAUSED [Space / D]",
            (w - 275, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Huong dan phim tat duoi chan man hinh
    guide_str = "[Space]: Pause/Resume | [D]: Step | [S]: Save | [R]: Replay | [Q]: Quit"
    cv2.putText(
        frame,
        guide_str,
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame


def test_holistic_visualizer(
    source_arg: str = "0",
    complexity: int = 1,
    min_det_conf: float = 0.5,
    min_track_conf: float = 0.5,
    refine_face: bool = True,
):
    print("=" * 70)
    print("   KHOI DONG TEST MEDIAPIPE HOLISTIC FRAME-BY-FRAME OVERLAY")
    print("=" * 70)

    # Xac dinh nguon video (Webcam index hay File Path)
    is_webcam = source_arg.isdigit()
    source_name = f"Webcam #{source_arg}" if is_webcam else Path(source_arg).name

    if is_webcam:
        cam_idx = int(source_arg)
        print(f"[*] Dang mo Webcam (Index: {cam_idx})...")
        # Tren Windows, cv2.CAP_DSHOW giup khoi dong nhanh va on dinh
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[!] DirectShow that bai, thu lai voi che do mac dinh...")
            cap = cv2.VideoCapture(cam_idx)
    else:
        video_path = Path(source_arg)
        if not video_path.exists():
            print(f"[LOI] Khong tim thay file video tai duong dan: {video_path}")
            return
        print(f"[*] Dang mo Video tap tin: {video_path}...")
        cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[LOI] Khong the mo luong video tu nguon: {source_arg}")
        return

    # Lay thong tin video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[OK] Mo nguon thanh cong: {width}x{height} | FPS goc: {src_fps:.1f}")
    if total_frames > 0:
        print(f"     Tong so khung hinh: {total_frames}")

    # Thu muc luu snapshot
    snapshot_dir = Path(__file__).resolve().parent / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Khoi tao MediaPipe Holistic va cac cong cu ve
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print(f"[*] Dang khoi tao MediaPipe Holistic (complexity={complexity})...")
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
        refine_face_landmarks=refine_face,
    )
    print("[OK] MediaPipe Holistic da san sang!\n")

    window_name = "MediaPipe Holistic Test Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, max(640, width)), min(720, max(480, height)))

    prev_time = time.time()
    frame_idx = 0
    is_paused = False
    current_frame = None

    print(">>> DANG TRUC QUAN HOA. Nhan [Space] de dung, [S] de luu anh, [Q] de thoat <<<\n")

    try:
        while cap.isOpened():
            if not is_paused or current_frame is None:
                ret, frame = cap.read()
                if not ret:
                    if not is_webcam and total_frames > 0:
                        print("[*] Da xem het video. Nhan [R] de phat lai hoac [Q] de thoat.")
                        is_paused = True
                    else:
                        print("[!] Khong the doc khung hinh tiep theo.")
                        break
                else:
                    frame_idx += 1
                    current_frame = frame

            if current_frame is None:
                continue

            # 1. Chuyen doi sang RGB va dat cờ writeable=False de toi uu hoa MediaPipe
            rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # 2. Tao ban sao de ve overlay len anh BGR
            annotated = current_frame.copy()

            # --- A. Ve Khuon mat (Face Contours) ---
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            # --- B. Ve Khung xuong co the (Pose) ---
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # --- C. Ve Ban tay trai (Left Hand) ---
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )

            # --- D. Ve Ban tay phai (Right Hand) ---
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )

            # 3. Tinh toan FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_time = curr_time

            # 4. Ve HUD thong tin len anh
            draw_hud(
                annotated,
                fps=fps,
                frame_idx=frame_idx,
                total_frames=total_frames,
                results=results,
                is_paused=is_paused,
                source_name=source_name,
            )

            # 5. Hien thi len man hinh
            cv2.imshow(window_name, annotated)

            # 6. Xu ly su kien ban phim
            key_delay = 0 if is_paused else 20
            key = cv2.waitKey(key_delay) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 'q' hoac ESC
                print("[*] Nguoi dung yeu cau thoat chuong trinh.")
                break

            elif key == ord(" "):  # Phim Space: Pause / Resume
                is_paused = not is_paused
                status = "DUNG (PAUSED)" if is_paused else "TIEP TUC (PLAYING)"
                print(f"[*] Trang thai: {status}")

            elif key in (ord("d"), ord("D")) and is_paused:  # Phim D: Next frame khi pause
                ret, frame = cap.read()
                if ret:
                    frame_idx += 1
                    current_frame = frame
                    print(f"[*] Da tien toi khung hinh: {frame_idx}")
                else:
                    print("[!] Da toi khung hinh cuoi cung cua video.")

            elif key in (ord("r"), ord("R")):  # Phim R: Reset video tu dau
                if not is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    is_paused = False
                    print("[*] Da khoi dong lai video tu dau.")

            elif key in (ord("s"), ord("S")):  # Phim S: Snapshot
                snap_path = snapshot_dir / f"snapshot_{Path(source_name).stem}_frame_{frame_idx}.png"
                cv2.imwrite(str(snap_path), annotated)
                print(f"[+] DA LUU ANH SNAPSHOT: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("[OK] Da giai phong toan bo tai nguyen camera va MediaPipe.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool test truc quan MediaPipe Holistic frame-by-frame co overlay khung xuong."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="data/raw_videos/00618.mp4",
        help="Nguon phat video: nhap so (vd: '0') cho Webcam, hoac duong dan file .mp4 (mac dinh: data/raw_videos/00618.mp4)",
    )
    parser.add_argument(
        "--complexity",
        "-c",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Do phuc tap cua model MediaPipe (0: nhanh/nhe, 1: can bang, 2: chinh xac nhat). Mac dinh: 1",
    )
    parser.add_argument(
        "--min_det_conf",
        type=float,
        default=0.5,
        help="Nguong tin cay toi thieu de phat hien (min_detection_confidence). Mac dinh: 0.5",
    )
    parser.add_argument(
        "--min_track_conf",
        type=float,
        default=0.5,
        help="Nguong tin cay toi thieu de theo dau (min_tracking_confidence). Mac dinh: 0.5",
    )
    parser.add_argument(
        "--no_refine_face",
        action="store_true",
        help="Tat tinh nang lam min/chi tiet hoa moi va dong tu de tang toc",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_holistic_visualizer(
        source_arg=args.source,
        complexity=args.complexity,
        min_det_conf=args.min_det_conf,
        min_track_conf=args.min_track_conf,
        refine_face=not args.no_refine_face,
    )
