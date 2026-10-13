"""
Script test camera/webcam su dung OpenCV.
Chay lenh:
    python tests/test_camera.py
Nhan 'q' hoac 'ESC' tren cua so camera de thoat.
"""

import sys
import time
import cv2


def test_webcam(camera_index: int = 0):
    print("=" * 60)
    print(f"DANG KET NOI VOI CAMERA (Index: {camera_index})...")
    print("=" * 60)

    # Tren Windows, cv2.CAP_DSHOW giup khoi dong camera nhanh va on dinh hon
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Neu DirectShow khong mo duoc, thu che do mac dinh
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[LOI] Khong the ket noi toi camera tai index {camera_index}!")
        print("Cac nguyen nhan pho bien:")
        print("  1. May tinh chua cam webcam hoac webcam bi tat cong tac vat ly.")
        print("  2. Quyen truy cap Camera trong Windows Privacy Settings chua duoc cap.")
        print("  3. Mot ung dung khac (Zoom, Teams, Browser, ...) dang chiem webcam.")
        return False

    # Lay thong tin do phan giai
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[OK] Camera da ket noi!")
    print(f"     Do phan giai: {width} x {height}")
    print(f"     FPS goc: {fps if fps > 0 else 'N/A'}")
    print("\nDang mo cua so camera...")
    print(">>> Nhan phim 'q' hoac 'ESC' tren cua so video de THOAT <<<\n")

    window_name = "Test OpenCV Webcam (Nhan 'q' de thoat)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    frame_count = 0
    actual_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CANH BAO] Khong doc duoc frame tu camera.")
            break

        frame_count += 1
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff >= 1.0:
            actual_fps = frame_count / time_diff
            frame_count = 0
            prev_time = curr_time

        # Hien thi thong so FPS va huong dan thoat tren man hinh
        info_text = f"FPS: {actual_fps:.1f} | Res: {width}x{height} | Press 'q' to Quit"
        cv2.putText(
            frame,
            info_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)

        # Nhan 'q' hoac ESC (ma ascii 27) de thoat
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("[INFO] Nguoi dung da dong camera.")
            break

    # Giai phong camera va dong cua so
    cap.release()
    cv2.destroyAllWindows()
    print("[THANH CONG] Da giai phong camera an toan.")
    return True


if __name__ == "__main__":
    success = test_webcam(0)
    sys.exit(0 if success else 1)
