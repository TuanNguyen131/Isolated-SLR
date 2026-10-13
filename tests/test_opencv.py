"""
Script kiem tra hoat dong cua OpenCV (cv2) trong moi truong hien tai.
Co the chay truc tiep:
    python tests/test_opencv.py
Hoac chay qua pytest:
    pytest tests/test_opencv.py
"""

import sys
import numpy as np


def test_opencv_import_and_version():
    """Kiem tra import va phien ban OpenCV."""
    import cv2
    assert hasattr(cv2, "__version__"), "OpenCV khong co thuoc tinh __version__"
    print(f"\n[OK] OpenCV da duoc import thanh cong. Phien ban: {cv2.__version__}")


def test_opencv_basic_image_operations():
    """Kiem tra cac thao tac ma tran anh co ban (ve hinh, chuyen doi mau sac, loc anh)."""
    import cv2

    # 1. Tao anh mau 200x200
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # 2. Ve hinh chu nhat, hinh tron va text
    cv2.rectangle(img, (20, 20), (180, 180), (0, 255, 0), 2)
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    cv2.putText(img, "CV2", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 3. Chuyen doi khong gian mau BGR -> Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (200, 200), "Kich thuoc anh xam khong dung"

    # 4. Ap dung GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    assert blurred.shape == (200, 200), "Kich thuoc anh lam mo khong dung"

    # 5. Kiem tra ma hoa/giai ma anh (Encode / Decode)
    success, encoded = cv2.imencode(".png", img)
    assert success, "Khong the ma hoa anh sang dinh dang PNG"
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == img.shape, "Giai ma anh that bai"

    print("[OK] Cac thao tac xu ly anh co ban hoat dong tot.")


def test_opencv_video_module():
    """Kiem tra module VideoCapture va VideoWriter co the khoi tao."""
    import cv2

    # Khoi tao VideoCapture rong (kiem tra API bindings)
    cap = cv2.VideoCapture()
    assert not cap.isOpened(), "VideoCapture chua mo nguon nao nen isOpened phai la False"
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    assert isinstance(fourcc, int), "VideoWriter_fourcc khong tra ve integer hop le"
    print("[OK] Module Video (VideoCapture & VideoWriter) san sang.")


def main():
    print("=" * 60)
    print("KIEM TRA TRANG THAI HOAT DONG CUA OPENCV")
    print("=" * 60)
    print(f"Python: {sys.executable}")

    try:
        import cv2
    except ImportError as e:
        print(f"\n[THAT BAI] Khong the import OpenCV (cv2): {e}")
        print("Huong dan: Vui long cai dat OpenCV bang lenh:")
        print("    pip install opencv-python")
        return 1

    try:
        test_opencv_import_and_version()
        test_opencv_basic_image_operations()
        test_opencv_video_module()

        # Kiem tra CUDA neu co
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"[INFO] Ho tro CUDA: {cuda_count} GPU duoc phat hien.")
        except Exception:
            print("[INFO] OpenCV khong duoc bien dich voi ho tro CUDA (binh thuong voi opencv-python mac dinh).")

        print("=" * 60)
        print("[THANH CONG] OpenCV dang hoat dong binh thuong tren he thong cua ban!")
        print("=" * 60)
        return 0
    except AssertionError as ae:
        print(f"\n[LOI KIEM TRA] {ae}")
        return 1
    except Exception as ex:
        print(f"\n[LOI KHONG XAC DINH] {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
