"""
Script đa tiến trình (multiprocessing) trích xuất toàn bộ landmarks/đặc trưng
cho tập dữ liệu video WLASL-100 sử dụng MediaPipe Holistic.

Hỗ trợ:
- Đa tiến trình song song (multiprocessing pool) tối ưu hóa CPU đa nhân
- Lưu ra đồng thời hoặc tùy chọn định dạng .npy (từng video) và file HDF5 (.h5) tổng hợp
- Tương thích 100% với cấu trúc dữ liệu WLASL và pipeline huấn luyện PyTorch
- Tính năng Resume (bỏ qua video đã trích xuất hợp lệ)
- Báo cáo thống kê chi tiết và ảnh preview minh họa khung xương (skeleton visualization)
"""

import os
import sys
import time
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Tắt cảnh báo deprecation từ protobuf / mediapipe
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

# Thêm PROJECT_ROOT vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo UTF-8 cho Windows Console
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
import h5py
from tqdm import tqdm
import multiprocessing as mp

from src.utils.holistic_config import HolisticConfig
from src.utils.mediapipe_holistic import MediaPipeHolisticExtractor
from src.data.feature_extractor import SignFeatureExtractor


# Biến toàn cục lưu extractor cho từng worker process
_worker_extractor = None
_worker_feature_type = "holistic"


def init_worker(feature_type: str, config_path: Optional[str] = None):
    """
    Khởi tạo worker process độc lập.
    Mỗi process sở hữu một phiên bản Extractor riêng biệt và giới hạn 1 thread cho OpenCV/BLAS
    để tránh hiện tượng tranh chấp tài nguyên (thread thrashing) khi chạy song song.
    """
    global _worker_extractor, _worker_feature_type
    _worker_feature_type = feature_type

    # Giới hạn thread nội bộ trong từng worker
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if feature_type == "holistic":
        cfg = HolisticConfig.load_from_yaml(config_path) if config_path else HolisticConfig()
        _worker_extractor = MediaPipeHolisticExtractor(cfg)
    elif feature_type == "vector201":
        _worker_extractor = SignFeatureExtractor(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            normalize=True,
        )
    else:
        raise ValueError(f"feature_type không hợp lệ: {feature_type}")


def process_single_video(task_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm xử lý một video clip đơn lẻ trong worker process.
    
    Args:
        task_item: Dictionary chứa thông tin task
            {
                "video_id": str,
                "video_path": str,
                "gloss": str,
                "label": int,
                "split": str,
                "output_dir": str,
                "save_npy": bool,
                "skip_existing": bool,
                "save_preview": bool,
            }
            
    Returns:
        Dictionary kết quả xử lý
    """
    global _worker_extractor, _worker_feature_type
    
    video_id = task_item["video_id"]
    video_path = Path(task_item["video_path"])
    gloss = task_item.get("gloss", "")
    label = task_item.get("label", -1)
    split = task_item.get("split", "unknown")
    output_dir = Path(task_item["output_dir"])
    save_npy = task_item.get("save_npy", True)
    skip_existing = task_item.get("skip_existing", True)
    
    npy_file = output_dir / f"{video_id}.npy"
    
    # 1. Kiểm tra nếu file .npy đã tồn tại và hợp lệ
    if skip_existing and save_npy and npy_file.is_file():
        try:
            existing_arr = np.load(npy_file)
            if existing_arr.size > 0:
                return {
                    "video_id": video_id,
                    "gloss": gloss,
                    "label": label,
                    "split": split,
                    "status": "SKIPPED",
                    "shape": existing_arr.shape,
                    "frames": existing_arr.shape[0],
                    "features": existing_arr,  # Trả về mảng để cập nhật HDF5 nếu cần
                    "error": None,
                }
        except Exception:
            pass  # Nếu file hỏng thì trích xuất lại
            
    if not video_path.is_file():
        return {
            "video_id": video_id,
            "gloss": gloss,
            "label": label,
            "split": split,
            "status": "ERROR",
            "shape": None,
            "frames": 0,
            "features": None,
            "error": f"Video file not found: {video_path}",
        }
        
    try:
        t0 = time.time()
        
        # 2. Thực hiện trích xuất landmarks
        if _worker_feature_type == "holistic":
            features = _worker_extractor.extract_from_video(video_path)
        else:
            features = _worker_extractor.extract_video(video_path)
            
        extract_time = time.time() - t0
        
        if features is None or features.shape[0] == 0:
            return {
                "video_id": video_id,
                "gloss": gloss,
                "label": label,
                "split": split,
                "status": "EMPTY",
                "shape": None,
                "frames": 0,
                "features": None,
                "error": "No frames extracted or video corrupted",
            }
            
        # 3. Lưu file .npy nếu được yêu cầu
        if save_npy:
            npy_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_file, features)
            
        return {
            "video_id": video_id,
            "gloss": gloss,
            "label": label,
            "split": split,
            "status": "SUCCESS",
            "shape": features.shape,
            "frames": features.shape[0],
            "extract_time": extract_time,
            "features": features,
            "error": None,
        }
        
    except Exception as e:
        return {
            "video_id": video_id,
            "gloss": gloss,
            "label": label,
            "split": split,
            "status": "ERROR",
            "shape": None,
            "frames": 0,
            "features": None,
            "error": str(e),
        }


def load_dataset_metadata(
    annotations_path: Path,
    videos_dir: Path,
    split_filter: str = "all",
    class_mapping_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[int, str]]:
    """
    Tải danh sách các video cần xử lý cùng thông tin nhãn (gloss, label_id, split).
    """
    # Nạp class mapping nếu có
    gloss_to_id = {}
    id_to_gloss = {}
    if class_mapping_path and class_mapping_path.is_file():
        try:
            with open(class_mapping_path, "r", encoding="utf-8") as f:
                cmap = json.load(f)
                gloss_to_id = cmap.get("gloss_to_id", {})
                id_to_gloss = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in cmap.get("id_to_gloss", {}).items()}
        except Exception:
            pass

    targets = []
    
    if annotations_path.is_file():
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        splits_to_check = ["train", "val", "test"] if split_filter == "all" else [split_filter]

        if isinstance(data, dict) and any(k in data for k in ["train", "val", "test"]):
            for sp in splits_to_check:
                for item in data.get(sp, []):
                    vid_id = item.get("video_id")
                    v_file = videos_dir / f"{vid_id}.mp4"
                    if v_file.is_file():
                        gloss = item.get("gloss", "")
                        label = item.get("label")
                        if label is None and gloss in gloss_to_id:
                            label = gloss_to_id[gloss]
                        elif label is None:
                            label = -1
                        targets.append({
                            "video_id": vid_id,
                            "gloss": gloss,
                            "label": int(label),
                            "split": sp,
                            "video_path": str(v_file),
                            "source_meta": item,
                        })
        elif isinstance(data, list):
            for item in data:
                vid_id = item.get("video_id")
                sp = item.get("split", "unknown")
                if split_filter == "all" or sp == split_filter:
                    v_file = videos_dir / f"{vid_id}.mp4"
                    if v_file.is_file():
                        gloss = item.get("gloss", "")
                        label = item.get("label", gloss_to_id.get(gloss, -1))
                        targets.append({
                            "video_id": vid_id,
                            "gloss": gloss,
                            "label": int(label),
                            "split": sp,
                            "video_path": str(v_file),
                            "source_meta": item,
                        })

    # Nếu không có file annotation hoặc annotation rỗng -> quét trực tiếp folder
    if not targets:
        for v_file in sorted(videos_dir.glob("*.mp4")):
            targets.append({
                "video_id": v_file.stem,
                "gloss": "unknown",
                "label": -1,
                "split": "unknown",
                "video_path": str(v_file),
                "source_meta": {},
            })

    return targets, gloss_to_id, id_to_gloss


def save_preview_image(video_path: Path, output_image_path: Path, feature_type: str, config_path: Optional[str] = None):
    """
    Trích xuất frame đầu tiên của video và vẽ khung xương landmark minh họa.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        if feature_type == "holistic":
            cfg = HolisticConfig.load_from_yaml(config_path) if config_path else HolisticConfig()
            with MediaPipeHolisticExtractor(cfg) as ext:
                _, results = ext.process_frame(frame)
                annotated = ext.draw_landmarks(frame, results)
        else:
            with SignFeatureExtractor(static_image_mode=False, model_complexity=1) as ext:
                _, results = ext.extract_frame_with_results(frame)
                annotated = ext.draw_landmarks(frame, results)

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), annotated)
    except Exception as e:
        print(f"[!] Khong the luu anh preview: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script da tien trinh (multiprocessing) trich xuat offline landmarks tap WLASL-100."
    )
    parser.add_argument(
        "--videos_dir",
        "-v",
        type=str,
        default="data/raw_videos",
        help="Thu muc chua video .mp4 (mac dinh: data/raw_videos)",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default="data/annotations/wlasl_100/splits_available.json",
        help="Duong dan file annotations splits (mac dinh: data/annotations/wlasl_100/splits_available.json)",
    )
    parser.add_argument(
        "--class_mapping",
        "-m",
        type=str,
        default="data/annotations/wlasl_100/class_mapping.json",
        help="Duong dan file class mapping (mac dinh: data/annotations/wlasl_100/class_mapping.json)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/holistic_config.yaml",
        help="Duong dan file config MediaPipe (mac dinh: configs/holistic_config.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/landmarks",
        help="Thu muc luu ket qua trich xuat (mac dinh: data/landmarks)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="both",
        choices=["both", "npy", "hdf5", "h5"],
        help="Dinh dang luu tru: 'both' (ca .npy va HDF5), 'npy' (chi .npy), 'hdf5' (chi file .h5)",
    )
    parser.add_argument(
        "--h5_filename",
        type=str,
        default="wlasl100_landmarks.h5",
        help="Ten file HDF5 tong hop khi chon luu HDF5 (mac dinh: wlasl100_landmarks.h5)",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="holistic",
        choices=["holistic", "vector201"],
        help="Loai dac trung: 'holistic' (543 diem chuan WLASL) hoac 'vector201' (201 dac trung tay+pose)",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Split can trich xuat ('all', 'train', 'val', 'test')",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=0,
        help="So tien trinh worker song song (0 la tu dong: toi uu theo so CPU cores)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Gioi han so luong video can trich xuat (0 la toan bo tap du lieu)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Bo qua video da co file trich xuat hop le (mac dinh: True)",
    )
    parser.add_argument(
        "--save_preview",
        action="store_true",
        default=True,
        help="Luu anh mau ve khung xuong landmark (mac dinh: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    annotations_path = Path(args.annotations)
    class_mapping_path = Path(args.class_mapping)
    config_path = Path(args.config) if args.config else None
    save_format = args.format.lower()

    output_dir.mkdir(parents=True, exist_ok=True)

    save_npy = save_format in ["both", "npy"]
    save_h5 = save_format in ["both", "hdf5", "h5"]
    h5_path = output_dir / args.h5_filename if save_h5 else None

    # Xác định số lượng workers tối ưu theo CPU và bộ nhớ RAM khả dụng
    total_cpus = os.cpu_count() or 4
    if args.num_workers > 0:
        num_workers = min(args.num_workers, total_cpus)
    else:
        # Tự động: Chọn 4 worker trên máy 8GB RAM để đảm bảo an toàn bộ nhớ và tối đa throughput
        num_workers = min(4, max(1, total_cpus - 2))

    print("=" * 80)
    print("      MULTIPROCESSING OFFLINE LANDMARK EXTRACTION - WLASL-100")
    print("=" * 80)
    print(f"[*] CPU Cores he thong:       {total_cpus}")
    print(f"[*] So luong tien trinh song song: {num_workers} workers")
    print(f"[*] Loai dac trung:           {args.feature_type}")
    print(f"[*] Dinh dang luu tru:        {save_format.upper()} (npy={save_npy}, hdf5={save_h5})")
    print(f"[*] Thu muc video:            {videos_dir.resolve()}")
    print(f"[*] Thu muc xuat ket qua:     {output_dir.resolve()}")
    if save_h5:
        print(f"[*] File HDF5 tong hop:       {h5_path.resolve()}")
    print(f"[*] Split can xu ly:          {args.split}")
    print(f"[*] Bo qua video da co:       {args.skip_existing}")

    # 1. Tải danh sách video mục tiêu
    targets, gloss_to_id, id_to_gloss = load_dataset_metadata(
        annotations_path=annotations_path,
        videos_dir=videos_dir,
        split_filter=args.split,
        class_mapping_path=class_mapping_path,
    )

    if not targets:
        print(f"[!] Khong tim thay video nao phu hop tai: {videos_dir}")
        return

    print(f"[*] Tong so video tim thay:   {len(targets)}")

    if args.limit > 0:
        targets = targets[:args.limit]
        print(f"[*] Gioi han so luong xu ly:  {len(targets)}")

    # Thống kê theo split
    split_counts = {}
    for t in targets:
        sp = t["split"]
        split_counts[sp] = split_counts.get(sp, 0) + 1
    print(f"[*] Phan bo theo split:       {split_counts}")
    print("-" * 80)

    # 2. Chuẩn bị HDF5 file nếu bật chế độ lưu HDF5
    h5_file = None
    existing_h5_keys = set()
    if save_h5:
        mode = "a" if (args.skip_existing and h5_path.is_file()) else "w"
        h5_file = h5py.File(str(h5_path), mode)
        
        # Tạo metadata attributes cho file HDF5
        h5_file.attrs["dataset"] = "WLASL-100"
        h5_file.attrs["feature_type"] = args.feature_type
        h5_file.attrs["created_at"] = datetime.now().isoformat()
        if gloss_to_id:
            h5_file.attrs["num_classes"] = len(gloss_to_id)
            h5_file.attrs["class_mapping_json"] = json.dumps(gloss_to_id)

        # Quét các key đã có trong HDF5 để hỗ trợ skip
        if mode == "a":
            def collect_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    existing_h5_keys.add(name)
            h5_file.visititems(collect_keys)
            print(f"[*] Tim thay {len(existing_h5_keys)} samples da ton tai trong file HDF5.")

    # 3. Chuẩn bị danh sách tasks cho worker pool
    tasks = []
    for item in targets:
        vid_id = item["video_id"]
        sp = item["split"]
        
        # Kiểm tra xem có thể skip trực tiếp hay không
        npy_exists = (output_dir / f"{vid_id}.npy").is_file()
        h5_key = f"{sp}/{vid_id}"
        h5_exists = (h5_key in existing_h5_keys) if save_h5 else True

        if args.skip_existing and (not save_npy or npy_exists) and (not save_h5 or h5_exists):
            continue  # Đã có đủ ở cả 2 định dạng

        tasks.append({
            "video_id": vid_id,
            "video_path": item["video_path"],
            "gloss": item["gloss"],
            "label": item["label"],
            "split": sp,
            "output_dir": str(output_dir),
            "save_npy": save_npy,
            "skip_existing": args.skip_existing,
            "save_preview": args.save_preview,
        })

    already_done = len(targets) - len(tasks)
    print(f"[*] So video can trich xuat moi: {len(tasks)} (da co san {already_done} video)")

    # Lưu preview demo cho 1 video mẫu nếu cần
    if args.save_preview and targets:
        preview_file = output_dir / f"sample_{args.feature_type}_preview.jpg"
        if not preview_file.is_file():
            print(f"[*] Dang tao anh minh hoa khung xuong: {preview_file.name} ...")
            sample_vid = Path(targets[0]["video_path"])
            save_preview_image(sample_vid, preview_file, args.feature_type, str(config_path) if config_path else None)

    if not tasks:
        print("[+] Tat ca video da duoc trich xuat day du! Khong can chay them.")
        if h5_file:
            h5_file.close()
        return

    # 4. Thực thi multiprocessing pool
    start_time = time.time()
    success_count = 0
    skipped_count = already_done
    error_count = 0
    total_frames = 0
    errors_log = []

    print("-" * 80)
    print(f"[*] Khoi chay {num_workers} workers song song...")

    try:
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(args.feature_type, str(config_path) if config_path else None),
            maxtasksperchild=50,
        ) as pool:
            
            pbar = tqdm(
                pool.imap_unordered(process_single_video, tasks, chunksize=1),
                total=len(tasks),
                desc="Trich xuat song song",
                unit="vid",
            )

            for res in pbar:
                vid_id = res["video_id"]
                sp = res["split"]
                status = res["status"]
                features = res.get("features")

                if status == "SUCCESS":
                    success_count += 1
                    total_frames += res["frames"]
                elif status == "SKIPPED":
                    skipped_count += 1
                    total_frames += res["frames"]
                else:
                    error_count += 1
                    errors_log.append({
                        "video_id": vid_id,
                        "error": res.get("error", "Unknown error"),
                    })

                # Ghi vào HDF5 trong luồng chính (Main process an toàn)
                if save_h5 and h5_file is not None and features is not None:
                    h5_key = f"{sp}/{vid_id}"
                    if h5_key not in h5_file:
                        try:
                            # Đảm bảo nhóm split tồn tại
                            if sp not in h5_file:
                                h5_file.create_group(sp)
                            
                            dset = h5_file.create_dataset(
                                h5_key,
                                data=features,
                                dtype="float32",
                                compression="gzip",
                                compression_opts=4,
                            )
                            dset.attrs["video_id"] = vid_id
                            dset.attrs["gloss"] = res.get("gloss", "")
                            dset.attrs["label"] = int(res.get("label", -1))
                            dset.attrs["split"] = sp
                            dset.attrs["num_frames"] = features.shape[0]
                            dset.attrs["feature_shape"] = features.shape
                        except Exception as e:
                            print(f"\n[!] Loi ghi HDF5 cho {h5_key}: {e}")

                pbar.set_postfix({
                    "OK": success_count,
                    "Skip": skipped_count,
                    "Err": error_count,
                    "Frames": total_frames,
                })

    except KeyboardInterrupt:
        print("\n[!] Nguoi dung ngat tien trinh. Dang luu du lieu an toan...")
    finally:
        if h5_file is not None:
            h5_file.flush()
            h5_file.close()
            print("[*] Da dong va flush file HDF5 thanh cong.")

    elapsed = time.time() - start_time
    fps_throughput = (total_frames / elapsed) if elapsed > 0 else 0
    vps_throughput = (success_count / elapsed) if elapsed > 0 else 0

    # 5. Xuất báo cáo tổng kết
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "total_targets": len(targets),
        "newly_extracted": success_count,
        "skipped_existing": skipped_count,
        "errors": error_count,
        "total_frames_extracted": total_frames,
        "elapsed_seconds": round(elapsed, 2),
        "video_per_sec": round(vps_throughput, 2),
        "frames_per_sec": round(fps_throughput, 2),
        "num_workers": num_workers,
        "feature_type": args.feature_type,
        "formats_saved": {
            "npy": save_npy,
            "hdf5": save_h5,
            "hdf5_file": str(h5_path) if save_h5 else None,
        },
        "errors_list": errors_log[:50],  # Lưu tối đa 50 lỗi đầu tiên
    }

    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("                     HOAN TAT TRICH XUAT OFFLINE")
    print("=" * 80)
    print(f"  * Tong so video tap WLASL-100:     {len(targets)}")
    print(f"  * So video trich xuat moi:         {success_count}")
    print(f"  * So video bo qua (da ton tai):    {skipped_count}")
    print(f"  * So video gap loi:                {error_count}")
    print(f"  * Tong so frames dac trung:        {total_frames:,}")
    print(f"  * Thoi gian thuc hien:             {elapsed:.1f} giay ({elapsed / 60:.2f} phut)")
    print(f"  * Toc do xu ly trung binh:         {vps_throughput:.2f} videos/s | {fps_throughput:.1f} frames/s")
    print(f"  * Thu muc landmarks .npy:          {output_dir.resolve()}")
    if save_h5:
        h5_size_mb = h5_path.stat().st_size / (1024 * 1024) if h5_path.is_file() else 0
        print(f"  * File HDF5 tong hop:              {h5_path.resolve()} ({h5_size_mb:.2f} MB)")
    print(f"  * Bao cao tom tat luu tai:         {summary_file.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
