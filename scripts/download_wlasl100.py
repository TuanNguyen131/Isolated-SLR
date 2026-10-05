"""
Script tải và crawl video clips cho tập dữ liệu WLASL-100.
Xử lý toàn diện các trường hợp:
- Download trực tiếp từ các nguồn HTTP/HTTPS (S3, Azure Blob, SignBank, Deafined, SpreadTheSign, etc.)
- Download video YouTube thông qua yt-dlp (chọn stream video mp4 tối ưu, không phụ thuộc ffmpeg ngoài)
- Bộ nhớ đệm video YouTube (caching) giúp tái sử dụng video gốc khi nhiều instance chia sẻ cùng 1 URL
- Tự động cắt clip theo frame_start và frame_end (nếu có yêu cầu từ WLASL annotation)
- Xử lý link lỗi (404, 403, SSL mismatch, domain die, video YouTube bị xóa/private)
- Kiểm tra tính hợp lệ của video ngay sau khi tải bằng OpenCV
- Ghi nhận chi tiết danh sách missing_videos (JSON + CSV) phục vụ audit và huấn luyện
- Hỗ trợ đa luồng (Multi-threading) và Resume (bỏ qua video đã tải hợp lệ)
"""

import os
import sys
import json
import csv
import time
import argparse
import urllib3
import requests
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List

import cv2
import yt_dlp

# Tắt cảnh báo SSL InsecureRequestWarning cho các server có chứng chỉ cũ/mismatch
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Đảm bảo UTF-8 hoạt động chuẩn trên Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "video/webm,video/ogg,video/*;q=0.9,application/octet-stream;q=0.8,*/*;q=0.5",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tải và crawl video WLASL-100.")
    parser.add_argument(
        "--annotations",
        "-a",
        type=str,
        default="data/annotations/wlasl_100/WLASL_100.json",
        help="Đường dẫn file annotation WLASL-100 (mặc định: data/annotations/wlasl_100/WLASL_100.json)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/raw_videos",
        help="Thư mục lưu trữ video thô (mặc định: data/raw_videos)",
    )
    parser.add_argument(
        "--max_workers",
        "-w",
        type=int,
        default=12,
        help="Số luồng tải đồng thời cho direct HTTP downloads (mặc định: 12)",
    )
    parser.add_argument(
        "--yt_workers",
        type=int,
        default=4,
        help="Số luồng tải đồng thời cho YouTube (mặc định: 4)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=10,
        help="Timeout (giây) cho mỗi kết nối HTTP (mặc định: 10)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Giới hạn số video cần tải để test (0 là tải toàn bộ)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Bỏ qua các video đã tồn tại và hợp lệ (mặc định: True)",
    )
    parser.add_argument(
        "--missing_out",
        type=str,
        default="data/annotations/wlasl_100/missing_videos.json",
        help="File lưu báo cáo các video lỗi/missing (JSON)",
    )
    return parser.parse_args()


def verify_video_file(file_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Kiểm tra tính toàn vẹn của file video bằng OpenCV.
    Trả về (is_valid, reason, info_dict).
    """
    if not file_path.is_file():
        return False, "FILE_NOT_FOUND", {}

    file_size = file_path.stat().st_size
    if file_size < 1024:  # Nhỏ hơn 1KB chắc chắn là lỗi hoặc 0-byte
        return False, f"FILE_TOO_SMALL_{file_size}B", {}

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        cap.release()
        return False, "CANNOT_OPEN_CONTAINER", {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Thử đọc frame đầu tiên để xác nhận giải mã thành công
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, "CANNOT_DECODE_FIRST_FRAME", {}

    if frame_count <= 0 or width <= 0 or height <= 0:
        return False, "INVALID_STREAM_METADATA", {}

    info = {
        "frames": frame_count,
        "fps": round(fps, 2) if fps else 25.0,
        "width": width,
        "height": height,
        "size_bytes": file_size,
    }
    return True, "OK", info


def trim_video(src_path: Path, dst_path: Path, frame_start: int, frame_end: int) -> bool:
    """
    Cắt video theo frame_start và frame_end sử dụng OpenCV.
    frame_start và frame_end tính theo 1-based index (chuẩn WLASL).
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        cap.release()
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Chuẩn bị writer
    tmp_trimmed = dst_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_trimmed), fourcc, fps, (width, height))

    current_idx = 1
    extracted_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_start <= current_idx <= frame_end:
            out.write(frame)
            extracted_frames += 1

        if current_idx > frame_end:
            break

        current_idx += 1

    cap.release()
    out.release()

    if extracted_frames > 0:
        if tmp_trimmed.exists():
            if dst_path.exists():
                dst_path.unlink()
            tmp_trimmed.rename(dst_path)
            return True
    else:
        if tmp_trimmed.exists():
            tmp_trimmed.unlink()
        return False


def get_youtube_id(url: str) -> str:
    """Trích xuất ID 11 ký tự của YouTube từ URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else url[-11:]


def download_http_video(inst: Dict[str, Any], output_path: Path, timeout: int = 10) -> Tuple[bool, str]:
    """
    Tải video trực tiếp từ HTTP/HTTPS.
    Xử lý headers, SSL, streaming và kiểm tra định dạng.
    """
    url = inst["url"]
    headers = dict(DEFAULT_HEADERS)
    source = inst.get("source", "")

    if "signingsavvy" in source or "signingsavvy" in url:
        headers["Referer"] = "https://www.signingsavvy.com/"
    elif "aslpro" in source:
        headers["Referer"] = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"
    elif "handspeak" in source:
        headers["Referer"] = "https://www.handspeak.com/"

    tmp_path = output_path.with_suffix(".part")

    try:
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=timeout, stream=True, verify=False)

        if resp.status_code != 200:
            return False, f"HTTP_STATUS_{resp.status_code}"

        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            return False, "INVALID_CONTENT_TYPE_HTML"

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        # Kiểm tra tính toàn vẹn file vừa tải
        is_valid, reason, _ = verify_video_file(tmp_path)
        if not is_valid:
            if tmp_path.exists():
                tmp_path.unlink()
            return False, f"DOWNLOADED_CORRUPTED_{reason}"

        # Kiểm tra cắt khung hình nếu cần
        f_start = inst.get("frame_start", 1)
        f_end = inst.get("frame_end", -1)
        if f_end > 0 and f_end >= f_start:
            trimmed = trim_video(tmp_path, output_path, f_start, f_end)
            if tmp_path.exists():
                tmp_path.unlink()
            if not trimmed:
                return False, "TRIM_FAILED"
        else:
            if output_path.exists():
                output_path.unlink()
            tmp_path.rename(output_path)

        return True, "SUCCESS"

    except requests.exceptions.Timeout:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, "CONNECTION_TIMEOUT"
    except requests.exceptions.SSLError as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"SSL_ERROR_{str(e)[:40]}"
    except requests.exceptions.ConnectionError:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, "CONNECTION_REFUSED_OR_DNS_FAIL"
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"EXCEPTION_{type(e).__name__}_{str(e)[:30]}"


def download_youtube_video(inst: Dict[str, Any], output_path: Path, yt_cache_dir: Path) -> Tuple[bool, str]:
    """
    Tải video YouTube sử dụng yt-dlp với bộ nhớ đệm theo video_id của YouTube.
    Tránh tải lại cùng một URL YouTube khi có nhiều instance sử dụng.
    """
    url = inst["url"]
    yt_id = get_youtube_id(url)
    f_start = inst.get("frame_start", 1)
    f_end = inst.get("frame_end", -1)

    cached_yt_path = yt_cache_dir / f"{yt_id}.mp4"

    # 1. Kiểm tra xem video YouTube gốc đã có trong cache chưa
    need_download = True
    if cached_yt_path.exists():
        is_valid, _, _ = verify_video_file(cached_yt_path)
        if is_valid:
            need_download = False
        else:
            cached_yt_path.unlink()

    # 2. Nếu chưa có trong cache -> tải về cache
    if need_download:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
            "outtmpl": str(cached_yt_path),
            "noplaylist": True,
            "socket_timeout": 12,
            "retries": 1,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as e:
            msg = str(e).lower()
            if "video unavailable" in msg:
                return False, "YT_VIDEO_UNAVAILABLE"
            elif "private video" in msg:
                return False, "YT_VIDEO_PRIVATE"
            elif "terminated" in msg:
                return False, "YT_ACCOUNT_TERMINATED"
            elif "bot" in msg or "sign in" in msg:
                return False, "YT_BOT_CHALLENGE"
            else:
                return False, f"YT_DOWNLOAD_ERROR_{str(e)[:45]}"
        except Exception as e:
            return False, f"YT_EXCEPTION_{type(e).__name__}_{str(e)[:30]}"

        if not cached_yt_path.exists():
            return False, "YT_FILE_NOT_SAVED"

        is_valid, reason, _ = verify_video_file(cached_yt_path)
        if not is_valid:
            if cached_yt_path.exists():
                cached_yt_path.unlink()
            return False, f"YT_CORRUPTED_{reason}"

    # 3. Trích xuất hoặc sao chép vào output_path
    if f_end > 0 and f_end >= f_start:
        trimmed = trim_video(cached_yt_path, output_path, f_start, f_end)
        if not trimmed:
            return False, "YT_TRIM_FAILED"
    else:
        import shutil
        shutil.copyfile(cached_yt_path, output_path)

    return True, "SUCCESS"


def process_instance(
    inst: Dict[str, Any], output_dir: Path, yt_cache_dir: Path, skip_existing: bool, timeout: int
) -> Dict[str, Any]:
    """
    Xử lý tải một instance (kiểm tra tồn tại, download, kiểm tra hợp lệ, ghi nhận kết quả).
    """
    video_id = inst["video_id"]
    gloss = inst.get("gloss", "")
    split = inst.get("split", "")
    source = inst.get("source", "")
    url = inst.get("url", "")
    dst_path = output_dir / f"{video_id}.mp4"

    # Kiểm tra nếu file đã tồn tại và hợp lệ
    if skip_existing and dst_path.exists():
        is_valid, reason, info = verify_video_file(dst_path)
        if is_valid:
            return {
                "video_id": video_id,
                "gloss": gloss,
                "split": split,
                "source": source,
                "url": url,
                "status": "EXISTING_VALID",
                "reason": "ALREADY_DOWNLOADED",
                "info": info,
            }
        else:
            # File hỏng từ lần tải trước -> xóa và tải lại
            dst_path.unlink()

    is_yt = "youtube" in url or "youtu.be" in url

    if is_yt:
        success, reason = download_youtube_video(inst, dst_path, yt_cache_dir)
    else:
        success, reason = download_http_video(inst, dst_path, timeout=timeout)

    if success:
        is_valid, v_reason, info = verify_video_file(dst_path)
        if is_valid:
            return {
                "video_id": video_id,
                "gloss": gloss,
                "split": split,
                "source": source,
                "url": url,
                "status": "SUCCESS",
                "reason": "OK",
                "info": info,
            }
        else:
            if dst_path.exists():
                dst_path.unlink()
            return {
                "video_id": video_id,
                "gloss": gloss,
                "split": split,
                "source": source,
                "url": url,
                "status": "FAILED",
                "reason": f"POST_VERIFY_FAILED_{v_reason}",
                "info": {},
            }
    else:
        return {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "source": source,
            "url": url,
            "status": "FAILED",
            "reason": reason,
            "info": {},
        }


def load_instances_from_annotations(annotation_path: Path) -> List[Dict[str, Any]]:
    """
    Đọc tất cả instances từ file annotation WLASL_100.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []
    if isinstance(data, list):
        for entry in data:
            gloss = entry.get("gloss", "")
            for inst in entry.get("instances", []):
                inst_copy = dict(inst)
                inst_copy["gloss"] = gloss
                instances.append(inst_copy)
    elif isinstance(data, dict) and "train" in data:
        # File splits.json
        for split_name in ["train", "val", "test"]:
            for inst in data.get(split_name, []):
                instances.append(inst)
    return instances


def main():
    args = parse_args()
    ann_path = Path(args.annotations)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yt_cache_dir = out_dir / ".yt_cache"
    yt_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("           WLASL-100 ROBUST VIDEO CRAWLER & DOWNLOADER")
    print("=" * 70)
    print(f"[*] Annotation:        {ann_path.resolve()}")
    print(f"[*] Thư mục lưu video: {out_dir.resolve()}")
    print(f"[*] Direct Workers:    {args.max_workers} | YouTube Workers: {args.yt_workers}")
    print(f"[*] Skip Existing:     {args.skip_existing}")

    all_instances = load_instances_from_annotations(ann_path)
    total_total = len(all_instances)
    print(f"[*] Tổng số instances tìm thấy: {total_total}")

    if args.limit > 0:
        all_instances = all_instances[:args.limit]
        print(f"[*] Giới hạn kiểm tra / tải: {len(all_instances)} instances")

    # Phân loại instances theo direct HTTP và YouTube để chạy luồng tối ưu
    direct_instances = [
        inst for inst in all_instances
        if "youtube" not in inst.get("url", "") and "youtu.be" not in inst.get("url", "")
    ]
    yt_instances = [
        inst for inst in all_instances
        if "youtube" in inst.get("url", "") or "youtu.be" in inst.get("url", "")
    ]

    print(f"[*] Direct HTTP sources: {len(direct_instances)} video")
    print(f"[*] YouTube sources:     {len(yt_instances)} video")
    print("-" * 70)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    # 1. Chạy tải Direct HTTP sources với ThreadPoolExecutor
    print(f"\n[PHASE 1] Tải các nguồn Direct HTTP ({len(direct_instances)} items, {args.max_workers} workers)...")
    direct_success = 0
    direct_failed = 0
    direct_skipped = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_instance, inst, out_dir, yt_cache_dir, args.skip_existing, args.timeout): inst
            for inst in direct_instances
        }
        done_cnt = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            done_cnt += 1
            st = res["status"]
            if st == "SUCCESS":
                direct_success += 1
                prefix = "[+]"
            elif st == "EXISTING_VALID":
                direct_skipped += 1
                prefix = "[=]"
            else:
                direct_failed += 1
                prefix = "[-]"

            if done_cnt % 50 == 0 or done_cnt == len(direct_instances):
                elapsed = time.time() - start_time
                print(
                    f"  {prefix} [{done_cnt}/{len(direct_instances)}] "
                    f"Hợp lệ: {direct_success + direct_skipped} | Lỗi: {direct_failed} "
                    f"({elapsed:.1f}s)"
                )

    # 2. Chạy tải YouTube sources với yt_workers
    print(f"\n[PHASE 2] Tải các nguồn YouTube ({len(yt_instances)} items, {args.yt_workers} workers)...")
    yt_success = 0
    yt_failed = 0
    yt_skipped = 0

    with ThreadPoolExecutor(max_workers=args.yt_workers) as executor:
        futures = {
            executor.submit(process_instance, inst, out_dir, yt_cache_dir, args.skip_existing, args.timeout): inst
            for inst in yt_instances
        }
        done_cnt = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            done_cnt += 1
            st = res["status"]
            if st == "SUCCESS":
                yt_success += 1
                prefix = "[+]"
            elif st == "EXISTING_VALID":
                yt_skipped += 1
                prefix = "[=]"
            else:
                yt_failed += 1
                prefix = "[-]"

            if done_cnt % 25 == 0 or done_cnt == len(yt_instances):
                elapsed = time.time() - start_time
                print(
                    f"  {prefix} [{done_cnt}/{len(yt_instances)}] "
                    f"Hợp lệ: {yt_success + yt_skipped} | Lỗi: {yt_failed} "
                    f"({elapsed:.1f}s)"
                )

    # Phân tích tổng kết kết quả
    total_valid = sum(1 for r in results if r["status"] in ["SUCCESS", "EXISTING_VALID"])
    failed_items = [r for r in results if r["status"] == "FAILED"]
    total_failed = len(failed_items)

    # Thống kê nguyên nhân lỗi
    error_reasons: Dict[str, int] = {}
    for item in failed_items:
        r = item["reason"]
        error_reasons[r] = error_reasons.get(r, 0) + 1

    # Lưu danh sách missing videos (JSON + CSV)
    missing_json_path = Path(args.missing_out)
    missing_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(missing_json_path, "w", encoding="utf-8") as f:
        json.dump(failed_items, f, indent=4, ensure_ascii=False)

    missing_csv_path = missing_json_path.with_suffix(".csv")
    with open(missing_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "gloss", "split", "source", "url", "reason"])
        for item in failed_items:
            writer.writerow([
                item["video_id"],
                item["gloss"],
                item["split"],
                item["source"],
                item["url"],
                item["reason"],
            ])

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("                    TONG KET CRAWL & DOWNLOAD")
    print("=" * 70)
    print(f"  * Tong thoi gian thuc hien:       {elapsed:.1f}s")
    print(f"  * Tong so instances xu ly:        {len(results)}")
    print(f"  * So video hop le (San sang):     {total_valid} ({total_valid/len(results)*100:.1f}%)")
    print(f"  * So video loi / missing:         {total_failed} ({total_failed/len(results)*100:.1f}%)")
    print("-" * 70)
    print("  * Top nguyen nhan loi / missing:")
    for reason, count in sorted(error_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {reason:<40}: {count} video")
    print("-" * 70)
    print(f"  * File missing JSON: {missing_json_path}")
    print(f"  * File missing CSV:  {missing_csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
