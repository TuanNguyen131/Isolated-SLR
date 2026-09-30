"""
Script lọc và trích xuất danh sách 100 lớp từ vựng (WLASL-100) từ WLASL_v0.3.json
phân chia theo các trường split (train, val, test).

Hỗ trợ lưu ra nhiều định dạng:
1. WLASL_100.json: Cấu trúc gốc WLASL (list các gloss và instances)
2. splits.json: Cấu trúc nhóm theo split (metadata, train, val, test)
3. train.json, val.json, test.json: Từng file riêng biệt cho từng split
4. train.txt, val.txt, test.txt: Danh sách định dạng 'video_id gloss label_id'
5. classes.txt & class_mapping.json: Danh sách nhãn từ vựng và ánh xạ id <-> gloss
6. wlasl_100.csv: Bảng dữ liệu tổng hợp chi tiết cho 2038 instances
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# Đảm bảo UTF-8 hoạt động chuẩn trên Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lọc và trích xuất tập dữ liệu WLASL-100 theo train/val/test split."
    )
    parser.add_argument(
        "--input_json",
        "-i",
        type=str,
        default="data/annotations/WLASL_v0.3.json",
        help="Đường dẫn đến file annotation WLASL_v0.3.json gốc (mặc định: data/annotations/WLASL_v0.3.json)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/annotations/wlasl_100",
        help="Thư mục lưu các file trích xuất (mặc định: data/annotations/wlasl_100)",
    )
    parser.add_argument(
        "--num_classes",
        "-k",
        type=int,
        default=100,
        help="Số lượng lớp từ vựng cần trích xuất (mặc định: 100 cho WLASL-100)",
    )
    return parser.parse_args()


def extract_wlasl_subset(input_path: str, num_classes: int = 100):
    """
    Đọc file annotation gốc WLASL và trích xuất top num_classes.
    
    Trong WLASL benchmark, 100 lớp WLASL-100 là 100 từ vựng đầu tiên
    trong WLASL_v0.3.json (các từ có nhiều mẫu nhất).
    """
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file annotation tại: {input_file.resolve()}")

    print(f"[*] Đang đọc dữ liệu từ: {input_file.resolve()} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total_glosses = len(raw_data)
    print(f"[*] Tổng số từ vựng trong file gốc: {total_glosses}")

    if num_classes > total_glosses:
        raise ValueError(
            f"num_classes ({num_classes}) lớn hơn tổng số từ vựng trong file ({total_glosses})"
        )

    # Lấy top num_classes đầu tiên
    subset_raw = raw_data[:num_classes]

    # Xây dựng từ điển ánh xạ nhãn
    gloss_list = [item["gloss"] for item in subset_raw]
    gloss_to_id = {gloss: idx for idx, gloss in enumerate(gloss_list)}
    id_to_gloss = {idx: gloss for idx, gloss in enumerate(gloss_list)}

    # Phân loại instances theo split
    splits_data = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Danh sách phẳng chứa tất cả instances cùng label_id
    all_instances = []

    for entry in subset_raw:
        gloss = entry["gloss"]
        label_id = gloss_to_id[gloss]
        for inst in entry["instances"]:
            split_name = inst.get("split")
            record = {
                "video_id": inst.get("video_id"),
                "gloss": gloss,
                "label": label_id,
                "split": split_name,
                "fps": inst.get("fps"),
                "frame_start": inst.get("frame_start"),
                "frame_end": inst.get("frame_end"),
                "bbox": inst.get("bbox"),
                "signer_id": inst.get("signer_id"),
                "source": inst.get("source"),
                "url": inst.get("url"),
                "instance_id": inst.get("instance_id"),
                "variation_id": inst.get("variation_id"),
            }
            if split_name in splits_data:
                splits_data[split_name].append(record)
            else:
                print(f"[!] Cảnh báo: Gặp split không xác định '{split_name}' ở video_id {inst.get('video_id')}")
            all_instances.append(record)

    return {
        "num_classes": num_classes,
        "gloss_list": gloss_list,
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": id_to_gloss,
        "subset_raw": subset_raw,
        "splits_data": splits_data,
        "all_instances": all_instances,
    }


def save_extracted_data(data: dict, output_dir_path: str):
    """
    Lưu dữ liệu trích xuất ra các định dạng chuẩn phục vụ huấn luyện và nghiên cứu.
    """
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # 1. WLASL_100.json: Cấu trúc gốc tương thích với mã nguồn WLASL chuẩn
    wlasl_raw_path = output_dir / f"WLASL_{data['num_classes']}.json"
    with open(wlasl_raw_path, "w", encoding="utf-8") as f:
        json.dump(data["subset_raw"], f, indent=4, ensure_ascii=False)
    created_files.append(wlasl_raw_path)

    # 2. splits.json: Tổng hợp train, val, test cùng metadata
    splits_combined_path = output_dir / "splits.json"
    splits_combined_content = {
        "metadata": {
            "num_classes": data["num_classes"],
            "total_instances": len(data["all_instances"]),
            "train_instances": len(data["splits_data"]["train"]),
            "val_instances": len(data["splits_data"]["val"]),
            "test_instances": len(data["splits_data"]["test"]),
        },
        "train": data["splits_data"]["train"],
        "val": data["splits_data"]["val"],
        "test": data["splits_data"]["test"],
    }
    with open(splits_combined_path, "w", encoding="utf-8") as f:
        json.dump(splits_combined_content, f, indent=2, ensure_ascii=False)
    created_files.append(splits_combined_path)

    # 3. train.json, val.json, test.json: Từng file riêng lẻ cho Dataset DataLoader
    for split_name in ["train", "val", "test"]:
        split_file = output_dir / f"{split_name}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(data["splits_data"][split_name], f, indent=2, ensure_ascii=False)
        created_files.append(split_file)

    # 4. train.txt, val.txt, test.txt: Định dạng văn bản đơn giản (video_id gloss label)
    for split_name in ["train", "val", "test"]:
        txt_file = output_dir / f"{split_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            for item in data["splits_data"][split_name]:
                f.write(f"{item['video_id']} {item['gloss']} {item['label']}\n")
        created_files.append(txt_file)

    # 5. classes.txt & class_mapping.json
    classes_txt_path = output_dir / "classes.txt"
    with open(classes_txt_path, "w", encoding="utf-8") as f:
        for idx, gloss in enumerate(data["gloss_list"]):
            f.write(f"{idx} {gloss}\n")
    created_files.append(classes_txt_path)

    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gloss_to_id": data["gloss_to_id"],
                "id_to_gloss": data["id_to_gloss"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    created_files.append(mapping_path)

    # 6. wlasl_100.csv: Bảng dữ liệu tổng hợp
    csv_path = output_dir / f"wlasl_{data['num_classes']}.csv"
    csv_columns = [
        "video_id",
        "gloss",
        "label",
        "split",
        "fps",
        "frame_start",
        "frame_end",
        "signer_id",
        "source",
        "url",
        "bbox",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for row in data["all_instances"]:
            row_copy = dict(row)
            if isinstance(row_copy.get("bbox"), list):
                row_copy["bbox"] = str(row_copy["bbox"])
            writer.writerow(row_copy)
    created_files.append(csv_path)

    return created_files


def print_summary_report(data: dict, created_files: list):
    """
    In báo cáo thống kê trực quan ra màn hình.
    """
    num_classes = data["num_classes"]
    train_count = len(data["splits_data"]["train"])
    val_count = len(data["splits_data"]["val"])
    test_count = len(data["splits_data"]["test"])
    total_count = len(data["all_instances"])

    train_classes = set(item["gloss"] for item in data["splits_data"]["train"])
    val_classes = set(item["gloss"] for item in data["splits_data"]["val"])
    test_classes = set(item["gloss"] for item in data["splits_data"]["test"])

    samples_per_class = [len(entry["instances"]) for entry in data["subset_raw"]]
    min_samples = min(samples_per_class)
    max_samples = max(samples_per_class)
    avg_samples = sum(samples_per_class) / len(samples_per_class)

    sep = "=" * 65
    line = "-" * 65
    print("\n" + sep)
    print(f"            KET QUA TRICH XUAT WLASL-{num_classes}")
    print(sep)
    print(f"  * So luong lop tu vung (Classes):    {num_classes}")
    print(f"  * Tong so video/instances:          {total_count}")
    print(f"  * So mau train:                     {train_count} ({train_count/total_count*100:.1f}%)")
    print(f"  * So mau val:                       {val_count} ({val_count/total_count*100:.1f}%)")
    print(f"  * So mau test:                      {test_count} ({test_count/total_count*100:.1f}%)")
    print(line)
    print("  * Do phu tu vung theo Split:")
    print(f"    - Train split co: {len(train_classes)}/{num_classes} lop")
    print(f"    - Val split co:   {len(val_classes)}/{num_classes} lop")
    print(f"    - Test split co:  {len(test_classes)}/{num_classes} lop")
    print(line)
    print(f"  * Thong ke so mau moi lop: Min = {min_samples} | Max = {max_samples} | TB = {avg_samples:.2f}")
    print(f"  * 10 tu vung dau tien:")
    print(f"    {', '.join(data['gloss_list'][:10])}")
    print(sep)
    print("  DANH SACH FILE DA TAO:")
    for file_path in created_files:
        size_kb = file_path.stat().st_size / 1024
        print(f"  [+] {file_path.name:<25} ({size_kb:>8.2f} KB) -> {file_path}")
    print(sep + "\n")


def main():
    args = parse_args()
    data = extract_wlasl_subset(
        input_path=args.input_json,
        num_classes=args.num_classes,
    )
    created_files = save_extracted_data(
        data=data,
        output_dir_path=args.output_dir,
    )
    print_summary_report(data, created_files)


if __name__ == "__main__":
    main()
