"""
Module cung cấp các hàm tiện ích để lọc và tải dữ liệu WLASL (WLASL-100, WLASL-300, v.v.).
Có thể sử dụng độc lập hoặc import vào Dataset / DataLoader trong src/dataset.py.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def extract_wlasl_subset(input_path: Union[str, Path], num_classes: int = 100) -> dict:
    """
    Trích xuất top num_classes từ file annotation WLASL_v0.3.json gốc.
    
    Args:
        input_path: Đường dẫn tới file WLASL_v0.3.json.
        num_classes: Số lượng lớp cần trích xuất (mặc định 100).

    Returns:
        Dictionary chứa danh sách nhãn, ánh xạ nhãn và các instance phân chia theo split.
    """
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file annotation tại: {input_file.resolve()}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if num_classes > len(raw_data):
        raise ValueError(
            f"num_classes ({num_classes}) vượt quá tổng số từ vựng ({len(raw_data)})"
        )

    subset_raw = raw_data[:num_classes]
    gloss_list = [item["gloss"] for item in subset_raw]
    gloss_to_id = {gloss: idx for idx, gloss in enumerate(gloss_list)}
    id_to_gloss = {idx: gloss for idx, gloss in enumerate(gloss_list)}

    splits_data = {"train": [], "val": [], "test": []}
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


def load_wlasl_split(
    annotation_source: Union[str, Path],
    split: str = "train",
    num_classes: int = 100,
    only_available: bool = False,
    videos_dir: Optional[Union[str, Path]] = "data/raw_videos",
) -> List[dict]:
    """
    Tải danh sách các mẫu thuộc một split cụ thể ('train', 'val', 'test').
    Hỗ trợ cả việc truyền vào:
    - File split riêng (e.g. data/annotations/wlasl_100/train.json)
    - File splits tổng hợp (e.g. data/annotations/wlasl_100/splits.json hoặc splits_available.json)
    - File WLASL gốc (e.g. data/annotations/WLASL_v0.3.json) kết hợp num_classes
    - Lọc chỉ các video thực sự tồn tại nếu only_available=True
    """
    source_path = Path(annotation_source)

    if not source_path.exists():
        raise FileNotFoundError(f"Đường dẫn không tồn tại: {source_path}")

    records = []

    # Nếu truyền vào thư mục chứa splits
    if source_path.is_dir():
        # Kiểm tra ưu tiên splits_available.json nếu only_available=True
        if only_available:
            avail_file = source_path / "splits_available.json"
            if avail_file.is_file():
                with open(avail_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records = data.get(split, [])
        if not records:
            split_file = source_path / f"{split}.json"
            if split_file.is_file():
                with open(split_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

    # Nếu truyền vào file JSON
    elif source_path.is_file():
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Trường hợp 1: file train.json / val.json / test.json trực tiếp (danh sách bản ghi)
        if isinstance(data, list) and len(data) > 0 and "split" in data[0] and "video_id" in data[0]:
            records = [item for item in data if item.get("split") == split]

        # Trường hợp 2: file splits.json có keys 'train', 'val', 'test'
        elif isinstance(data, dict) and split in data:
            records = data[split]

        # Trường hợp 3: file WLASL_v0.3.json hoặc WLASL_100.json cấu trúc gốc
        else:
            subset = extract_wlasl_subset(source_path, num_classes=num_classes)
            records = subset["splits_data"].get(split, [])
    else:
        raise ValueError(f"Không nhận diện được định dạng annotation tại: {source_path}")

    # Lọc chỉ giữ lại video thực tế có sẵn trên đĩa nếu được yêu cầu
    if only_available and videos_dir:
        v_dir = Path(videos_dir)
        filtered = []
        for r in records:
            vid_id = r.get("video_id")
            if vid_id:
                v_path = v_dir / f"{vid_id}.mp4"
                if v_path.is_file() and v_path.stat().st_size > 1024:
                    filtered.append(r)
        return filtered

    return records

