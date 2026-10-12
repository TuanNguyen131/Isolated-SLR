# Cấu Trúc Thư Mục Dự Án — Isolated Sign Language Recognition

> **Repository:** `TuanNguyen131/Isolated-SLR`  
> **Ngày cập nhật:** 12/10/2026

---

```
Isolated-SLR/
│
├── configs/                              # Cấu hình huấn luyện & pipeline
│   └── holistic_config.yaml              #   Tham số MediaPipe, augmentation, resampling
│
├── data/                                 # Dữ liệu (không đẩy lên Git)
│   ├── annotations/                      #   Nhãn & phân chia tập dữ liệu
│   │   ├── class_mapping.json            #     Ánh xạ nhãn 0–99
│   │   ├── WLASL_v0.3.json              #     Bộ annotation gốc WLASL v0.3
│   │   ├── wlasl100.json                #     Annotation đã lọc cho 100 lớp
│   │   ├── train.json                   #     Phân chia Train (633 clips)
│   │   ├── val.json                     #     Phân chia Validation (155 clips)
│   │   ├── test.json                    #     Phân chia Test (144 clips)
│   │   ├── missing_videos.json          #     Danh sách video lỗi/mất
│   │   └── missing_videos.csv           #     CSV log video không tải được
│   │
│   ├── raw_videos/                       #   932 video .mp4 gốc đã tải
│   │   ├── 00623.mp4
│   │   ├── 00624.mp4
│   │   └── ... (932 files)
│   │
│   ├── landmarks/                        #   Landmarks thô (.npy + HDF5)
│   │   ├── <video_id>.npy               #     Mỗi video → 1 file numpy (T, 543, 4)
│   │   ├── wlasl100_landmarks.h5        #     HDF5 tổng hợp (381.3 MB, Gzip L4)
│   │   └── extraction_summary.json      #     Thống kê quá trình trích xuất
│   │
│   ├── landmarks_interpolated/           #   Landmarks sau nội suy
│   ├── processed_landmarks/              #   Landmarks đã xử lý (201D)
│   ├── processed_landmarks_interpolated/ #   201D + nội suy
│   ├── processed_landmarks_normalized/   #   201D + nội suy + chuẩn hóa
│   │
│   ├── sample_interpolation_comparison.png    # Hình minh họa nội suy
│   ├── sample_normalization_comparison.png    # Hình minh họa chuẩn hóa
│   ├── sample_resampling_comparison.png       # Hình minh họa resampling
│   ├── sample_resampling_upsample.png         # Hình minh họa upsampling
│   └── sample_augmentation_comparison.png     # Hình minh họa augmentation
│
├── scripts/                              # Scripts thực thi pipeline
│   ├── extract_wlasl100.py               #   Lọc WLASL-100 từ bộ gốc v0.3
│   ├── download_wlasl100.py              #   Tải video đa luồng (HTTP + yt-dlp)
│   ├── verify_wlasl100.py                #   Kiểm tra tính toàn vẹn video
│   ├── extract_landmarks.py              #   Trích xuất landmarks (đơn tiến trình)
│   ├── extract_multiprocess.py           #   Trích xuất landmarks (đa tiến trình, 4 workers)
│   ├── process_missing_hands.py          #   Xử lý bàn tay mất dấu
│   ├── visualize_interpolation.py        #   Trực quan hóa nội suy
│   ├── visualize_normalization.py        #   Trực quan hóa chuẩn hóa
│   ├── visualize_resampling.py           #   Trực quan hóa resampling
│   └── visualize_augmentation.py         #   Trực quan hóa augmentation
│
├── src/                                  # Mã nguồn chính (Python package)
│   ├── __init__.py
│   │
│   ├── data/                             #   Module tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── feature_extractor.py          #     Trích xuất 201D từ 543 landmarks gốc
│   │   ├── interpolation.py              #     Nội suy tuyến tính & Zero-Padding
│   │   ├── normalization.py              #     Chuẩn hóa 2 tầng (Vai + Cổ tay)
│   │   └── augmentation.py              #     Tăng cường dữ liệu tọa độ (5 phép biến đổi)
│   │
│   ├── utils/                            #   Tiện ích hỗ trợ
│   │   ├── __init__.py
│   │   ├── filter_wlasl.py               #     Hàm lọc & phân chia WLASL
│   │   ├── holistic_config.py            #     Đọc cấu hình từ YAML
│   │   └── mediapipe_holistic.py         #     Wrapper MediaPipe Holistic
│   │
│   ├── dataset.py                        #   WLASLDataset & create_dataloader (PyTorch)
│   ├── models.py                         #   Kiến trúc mô hình (BiLSTM, GRU, Transformer)
│   └── train.py                          #   Script huấn luyện & đánh giá
│
├── tests/                                # Bộ kiểm thử tự động (42 test cases)
│   ├── test_interpolation.py             #   9 tests — Nội suy & Zero-Padding
│   ├── test_normalization.py             #   8 tests — Chuẩn hóa 2 tầng
│   ├── test_temporal_resampling.py       #  14 tests — Resampling T=30
│   └── test_augmentation.py              #  11 tests — Augmentation & Zero-Mask
│
├── notebooks/                            # Jupyter notebooks (thí nghiệm)
│   └── .gitkeep
│
├── GT_report.html                        # Báo cáo giải trình kỹ thuật (HTML source)
├── GT.pdf                                # Báo cáo giải trình kỹ thuật (PDF 5 trang)
├── CT.md                                 # Cấu trúc thư mục dự án (file này)
├── requirements.txt                      # Danh sách thư viện Python
├── .gitignore                            # Quy tắc bỏ qua file cho Git
└── README.md                             # Giới thiệu dự án
```

---

## Mô Tả Các Thành Phần Chính

| Thư mục / File | Vai trò |
|---|---|
| `configs/` | Tham số cấu hình tập trung (YAML) cho toàn bộ pipeline |
| `data/annotations/` | Nhãn, ánh xạ lớp, phân chia train/val/test |
| `data/raw_videos/` | 932 video .mp4 gốc từ WLASL-100 |
| `data/landmarks/` | Landmarks thô trích xuất từ MediaPipe (`.npy` + `.h5`) |
| `scripts/` | Các script chạy 1 lần: tải, trích xuất, trực quan hóa |
| `src/data/` | **Core pipeline**: feature extraction → interpolation → normalization → augmentation |
| `src/dataset.py` | Tích hợp toàn bộ pipeline vào `WLASLDataset` (PyTorch) |
| `src/models.py` | Định nghĩa kiến trúc mạng neural |
| `src/train.py` | Vòng lặp huấn luyện, đánh giá, logging |
| `tests/` | 42 unit tests kiểm thử toàn bộ pipeline |

---

## Luồng Dữ Liệu (Data Flow)

```
Video .mp4 → MediaPipe Holistic (543 pts) → Feature Extractor (201D)
    → Interpolation (bù mất dấu) → Normalization (2 tầng)
    → Temporal Resampling (T=30) → Augmentation (train only)
    → PyTorch Tensor (Batch, 30, 201) → Model → Prediction
```
