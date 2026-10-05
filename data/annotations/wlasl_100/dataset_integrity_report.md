# Báo Cáo Kiểm Tra Tính Toàn Vẹn & Thống Kê Tập Dữ Liệu WLASL-100

## 1. Tổng quan tính toàn vẹn (Integrity Overview)

- **Tổng số video theo nhãn gốc (WLASL-100):** 2038 clips
- **Số video hợp lệ (Decode thành công):** 932 clips (45.73%)
- **Số video bị lỗi/hỏng (Corrupted):** 0 clips
- **Số video thiếu (Missing do link die/ngừng hỗ trợ):** 1106 clips

## 2. Thống kê độ dài video (Duration & Frames)

| Chỉ số | Thời lượng (Giây) | Số khung hình (Frames) | Dung lượng (MB) |
| :--- | :---: | :---: | :---: |
| **Nhỏ nhất (Min)** | 0.77s | 19.0 | 0.03 MB |
| **Trung bình (Mean)** | 2.54s | 71.61 | 0.48 MB |
| **Trung vị (Median)** | 2.5s | 70.0 | 0.33 MB |
| **Lớn nhất (Max)** | 8.12s | 212.0 | 7.34 MB |
| **Độ lệch chuẩn (Std)** | 0.89s | 28.03 | 0.74 MB |
| **Phân vị 25% (Q25)** | 1.96s | 53.0 | 0.14 MB |
| **Phân vị 75% (Q75)** | 3.04s | 88.0 | 0.54 MB |

### Phân bố thời lượng (Duration Distribution)
| Khoảng thời lượng | Số lượng video | Tỉ lệ (%) |
| :--- | :---: | :---: |
| < 1.0s (Rất ngắn) | 10 | 1.1% |
| 1.0s - 2.0s (Ngắn) | 233 | 25.0% |
| 2.0s - 3.5s (Trung bình) | 579 | 62.1% |
| 3.5s - 5.0s (Dài) | 101 | 10.8% |
| >= 5.0s (Rất dài) | 9 | 1.0% |

## 3. Thống kê độ phân giải & Tỉ lệ khung hình

### Top độ phân giải phổ biến nhất
| Độ phân giải (Width x Height) | Số lượng clips | Tỉ lệ (%) |
| :--- | :---: | :---: |
| `1920x1080` | 266 | 28.5% |
| `640x480` | 135 | 14.5% |
| `1280x720` | 119 | 12.8% |
| `656x370` | 105 | 11.3% |
| `320x240` | 91 | 9.8% |
| `480x320` | 72 | 7.7% |
| `736x414` | 54 | 5.8% |
| `626x360` | 33 | 3.5% |
| `854x480` | 13 | 1.4% |
| `852x480` | 11 | 1.2% |
| `720x480` | 10 | 1.1% |
| `848x480` | 8 | 0.9% |

### Phân bố Tỉ lệ khung hình (Aspect Ratio)
| Tỉ lệ khung hình | Số lượng clips | Tỉ lệ (%) |
| :--- | :---: | :---: |
| 16:9 | 622 | 66.7% |
| 4:3 | 228 | 24.5% |
| 3:2 | 82 | 8.8% |

### Phân bố Tốc độ khung hình (FPS)
| FPS | Số lượng clips | Tỉ lệ (%) |
| :--- | :---: | :---: |
| 30 FPS | 571 | 61.3% |
| 24 FPS | 216 | 23.2% |
| 25 FPS | 129 | 13.8% |
| 20 FPS | 6 | 0.6% |
| 29 FPS | 5 | 0.5% |
| 60 FPS | 4 | 0.4% |

## 4. Thống kê phân chia theo Split (Train / Val / Test)

| Split | Tổng số mẫu | Số mẫu hợp lệ | Số mẫu thiếu | Tỉ lệ bao phủ | Thời lượng TB | Frame TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 1442 | 633 | 809 | 43.9% | 2.53s | 72.11 frames |
| **VAL** | 338 | 155 | 183 | 45.9% | 2.66s | 73.47 frames |
| **TEST** | 258 | 144 | 114 | 55.8% | 2.44s | 67.44 frames |

## 5. Thống kê độ bao phủ từ vựng (100 Classes)

- **Tổng số lớp:** 100
- **Số lớp có ít nhất 1 video:** 100/100
- **Số lớp có ít nhất 5 video:** 99/100
- **Số mẫu trung bình mỗi lớp:** 9.32
- **Phạm vi số mẫu mỗi lớp:** Min = 4 | Max = 17

## 6. Thống kê theo nguồn gốc (Source)

| Nguồn (Source) | Tổng số | Hợp lệ (Tải được) | Thiếu / Lỗi | Tỉ lệ thành công |
| :--- | :---: | :---: | :---: | :---: |
| `aslu` | 255 | 117 | 138 | 45.9% |
| `signingsavvy` | 225 | 0 | 225 | 0.0% |
| `signschool` | 196 | 196 | 0 | 100.0% |
| `handspeak` | 174 | 0 | 174 | 0.0% |
| `aslpro` | 162 | 0 | 162 | 0.0% |
| `asldeafined` | 131 | 131 | 0 | 100.0% |
| `aslsearch` | 114 | 0 | 114 | 0.0% |
| `aslsignbank` | 108 | 106 | 2 | 98.1% |
| `startasl` | 99 | 99 | 0 | 100.0% |
| `asllex` | 96 | 0 | 96 | 0.0% |
| `spreadthesign` | 91 | 91 | 0 | 100.0% |
| `asl5200` | 82 | 72 | 10 | 87.8% |
| `lillybauer` | 72 | 0 | 72 | 0.0% |
| `nabboud` | 62 | 0 | 62 | 0.0% |
| `valencia-asl` | 53 | 26 | 27 | 49.1% |
| `northtexas` | 51 | 35 | 16 | 68.6% |
| `aslbrick` | 45 | 44 | 1 | 97.8% |
| `scott` | 14 | 7 | 7 | 50.0% |
| `elementalasl` | 8 | 8 | 0 | 100.0% |
