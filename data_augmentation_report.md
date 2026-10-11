# Báo Cáo Kỹ Thuật: Data Augmentation Trên Vector Tọa Độ Landmarks Cho Isolated-SLR

## 1. Mục Tiêu & Đặt Vấn Đề

Trong bài toán nhận dạng ngôn ngữ ký hiệu (Isolated Sign Language Recognition), việc huấn luyện mô hình học sâu trên tập dữ liệu nhỏ hoặc trung bình (như WLASL-100) thường đối mặt với nguy cơ **Overfitting**:
- Mô hình dễ phụ thuộc vào vị trí tuyệt đối hoặc kích thước cơ thể cụ thể của từng người ra dấu.
- Sự rung lắc nhẹ của camera hoặc sai số bám vết vi mô từ MediaPipe Holistic có thể làm giảm độ tin cậy khi nhận dạng thực tế.

Để giải quyết vấn đề này, module **Data Augmentation trên vector tọa độ** được thiết kế chuyên biệt để tăng độ khái quát hóa (Generalization) mà vẫn **bảo toàn hoàn toàn cấu trúc hình thái giải phẫu học của bàn tay và cơ thể**.

---

## 2. Các Kỹ Thuật Data Augmentation Đã Cài Đặt

### 2.1. Thêm Nhiễu Gaussian (Gaussian Noise Injection)
- **Nguyên lý**: Cộng thêm một vector nhiễu ngẫu nhiên $\mathcal{N}(\mu=0, \sigma^2)$ vào từng tọa độ $(x, y, z)$.
- **Tham số chuẩn**: $\sigma \approx 0.01 - 0.03$.
- **Ý nghĩa**: Mô phỏng sự dao động vi mô của các khớp ngón tay và sai số ước lượng tọa độ của MediaPipe, giúp mạng học được các biểu diễn đặc trưng bền vững với nhiễu.

### 2.2. Xoay Góc Nhẹ (Random Slight Rotation)
- **Nguyên lý**: Xoay toàn bộ hệ tọa độ khung xương quanh trục $Z$ (mặt phẳng ảnh $XY$) với góc $\theta \sim \mathcal{U}(-\theta_{\max}, +\theta_{\max})$, $\theta_{\max} \approx 5^\circ - 15^\circ$:
  $$\begin{bmatrix} x' \\ y' \\ z' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x - c_x \\ y - c_y \\ z - c_z \end{bmatrix} + \begin{bmatrix} c_x \\ c_y \\ c_z \end{bmatrix}$$
- **Ý nghĩa**: Mô phỏng góc nghiêng cơ thể tự nhiên của người ra dấu hoặc camera bị đặt hơi nghiêng.

### 2.3. Co Giãn Kích Thước (Random Scaling)
- **Nguyên lý**: Nhân tọa độ tương đối so với tâm quay bằng hệ số co giãn $s \sim \mathcal{U}(s_{\min}, s_{\max})$ (thường $s \in [0.9, 1.1]$ hoặc $[0.85, 1.15]$):
  $$X' = (X - C) \cdot s + C$$
- **Ý nghĩa**: Mô phỏng người ra dấu đứng gần hơn hoặc xa hơn camera, cũng như kích thước bàn tay to/nhỏ khác nhau giữa các người ra dấu.

### 2.4. Dịch Chuyển Vị Trí (Random Translation / Shift)
- **Nguyên lý**: Dịch chuyển tịnh tiến toàn bộ khung xương một khoảng $(\Delta x, \Delta y, \Delta z) \in [-\Delta_{\max}, +\Delta_{\max}]$.
- **Ý nghĩa**: Mô phỏng người ra dấu đứng lệch tâm trong khung hình.

### 2.5. Ẩn Ngẫu Nhiên Điểm (Landmark Dropout)
- **Nguyên lý**: Ngẫu nhiên gán một tỉ lệ nhỏ (ví dụ $3\%$) các điểm mốc về $(0, 0, 0)$.
- **Ý nghĩa**: Mô phỏng việc các ngón tay bị che khuất tạm thời (occlusion) trong quá trình thực hiện cử chỉ.

---

## 3. Đặc Tính Kỹ Thuật Nổi Bật: Bảo Toàn Zero-Mask

Trong dữ liệu Isolated-SLR, một số cử chỉ là **cử chỉ 1 tay** (ví dụ tay trái không xuất hiện và có tọa độ $(0, 0, 0)$) hoặc có các frame vắng mặt bàn tay.
- Nếu áp dụng nhiễu, xoay hoặc co giãn trực tiếp lên các điểm $(0, 0, 0)$, chúng sẽ biến thành các điểm có giá trị khác 0, tạo ra **bàn tay ảo gây nhiễu nghiêm trọng** cho mô hình.
- Tất cả các hàm trong module đều được tích hợp cơ chế **`mask_zeros=True`**:
  $$\text{Mask} = \mathbb{I}(\|P\|_2 > \varepsilon)$$
  Toàn bộ các điểm zero được giữ nguyên tuyệt đối $0.0$, chỉ áp dụng biến đổi hình học lên các bộ phận thực sự xuất hiện!

---

## 4. Trực Quan Hóa Thực Nghiệm

![Data Augmentation Comparison](file:///C:/Users/nguye/.gemini/antigravity-ide/brain/666cd38d-7de8-4935-94a8-09915ce4f388/sample_augmentation_comparison.png)

---

## 5. Danh Sách Tập Tin Đã Cập Nhật

| Tập tin | Chức năng |
| :--- | :--- |
| [`src/data/augmentation.py`](file:///d:/Isolated-SLR/src/data/augmentation.py) | Cài đặt các hàm tăng cường dữ liệu `add_gaussian_noise`, `rotate_landmarks`, `scale_landmarks`, `translate_landmarks`, `dropout_landmarks` và lớp pipeline `CoordinateAugmentor` |
| [`src/data/__init__.py`](file:///d:/Isolated-SLR/src/data/__init__.py) | Xuất khẩu toàn bộ các hàm và lớp augmentation |
| [`src/dataset.py`](file:///d:/Isolated-SLR/src/dataset.py) | Tích hợp tham số `augment=True` và `augmentor` vào `create_dataloader` |
| [`configs/holistic_config.yaml`](file:///d:/Isolated-SLR/configs/holistic_config.yaml) | Thêm cấu hình tham số cho augmentation (noise sigma, rotation angle, scale range) |
| [`scripts/visualize_augmentation.py`](file:///d:/Isolated-SLR/scripts/visualize_augmentation.py) | Script trực quan hóa so sánh khung xương bàn tay trước và sau các phép tăng cường |
| [`tests/test_augmentation.py`](file:///d:/Isolated-SLR/tests/test_augmentation.py) | 11 unit test cases kiểm thử tự động toàn diện |
