"""
Module Tăng cường dữ liệu (Data Augmentation) trên vector tọa độ Landmark cho Isolated-SLR.

Các kỹ thuật tăng cường cốt lõi:
1. Thêm nhiễu Gaussian ngẫu nhiên (Gaussian Noise Injection):
   Mô phỏng rung lắc vi mô của camera, sai số ước lượng của MediaPipe Holistic,
   giúp mô hình khái quát hóa tốt hơn và không bị phụ thuộc vào tọa độ tuyệt đối.
2. Xoay góc nhẹ trong không gian 2D/3D (Random Slight Rotation):
   Mô phỏng góc nghiêng cơ thể của người ra dấu hoặc góc đặt camera bị lệch nhẹ
   (thường xoay trong khoảng +/-5 đến +/-15 độ).
3. Co giãn tỉ lệ ngẫu nhiên (Random Scaling):
   Mô phỏng khoảng cách đứng gần/xa camera của người ra dấu, hoặc kích thước
   bàn tay/cơ thể to nhỏ khác nhau giữa các người ra dấu (thường tỉ lệ 0.9 - 1.1).
4. Dịch chuyển vị trí ngẫu nhiên (Random Translation / Shift):
   Mô phỏng người ra dấu đứng lệch tâm nhẹ trong khung hình.
5. Ẩn ngẫu nhiên một số điểm (Landmark Dropout):
   Mô phỏng hiện tượng một vài ngón tay bị che khuất cục bộ.

Tất cả các hàm đều hỗ trợ:
- Vector 201 đặc trưng: shape (T, 201) hoặc (201,)
- Holistic landmarks 3D: shape (T, 543, 3) hoặc (T, 67, 3) hoặc (T, K, 3)
- Cả NumPy ndarray và PyTorch Tensor (tự động giữ nguyên device, dtype)
- Bảo toàn Zero-Mask: Giữ nguyên các điểm (0, 0, 0) của bàn tay vắng mặt,
  tránh sinh ra nhiễu giả tạo tại các vị trí không có bàn tay.
"""

from typing import Optional, Tuple, Union, Any, List, Dict
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _to_3d_landmarks(
    coords: Union[np.ndarray, Any],
) -> Tuple[np.ndarray, Tuple[int, ...], bool, Any, Any]:
    """
    Chuyển đổi linh hoạt đầu vào dạng bất kỳ về mảng numpy 3D (T, K, 3) để thao tác hình học.

    Args:
        coords: np.ndarray hoặc torch.Tensor với shape (T, 201), (201,), (T, K, 3), (K, 3), ...

    Returns:
        coords_3d: np.ndarray shape (T, K, 3) kiểu float32.
        orig_shape: shape ban đầu của đầu vào.
        was_tensor: True nếu đầu vào ban đầu là torch.Tensor.
        device: device của tensor (nếu có).
        dtype: dtype ban đầu của tensor (nếu có).
    """
    was_tensor = False
    device = None
    dtype = None

    if HAS_TORCH and isinstance(coords, torch.Tensor):
        was_tensor = True
        device = coords.device
        dtype = coords.dtype
        arr = coords.detach().cpu().numpy()
    elif isinstance(coords, np.ndarray):
        arr = coords
    else:
        raise TypeError(f"Dữ liệu coords phải là np.ndarray hoặc torch.Tensor, nhận được: {type(coords)}")

    orig_shape = arr.shape

    # Trường hợp 1: Vector 201 1D: (201,)
    if arr.ndim == 1:
        if arr.shape[0] % 3 != 0:
            raise ValueError(f"Chiều của vector 1D ({arr.shape[0]}) không chia hết cho 3!")
        k = arr.shape[0] // 3
        coords_3d = arr.reshape(1, k, 3).astype(np.float32, copy=True)

    # Trường hợp 2: Vector 2D: (T, 201) hoặc (T, K*3) hoặc (K, 3)
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            # (K, 3) -> 1 frame
            coords_3d = arr.reshape(1, arr.shape[0], 3).astype(np.float32, copy=True)
        elif arr.shape[1] % 3 == 0:
            # (T, D) với D = K * 3 (ví dụ (T, 201) với K=67)
            t = arr.shape[0]
            k = arr.shape[1] // 3
            coords_3d = arr.reshape(t, k, 3).astype(np.float32, copy=True)
        else:
            raise ValueError(f"Chiều đặc trưng của mảng 2D ({arr.shape[1]}) không chia hết cho 3!")

    # Trường hợp 3: Mảng 3D: (T, K, 3)
    elif arr.ndim == 3:
        if arr.shape[2] != 3:
            raise ValueError(f"Chiều tọa độ cuối cùng phải bằng 3 (x, y, z), nhận được shape: {arr.shape}")
        coords_3d = arr.astype(np.float32, copy=True)

    else:
        raise ValueError(f"Số chiều dữ liệu không hợp lệ: ndim={arr.ndim}, shape={orig_shape}")

    return coords_3d, orig_shape, was_tensor, device, dtype


def _from_3d_landmarks(
    coords_3d: np.ndarray,
    orig_shape: Tuple[int, ...],
    was_tensor: bool,
    device: Any,
    dtype: Any,
) -> Union[np.ndarray, Any]:
    """
    Phục hồi mảng 3D (T, K, 3) về lại định dạng và shape gốc (np.ndarray hoặc torch.Tensor).
    """
    res = coords_3d.reshape(orig_shape)

    if was_tensor and HAS_TORCH:
        res_tensor = torch.from_numpy(res)
        if dtype is not None:
            res_tensor = res_tensor.to(dtype=dtype)
        if device is not None:
            res_tensor = res_tensor.to(device=device)
        return res_tensor

    return res


def _compute_center(
    coords_3d: np.ndarray,
    center_mode: Union[str, np.ndarray, Tuple[float, float, float]] = "origin",
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Tính tâm quay/co giãn của hệ tọa độ landmarks.

    Args:
        coords_3d: Mảng shape (T, K, 3).
        center_mode:
            - "origin": Gốc tọa độ (0, 0, 0).
            - "mean" hoặc "centroid": Trung bình các điểm khác 0 trong chuỗi.
            - Hoặc mảng/tuple tọa độ 3D cụ thể (cx, cy, cz).
    """
    if isinstance(center_mode, (list, tuple, np.ndarray)):
        c = np.asarray(center_mode, dtype=np.float32)
        if c.shape != (3,):
            raise ValueError(f"Tâm tùy chọn phải có 3 tọa độ (x, y, z), nhận được: {c.shape}")
        return c

    if center_mode == "origin":
        return np.zeros(3, dtype=np.float32)

    if center_mode in ("mean", "centroid"):
        nonzero_mask = np.any(np.abs(coords_3d) > eps, axis=-1)  # (T, K)
        if np.any(nonzero_mask):
            return coords_3d[nonzero_mask].mean(axis=0).astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    raise ValueError(f"center_mode không hợp lệ: '{center_mode}'. Chọn 'origin', 'mean', hoặc tọa độ (3,).")


def add_gaussian_noise(
    coords: Union[np.ndarray, Any],
    sigma: float = 0.01,
    mean: float = 0.0,
    mask_zeros: bool = True,
    prob: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-5,
) -> Union[np.ndarray, Any]:
    """
    Thêm nhiễu Gaussian ngẫu nhiên N(mean, sigma^2) vào vector tọa độ landmark.

    Nhiễu Gaussian mô phỏng rung chấn nhẹ của camera và sai số phát hiện của MediaPipe,
    giúp mô hình nhận dạng không bị overfitting vào các tọa độ tuyệt đối.

    Args:
        coords: Dữ liệu tọa độ landmarks (shape (T, 201), (201,), (T, K, 3), ...).
        sigma: Độ lệch chuẩn của phân phối Gaussian (mặc định: 0.01).
        mean: Giá trị trung bình của nhiễu (mặc định: 0.0).
        mask_zeros: Nếu True, không thêm nhiễu vào các điểm (0, 0, 0) biểu thị bàn tay vắng mặt.
        prob: Xác suất áp dụng phép tăng cường (0.0 đến 1.0).
        seed: Random seed (tùy chọn).
        eps: Ngưỡng kiểm tra điểm 0.

    Returns:
        noisy_coords: Dữ liệu tọa độ sau khi thêm nhiễu, cùng kiểu và shape với đầu vào.
    """
    if prob < 1.0:
        rng = np.random.default_rng(seed)
        if rng.random() > prob:
            return coords

    coords_3d, orig_shape, was_tensor, device, dtype = _to_3d_landmarks(coords)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=coords_3d.shape).astype(np.float32)

    if mask_zeros:
        # Mặt nạ các điểm có tọa độ khác 0
        nonzero_mask = np.any(np.abs(coords_3d) > eps, axis=-1, keepdims=True)
        augmented_3d = np.where(nonzero_mask, coords_3d + noise, coords_3d)
    else:
        augmented_3d = coords_3d + noise

    return _from_3d_landmarks(augmented_3d, orig_shape, was_tensor, device, dtype)


def rotate_landmarks(
    coords: Union[np.ndarray, Any],
    angle_deg: Optional[float] = None,
    max_angle_deg: float = 10.0,
    axis: str = "z",
    center: Union[str, np.ndarray, Tuple[float, float, float]] = "origin",
    mask_zeros: bool = True,
    prob: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-5,
) -> Union[np.ndarray, Any]:
    """
    Xoay góc nhẹ toàn bộ hệ khung xương landmarks trong không gian 2D/3D.

    Trong ngôn ngữ ký hiệu, việc xoay quanh trục Z (mặt phẳng ảnh XY) là phép biến đổi
    thực tế nhất, mô phỏng người ra dấu hơi nghiêng người hoặc camera bị nghiêng góc nhẹ.

    Args:
        coords: Dữ liệu tọa độ landmarks (shape (T, 201), (201,), (T, K, 3), ...).
        angle_deg: Góc xoay cố định theo độ (nếu None, lấy ngẫu nhiên trong [-max_angle_deg, max_angle_deg]).
        max_angle_deg: Giới hạn góc xoay tối đa theo độ (mặc định: 10.0 độ).
        axis: Trục xoay ("z" cho mặt phẳng ảnh XY, "x", "y", hoặc "3d").
        center: Tâm quay ("origin" cho (0,0,0), "mean" cho trung tâm bàn tay/cơ thể, hoặc tọa độ cụ thể).
        mask_zeros: Giữ nguyên các điểm (0, 0, 0) của bàn tay vắng mặt.
        prob: Xác suất áp dụng phép tăng cường (0.0 đến 1.0).
        seed: Random seed (tùy chọn).
        eps: Ngưỡng kiểm tra điểm 0.

    Returns:
        rotated_coords: Dữ liệu tọa độ sau khi xoay, cùng kiểu và shape với đầu vào.
    """
    if prob < 1.0:
        rng = np.random.default_rng(seed)
        if rng.random() > prob:
            return coords

    coords_3d, orig_shape, was_tensor, device, dtype = _to_3d_landmarks(coords)
    rng = np.random.default_rng(seed)

    # 1. Tính toán ma trận xoay R (3x3)
    if axis in ("z", "2d"):
        deg = angle_deg if angle_deg is not None else rng.uniform(-max_angle_deg, max_angle_deg)
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    elif axis == "x":
        deg = angle_deg if angle_deg is not None else rng.uniform(-max_angle_deg, max_angle_deg)
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  c, -s],
            [0.0,  s,  c],
        ], dtype=np.float32)

    elif axis == "y":
        deg = angle_deg if angle_deg is not None else rng.uniform(-max_angle_deg, max_angle_deg)
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ], dtype=np.float32)

    elif axis == "3d":
        # Xoay 3D với 3 góc Euler nhỏ
        ax_deg = rng.uniform(-max_angle_deg, max_angle_deg)
        ay_deg = rng.uniform(-max_angle_deg, max_angle_deg)
        az_deg = rng.uniform(-max_angle_deg, max_angle_deg)
        rx, ry, rz = np.deg2rad(ax_deg), np.deg2rad(ay_deg), np.deg2rad(az_deg)

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = (Rz @ Ry @ Rx).astype(np.float32)

    else:
        raise ValueError(f"Trục xoay axis='{axis}' không hợp lệ. Chọn 'z', 'x', 'y', hoặc '3d'.")

    # 2. Xác định tâm quay
    center_pt = _compute_center(coords_3d, center_mode=center, eps=eps)  # shape (3,)

    # 3. Áp dụng ma trận xoay quanh tâm: (X - C) @ R.T + C
    shifted = coords_3d - center_pt
    rotated_pts = np.matmul(shifted, R.T) + center_pt

    if mask_zeros:
        nonzero_mask = np.any(np.abs(coords_3d) > eps, axis=-1, keepdims=True)
        augmented_3d = np.where(nonzero_mask, rotated_pts, coords_3d)
    else:
        augmented_3d = rotated_pts

    return _from_3d_landmarks(augmented_3d, orig_shape, was_tensor, device, dtype)


def scale_landmarks(
    coords: Union[np.ndarray, Any],
    scale_factor: Optional[Union[float, Tuple[float, float, float]]] = None,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    center: Union[str, np.ndarray, Tuple[float, float, float]] = "origin",
    mask_zeros: bool = True,
    prob: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-5,
) -> Union[np.ndarray, Any]:
    """
    Co giãn (Scale) tỉ lệ khung xương ngẫu nhiên hoặc theo hệ số chỉ định.

    Mô phỏng khoảng cách camera gần/xa và độ dài ngón tay/sải tay của các người ra dấu khác nhau.

    Args:
        coords: Dữ liệu tọa độ landmarks (shape (T, 201), (201,), (T, K, 3), ...).
        scale_factor: Hệ số co giãn cụ thể (float hoặc tuple 3 chiều (sx, sy, sz)).
                      Nếu None, lấy ngẫu nhiên trong [scale_range[0], scale_range[1]].
        scale_range: Khoảng giá trị co giãn ngẫu nhiên (mặc định: (0.9, 1.1)).
        center: Tâm co giãn ("origin", "mean", hoặc tọa độ cụ thể).
        mask_zeros: Giữ nguyên các điểm (0, 0, 0).
        prob: Xác suất áp dụng (0.0 đến 1.0).
        seed: Random seed.
        eps: Ngưỡng kiểm tra điểm 0.

    Returns:
        scaled_coords: Dữ liệu tọa độ sau khi co giãn, cùng kiểu và shape với đầu vào.
    """
    if prob < 1.0:
        rng = np.random.default_rng(seed)
        if rng.random() > prob:
            return coords

    coords_3d, orig_shape, was_tensor, device, dtype = _to_3d_landmarks(coords)
    rng = np.random.default_rng(seed)

    if scale_factor is not None:
        if isinstance(scale_factor, (tuple, list, np.ndarray)):
            s = np.asarray(scale_factor, dtype=np.float32)
        else:
            s = float(scale_factor)
    else:
        s = float(rng.uniform(scale_range[0], scale_range[1]))

    center_pt = _compute_center(coords_3d, center_mode=center, eps=eps)

    # Co giãn quanh tâm: (X - C) * s + C
    shifted = coords_3d - center_pt
    scaled_pts = shifted * s + center_pt

    if mask_zeros:
        nonzero_mask = np.any(np.abs(coords_3d) > eps, axis=-1, keepdims=True)
        augmented_3d = np.where(nonzero_mask, scaled_pts, coords_3d)
    else:
        augmented_3d = scaled_pts

    return _from_3d_landmarks(augmented_3d, orig_shape, was_tensor, device, dtype)


def translate_landmarks(
    coords: Union[np.ndarray, Any],
    shift: Optional[Union[List[float], Tuple[float, float, float], np.ndarray]] = None,
    max_shift: float = 0.03,
    mask_zeros: bool = True,
    prob: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-5,
) -> Union[np.ndarray, Any]:
    """
    Dịch chuyển (Shift / Translation) ngẫu nhiên toàn bộ khung xương trong không gian.

    Mô phỏng người ra dấu đứng lệch tâm nhẹ sang trái/phải hoặc lên/xuống trong khung hình.

    Args:
        coords: Dữ liệu tọa độ landmarks.
        shift: Vector dịch chuyển cụ thể (dx, dy, dz). Nếu None, lấy ngẫu nhiên trong [-max_shift, max_shift].
        max_shift: Độ dịch chuyển tối đa cho mỗi trục (mặc định: 0.03).
        mask_zeros: Giữ nguyên các điểm 0.
        prob: Xác suất áp dụng.
        seed: Random seed.
        eps: Ngưỡng kiểm tra điểm 0.

    Returns:
        shifted_coords: Dữ liệu tọa độ sau khi dịch chuyển.
    """
    if prob < 1.0:
        rng = np.random.default_rng(seed)
        if rng.random() > prob:
            return coords

    coords_3d, orig_shape, was_tensor, device, dtype = _to_3d_landmarks(coords)
    rng = np.random.default_rng(seed)

    if shift is not None:
        delta = np.asarray(shift, dtype=np.float32)
    else:
        # Dịch chuyển x, y chính, z nhẹ hơn
        dx = rng.uniform(-max_shift, max_shift)
        dy = rng.uniform(-max_shift, max_shift)
        dz = rng.uniform(-max_shift * 0.5, max_shift * 0.5)
        delta = np.array([dx, dy, dz], dtype=np.float32)

    shifted_pts = coords_3d + delta

    if mask_zeros:
        nonzero_mask = np.any(np.abs(coords_3d) > eps, axis=-1, keepdims=True)
        augmented_3d = np.where(nonzero_mask, shifted_pts, coords_3d)
    else:
        augmented_3d = shifted_pts

    return _from_3d_landmarks(augmented_3d, orig_shape, was_tensor, device, dtype)


def dropout_landmarks(
    coords: Union[np.ndarray, Any],
    drop_prob: float = 0.03,
    mask_zeros: bool = True,
    prob: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-5,
) -> Union[np.ndarray, Any]:
    """
    Ẩn ngẫu nhiên (Dropout) một số điểm landmark về 0.

    Mô phỏng hiện tượng MediaPipe bị mất dấu cục bộ 1 vài khớp ngón tay trong một số khung hình.

    Args:
        coords: Dữ liệu tọa độ landmarks.
        drop_prob: Tỉ lệ phần trăm các điểm landmark bị gán về 0 (mặc định: 0.03, tức 3%).
        mask_zeros: Giữ nguyên các điểm 0 có sẵn.
        prob: Xác suất áp dụng phép biến đổi.
        seed: Random seed.
        eps: Ngưỡng kiểm tra điểm 0.

    Returns:
        dropped_coords: Dữ liệu sau khi dropout ngẫu nhiên.
    """
    if prob < 1.0:
        rng = np.random.default_rng(seed)
        if rng.random() > prob:
            return coords

    coords_3d, orig_shape, was_tensor, device, dtype = _to_3d_landmarks(coords)
    rng = np.random.default_rng(seed)

    # Sinh mask dropout (T, K, 1)
    t, k, _ = coords_3d.shape
    drop_mask = rng.random((t, k, 1)) < drop_prob  # True nghĩa là bị drop

    augmented_3d = coords_3d.copy()
    augmented_3d[drop_mask.squeeze(-1)] = 0.0

    return _from_3d_landmarks(augmented_3d, orig_shape, was_tensor, device, dtype)


class CoordinateAugmentor:
    """
    Lớp Pipeline tổng hợp các kỹ thuật Data Augmentation trên vector tọa độ Landmarks.

    Tương thích hoàn hảo làm đối số `transform` trong PyTorch `DataLoader` và `Dataset`:
        dataset = WLASLDataset(..., transform=CoordinateAugmentor(apply_noise=True, ...))

    Thứ tự áp dụng chuẩn trong Pipeline:
    1. Co giãn tỉ lệ (Scaling)
    2. Xoay góc nhẹ (Rotation)
    3. Dịch chuyển vị trí (Translation / Shift)
    4. Thêm nhiễu Gaussian (Gaussian Noise)
    5. Dropout ngẫu nhiên (tùy chọn)
    """

    def __init__(
        self,
        # 1. Gaussian Noise
        apply_noise: bool = True,
        noise_sigma: float = 0.01,
        noise_mean: float = 0.0,
        noise_prob: float = 0.5,
        # 2. Rotation
        apply_rotation: bool = True,
        rotation_max_angle: float = 10.0,
        rotation_axis: str = "z",
        rotation_prob: float = 0.5,
        # 3. Scaling
        apply_scale: bool = True,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        scale_prob: float = 0.5,
        # 4. Translation
        apply_shift: bool = False,
        max_shift: float = 0.03,
        shift_prob: float = 0.3,
        # 5. Dropout
        apply_dropout: bool = False,
        drop_prob: float = 0.03,
        dropout_prob: float = 0.2,
        # Cấu hình chung
        center_mode: str = "origin",
        mask_zeros: bool = True,
        seed: Optional[int] = None,
    ):
        self.apply_noise = apply_noise
        self.noise_sigma = noise_sigma
        self.noise_mean = noise_mean
        self.noise_prob = noise_prob

        self.apply_rotation = apply_rotation
        self.rotation_max_angle = rotation_max_angle
        self.rotation_axis = rotation_axis
        self.rotation_prob = rotation_prob

        self.apply_scale = apply_scale
        self.scale_range = scale_range
        self.scale_prob = scale_prob

        self.apply_shift = apply_shift
        self.max_shift = max_shift
        self.shift_prob = shift_prob

        self.apply_dropout = apply_dropout
        self.drop_prob = drop_prob
        self.dropout_prob = dropout_prob

        self.center_mode = center_mode
        self.mask_zeros = mask_zeros
        self.seed = seed

    def __call__(self, coords: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Thực thi pipeline tăng cường dữ liệu trên tọa độ đầu vào."""
        out = coords

        # 1. Co giãn tỉ lệ (Scale)
        if self.apply_scale:
            out = scale_landmarks(
                out,
                scale_range=self.scale_range,
                center=self.center_mode,
                mask_zeros=self.mask_zeros,
                prob=self.scale_prob,
                seed=self.seed,
            )

        # 2. Xoay góc nhẹ (Rotation)
        if self.apply_rotation:
            out = rotate_landmarks(
                out,
                max_angle_deg=self.rotation_max_angle,
                axis=self.rotation_axis,
                center=self.center_mode,
                mask_zeros=self.mask_zeros,
                prob=self.rotation_prob,
                seed=self.seed,
            )

        # 3. Dịch chuyển vị trí (Shift)
        if self.apply_shift:
            out = translate_landmarks(
                out,
                max_shift=self.max_shift,
                mask_zeros=self.mask_zeros,
                prob=self.shift_prob,
                seed=self.seed,
            )

        # 4. Thêm nhiễu Gaussian (Noise)
        if self.apply_noise:
            out = add_gaussian_noise(
                out,
                sigma=self.noise_sigma,
                mean=self.noise_mean,
                mask_zeros=self.mask_zeros,
                prob=self.noise_prob,
                seed=self.seed,
            )

        # 5. Dropout ngẫu nhiên (tùy chọn)
        if self.apply_dropout:
            out = dropout_landmarks(
                out,
                drop_prob=self.drop_prob,
                mask_zeros=self.mask_zeros,
                prob=self.dropout_prob,
                seed=self.seed,
            )

        return out

    def forward(self, coords: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        return self.__call__(coords)


# Alias tương thích
SignAugmentor = CoordinateAugmentor
