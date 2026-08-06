import torch
import cv2
import numpy as np


class DepthEstimator:
    """Ước lượng độ sâu đơn hướng sử dụng MiDaS-small."""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DepthEstimator] Đang tải MiDaS-small trên: {self.device}")

        # Tải mô hình MiDaS torch.hub
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def predict(self, frame):
        """Trả về bản đồ độ sâu chuẩn hóa theo khung hình."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Chuẩn hóa per-frame (vì MiDaS chỉ mang tính chất tương đối trong mỗi frame)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-5:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map
