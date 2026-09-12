"""Optional safe NPZ traffic-light classifier with deterministic HSV fallback."""

import os
import numpy as np

try:
    from traffic_light import classify_traffic_light
except ImportError:
    from modules.traffic_light import classify_traffic_light


CLASSES = ("red", "yellow", "green", "unknown")


def build_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(CLASSES)))


class TrafficLightStateClassifier:
    def __init__(self, model_path="", device="cuda"):
        self.model_path = os.path.abspath(model_path) if model_path else ""
        self.device_name = device
        self.model = None
        self.error = None
        if not self.model_path:
            self.error = "learned traffic-light model not configured"
            return
        if not os.path.isfile(self.model_path):
            self.error = f"traffic-light artifact missing: {self.model_path}"
            return
        try:
            import torch
            self.device = torch.device(device if device == "cpu" or torch.cuda.is_available()
                                       else "cpu")
            model = build_model()
            arrays = np.load(self.model_path, allow_pickle=False)
            state = model.state_dict()
            if set(arrays.files) != set(state):
                raise ValueError("NPZ parameter names do not match classifier architecture")
            for name, tensor in state.items():
                value = arrays[name]
                if tuple(value.shape) != tuple(tensor.shape):
                    raise ValueError(f"shape mismatch for {name}: {value.shape} != {tuple(tensor.shape)}")
                tensor.copy_(torch.from_numpy(value))
            self.model = model.to(self.device).eval()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def ready(self):
        return self.model is not None

    def predict(self, crop):
        if not self.ready:
            return classify_traffic_light(crop)
        if crop is None or crop.size == 0:
            return "unknown"
        import cv2
        import torch
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        rgb = resized[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            probabilities = self.model(tensor.unsqueeze(0).to(self.device)).softmax(1)[0]
        confidence, class_id = probabilities.max(0)
        return CLASSES[int(class_id)] if float(confidence) >= 0.55 else "unknown"

