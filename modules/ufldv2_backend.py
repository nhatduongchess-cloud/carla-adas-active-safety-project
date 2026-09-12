"""Restricted UFLDv2 Tusimple ResNet18 inference backend.

No upstream module or install hook is imported. The graph and decoder follow
Ultra-Fast-Lane-Detection-v2 commit c903880678454dfd9b55a63022368db05c00bc6d
(MIT), while the local checkpoint is hash-pinned and loaded weights-only.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext

import cv2
import numpy as np


SOURCE_COMMIT = "c903880678454dfd9b55a63022368db05c00bc6d"
INPUT_SIZE = (800, 320)
RESIZE_HEIGHT = 400
ROW_ANCHORS = np.linspace(160.0, 710.0, 56, dtype=np.float32) / 720.0
COL_ANCHORS = np.linspace(0.0, 1.0, 41, dtype=np.float32)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model(torch, torchvision):
    class ResNet18Features(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = torchvision.models.resnet18(weights=None)  # never download
            for name in ("conv1", "bn1", "relu", "maxpool", "layer1",
                         "layer2", "layer3", "layer4"):
                setattr(self, name, getattr(base, name))

        def forward(self, image):
            image = self.maxpool(self.relu(self.bn1(self.conv1(image))))
            image = self.layer1(image)
            x2 = self.layer2(image)
            x3 = self.layer3(x2)
            return x2, x3, self.layer4(x3)

    class ParsingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ResNet18Features()
            self.pool = torch.nn.Conv2d(512, 8, 1)
            self.cls = torch.nn.Sequential(
                torch.nn.Identity(), torch.nn.Linear(2000, 2048),
                torch.nn.ReLU(), torch.nn.Linear(2048, 39576))
            self.inner_lanes_only = False

        def retain_inner_lane_head(self):
            """Keep only the two row-classification lanes around the ego car.

            The official Tusimple output also contains the two outer lanes via
            column classification. They are useful for full road annotation,
            but the controller consumes only the nearest left/right boundary.
            Slicing the already verified final-layer weights avoids computing
            28,152 unused logits without changing the retained predictions.
            """
            final = self.cls[3]
            loc_indices = torch.arange(22400).reshape(100, 56, 4)[:, :, 1:3]
            exist_indices = (
                torch.arange(38800, 39248).reshape(2, 56, 4)[:, :, 1:3])
            selected = torch.cat((loc_indices.reshape(-1),
                                  exist_indices.reshape(-1)))
            reduced = torch.nn.Linear(final.in_features, int(selected.numel()))
            with torch.no_grad():
                reduced.weight.copy_(final.weight[selected])
                reduced.bias.copy_(final.bias[selected])
            self.cls[3] = reduced
            self.inner_lanes_only = True

        def forward(self, image):
            _x2, _x3, features = self.model(image)
            output = self.cls(self.pool(features).reshape(-1, 2000))
            if self.inner_lanes_only:
                return {
                    "loc_row": output[:, :11200].reshape(-1, 100, 56, 2),
                    "exist_row": output[:, 11200:].reshape(-1, 2, 56, 2),
                }
            return {
                "loc_row": output[:, :22400].reshape(-1, 100, 56, 4),
                "loc_col": output[:, 22400:38800].reshape(-1, 100, 41, 4),
                "exist_row": output[:, 38800:39248].reshape(-1, 2, 56, 4),
                "exist_col": output[:, 39248:].reshape(-1, 2, 41, 4),
            }

    return ParsingNet()


def _tensor_state(torch, state):
    if not isinstance(state, dict) or not state:
        raise ValueError("UFLDv2 checkpoint has no model state dictionary")
    clean = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("UFLDv2 state dictionary must contain tensors only")
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


class UFLDv2Backend:
    """Callable OpenCV-BGR to original-frame lane-pixel adapter."""

    def __init__(self, model_path, expected_sha256, device="cuda", use_fp16=True):
        import torch
        import torchvision

        self.torch = torch
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"UFLDv2 checkpoint not found: {self.model_path}")
        expected = str(expected_sha256 or "").strip().lower()
        if len(expected) != 64:
            raise ValueError("UFLDv2 requires a 64-character SHA-256")
        actual = sha256_file(self.model_path)
        if actual != expected:
            raise ValueError(f"UFLDv2 SHA-256 mismatch: expected {expected}, got {actual}")
        if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("UFLDv2 requested CUDA but CUDA is unavailable")

        self.device = torch.device(str(device).lower())
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        model = _build_model(torch, torchvision)
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(_tensor_state(torch, state), strict=True)
        del checkpoint, state
        model.retain_inner_lane_head()
        model.eval().requires_grad_(False).to(self.device)
        if self.use_fp16:
            model.half()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            model.to(memory_format=torch.channels_last)
        self.model = model
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        # A dedicated stream lets this independent network overlap with the
        # object detector when both are requested for the same bounded job.
        self.cuda_stream = (torch.cuda.Stream(device=self.device)
                            if self.device.type == "cuda" else None)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device,
                                 dtype=self.dtype).reshape(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device,
                                dtype=self.dtype).reshape(1, 3, 1, 1)

    def _prepare(self, frame, input_size):
        if tuple(map(int, input_size)) != INPUT_SIZE:
            raise ValueError(f"Tusimple ResNet18 requires {INPUT_SIZE}, got {input_size}")
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("UFLDv2 expects an HxWx3 BGR frame")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE[0], RESIZE_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
        cropped = np.ascontiguousarray(resized[-INPUT_SIZE[1]:])
        tensor = self.torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, self.dtype).div_(255.0)
        if self.device.type == "cuda":
            tensor = tensor.contiguous(memory_format=self.torch.channels_last)
        return tensor.sub_(self.mean).div_(self.std)

    @staticmethod
    def _coordinates(torch, logits):
        """Vectorized upstream local-softmax decoder for every anchor."""
        # logits: [grid, anchors]
        grid_size = logits.shape[0]
        peaks = logits.argmax(dim=0)
        offsets = torch.tensor((-1, 0, 1), device=logits.device)
        indices = peaks[:, None] + offsets[None, :]
        in_range = (indices >= 0) & (indices < grid_size)
        safe_indices = indices.clamp(0, grid_size - 1)
        local_logits = logits.transpose(0, 1).gather(1, safe_indices)
        local_logits = local_logits.float().masked_fill(~in_range, float("-inf"))
        weights = local_logits.softmax(dim=1)
        return (weights * safe_indices.float()).sum(dim=1) + 0.5

    def _decode(self, prediction, image_width, image_height):
        torch = self.torch
        lanes, confidences = [], []
        loc = prediction["loc_row"][0]
        existence = prediction["exist_row"][0].float().softmax(0)[1]
        valid = existence > 0.5
        for lane_index in range(loc.shape[2]):
            lane_valid = valid[:, lane_index]
            if int(lane_valid.sum().item()) <= loc.shape[1] / 2:
                continue
            x = self._coordinates(torch, loc[:, :, lane_index])
            y = torch.as_tensor(ROW_ANCHORS, device=loc.device) * image_height
            points = torch.stack(
                (x / (loc.shape[0] - 1) * image_width, y), dim=1)[lane_valid]
            lanes.append(points.detach().cpu().tolist())
            confidences.append(float(existence[lane_valid, lane_index].mean().item()))
        return lanes, confidences

    def __call__(self, frame, input_size):
        height, width = frame.shape[:2]
        stream_context = (self.torch.cuda.stream(self.cuda_stream)
                          if self.cuda_stream is not None else nullcontext())
        with self.torch.inference_mode(), stream_context:
            prediction = self.model(self._prepare(frame, input_size))
            lanes, scores = self._decode(prediction, width, height)
        if self.cuda_stream is not None:
            self.cuda_stream.synchronize()
        confidence = min(sorted(scores, reverse=True)[:2]) if len(scores) >= 2 else 0.0
        return {"lane_pixels": lanes, "lane_confidences": scores,
                "confidence": float(confidence), "source_commit": SOURCE_COMMIT}
