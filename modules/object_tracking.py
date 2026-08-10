# fmt: off
# isort: skip_file
"""Phát hiện phương tiện bằng YOLO — hỗ trợ mô hình TỐI ƯU (TensorRT/ONNX).

Tự động dùng bản đã biên dịch nếu có: <model>.engine (TensorRT) > <model>.onnx
(ONNX Runtime) > <model>.pt (PyTorch gốc). Xuất bằng export_models.py. FP16 nên
GIỮ NGUYÊN độ chính xác nhưng suy luận nhanh 2-4x -> giảm độ trễ inference và bớt
kẹt vòng đồng bộ (server ít phải chờ client).
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
import concurrent.futures


def _resolve_model(name, use_optimized=True):
    """Trả về (đường_dẫn, đã_tối_ưu?) — ưu tiên .engine > .onnx > .pt."""
    if use_optimized:
        base = os.path.splitext(name)[0]
        for ext in ('.engine', '.onnx'):
            cand = base + ext
            if os.path.exists(cand):
                return cand, True
    return name, False


def draw_detections(frame, detections):
    """Vẽ lại box từ danh sách detection đã lưu (dùng cho khung 'nội suy' khi bỏ qua YOLO)."""
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, f"{d.get('class', '?')} {d.get('confidence', 0):.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    return frame


class EnsembleVehicleTracker:
    def __init__(self, device='cuda', use_ensemble=False,
                 model_a='yolov8n.pt', model_b='yolov10n.pt',
                 imgsz=640, half=True, use_optimized=True):
        self.device = device
        self.imgsz = imgsz
        self.half = bool(half) and device != 'cpu'

        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Danh sách (model, đã_tối_ưu?). Nạp 1 hoặc 2 mô hình (ensemble).
        self.models = []
        names = [model_a] + ([model_b] if use_ensemble else [])
        for nm in names:
            path, optimized = _resolve_model(nm, use_optimized)
            model = YOLO(path)
            if not optimized:
                model.to(device)   # .engine/.onnx đã gắn thiết bị sẵn
            self.models.append((model, optimized))
            tag = "TỐI ƯU" if optimized else "PyTorch"
            print(f"[YOLO] {os.path.basename(path)} ({tag})")

        self.target_classes = [2, 3, 5, 7]  # COCO: Car, Motorcycle, Bus, Truck
        self.class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        self.nms_threshold = 0.45
        self.conf_threshold = 0.40

        # Thread pool DÙNG LẠI (không tạo mới mỗi khung) -> bớt trễ/giật khi chạy ensemble.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(self.models)))

    def _infer_single_model(self, model, optimized, rgb_frame):
        kwargs = dict(classes=self.target_classes, verbose=False, imgsz=self.imgsz,
                      conf=self.conf_threshold, iou=self.nms_threshold)
        if not optimized:
            # .pt: chỉ định thiết bị + FP16 (API chuẩn Ultralytics là 'half=True';
            # 'quantize' KHÔNG phải tham số predict nên trước đây bị bỏ qua -> chưa
            # thực sự chạy FP16). half chỉ bật trên GPU.
            kwargs["device"] = self.device
            kwargs["half"] = self.half
        results = model(rgb_frame, **kwargs)[0]
        boxes, scores, class_ids = [], [], []
        for box in results.boxes:
            boxes.append(box.xyxy[0].cpu().numpy().astype(int).tolist())
            scores.append(float(box.conf[0]))
            class_ids.append(int(box.cls[0]))
        return boxes, scores, class_ids

    def process(self, rgb_frame, depth_map=None, frame_count=0):
        if len(self.models) > 1:
            futs = [self._executor.submit(self._infer_single_model, m, o, rgb_frame)
                    for (m, o) in self.models]
            results = [f.result() for f in futs]
            all_boxes, all_scores, all_cls = [], [], []
            for b, s, c in results:
                all_boxes += b
                all_scores += s
                all_cls += c
        else:
            m, o = self.models[0]
            all_boxes, all_scores, all_cls = self._infer_single_model(m, o, rgb_frame)

        detections = []
        annotated_frame = rgb_frame.copy()

        if all_boxes:
            nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_boxes]
            indices = cv2.dnn.NMSBoxes(nms_boxes, all_scores, self.conf_threshold, self.nms_threshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = all_boxes[i]
                    conf = all_scores[i]
                    label = self.class_names.get(all_cls[i], 'Unknown')
                    detections.append({'bbox': (x1, y1, x2, y2), 'class': label, 'confidence': conf})
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        metrics = {'detected_vehicles': len(detections)}
        return annotated_frame, detections, metrics
