# fmt: off
# isort: skip_file
"""Phát hiện phương tiện bằng YOLO — hỗ trợ mô hình TỐI ƯU (TensorRT/ONNX).

Tự động dùng bản đã biên dịch nếu có: <model>.engine (TensorRT) > <model>.onnx
(ONNX Runtime) > <model>.pt (PyTorch gốc). Xuất bằng export_models.py. FP16 nên
GIỮ NGUYÊN độ chính xác nhưng suy luận nhanh 2-4x -> giảm độ trễ inference và bớt
kẹt vòng đồng bộ (server ít phải chờ client).
"""
import os
import hashlib
import cv2
import numpy as np
from ultralytics import YOLO
import concurrent.futures
import time

try:
    from traffic_light import TrafficLightTemporalVoter
    from traffic_light_classifier import TrafficLightStateClassifier
except ImportError:
    from modules.traffic_light import TrafficLightTemporalVoter
    from modules.traffic_light_classifier import TrafficLightStateClassifier


def _resolve_model(name, use_optimized=True):
    """Trả về (đường_dẫn, đã_tối_ưu?) — ưu tiên .engine > .onnx > .pt."""
    if use_optimized:
        base = os.path.splitext(name)[0]
        for ext in ('.engine', '.onnx'):
            cand = base + ext
            if os.path.exists(cand):
                return cand, True
    return name, False


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                 imgsz=640, half=True, use_optimized=True,
                 target_classes=None, class_names=None,
                 conf_threshold=0.40, nms_threshold=0.45,
                 traffic_light_model_path=""):
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
        self.model_provenance = []
        self.provenance_errors = []
        names = [model_a] + ([model_b] if use_ensemble else [])
        for nm in names:
            path, optimized = _resolve_model(nm, use_optimized)
            # Ultralytics accepts aliases such as "yolov8n.pt" and downloads
            # them automatically. Demo runtime forbids that behavior: a weight
            # must already exist locally and is hashed before it is loaded.
            if not os.path.isfile(path):
                error = f"model weight missing; auto-download disabled: {os.path.abspath(path)}"
                self.provenance_errors.append(error)
                print(f"[YOLO] FALLBACK LiDAR-only safety — {error}")
                continue
            model = YOLO(path)
            if not optimized:
                model.to(device)   # .engine/.onnx đã gắn thiết bị sẵn
            self.models.append((model, optimized))
            tag = "TỐI ƯU" if optimized else "PyTorch"
            digest = _sha256(path)
            self.model_provenance.append({
                "path": os.path.abspath(path), "sha256": digest,
                "runtime": tag, "size_bytes": os.path.getsize(path)})
            print(f"[YOLO] {os.path.basename(path)} ({tag}, sha256={digest[:12]}…)")

        self.target_classes = list(target_classes or [0, 1, 2, 3, 5, 7, 9, 11])
        self.class_names = class_names or {
            0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle',
            5: 'Bus', 7: 'Truck', 9: 'TrafficLight', 11: 'StopSign'}
        self.nms_threshold = float(nms_threshold)
        self.conf_threshold = float(conf_threshold)
        self.light_voter = TrafficLightTemporalVoter(window=5, min_votes=3)
        self.light_classifier = TrafficLightStateClassifier(
            traffic_light_model_path, device=device)

        # Thread pool DÙNG LẠI (không tạo mới mỗi khung) -> bớt trễ/giật khi chạy ensemble.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(self.models)))

    def _infer_single_model(self, model, optimized, rgb_frame):
        kwargs = dict(classes=self.target_classes, verbose=False, imgsz=self.imgsz,
                      conf=self.conf_threshold, iou=self.nms_threshold)
        if not optimized:
            # .pt: chỉ định thiết bị + độ chính xác suy luận. Ultralytics 8.4 HỢP NHẤT
            # cờ precision vào 'quantize' (16=FP16, 32=FP32); 'half'/'int8' đã DEPRECATED
            # và chính chúng phát cảnh báo. FP16 chỉ bật trên GPU (self.half=False khi CPU).
            kwargs["device"] = self.device
            kwargs["quantize"] = 16 if self.half else 32
        results = model(rgb_frame, **kwargs)[0]
        boxes, scores, class_ids = [], [], []
        for box in results.boxes:
            boxes.append(box.xyxy[0].cpu().numpy().astype(int).tolist())
            scores.append(float(box.conf[0]))
            class_ids.append(int(box.cls[0]))
        return boxes, scores, class_ids

    def process(self, rgb_frame, depth_map=None, frame_count=0):
        if not self.models:
            return rgb_frame.copy(), [], {
                'detected_objects': 0, 'detected_vehicles': 0,
                'detected_vrus': 0, 'traffic_lights': 0, 'stop_signs': 0,
                'perception_fallback': 'lidar_only_safety',
                'provenance_error': '; '.join(self.provenance_errors)}
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
                    detection = {
                        'bbox': (x1, y1, x2, y2), 'class': label,
                        'class_id': all_cls[i], 'confidence': conf}
                    if label == 'TrafficLight':
                        height, width = rgb_frame.shape[:2]
                        cx1, cy1 = max(0, min(width, x1)), max(0, min(height, y1))
                        cx2, cy2 = max(0, min(width, x2)), max(0, min(height, y2))
                        crop = rgb_frame[cy1:cy2, cx1:cx2]
                        raw_state = self.light_classifier.predict(crop)
                        key = (int(((x1 + x2) * 0.5) // 48),
                               int(((y1 + y2) * 0.5) // 48))
                        detection['traffic_light_state_raw'] = raw_state
                        detection['traffic_light_state'] = self.light_voter.update(
                            key, raw_state, frame_count)
                    detections.append(detection)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        vehicle_classes = {'Car', 'Motorcycle', 'Bus', 'Truck'}
        vru_classes = {'Person', 'Bicycle'}
        metrics = {
            'detected_objects': len(detections),
            'detected_vehicles': sum(d['class'] in vehicle_classes for d in detections),
            'detected_vrus': sum(d['class'] in vru_classes for d in detections),
            'traffic_lights': sum(d['class'] == 'TrafficLight' for d in detections),
            'stop_signs': sum(d['class'] == 'StopSign' for d in detections),
        }
        return annotated_frame, detections, metrics

    def warmup(self, width=960, height=540, runs=2):
        """Materialize CUDA kernels before KPI collection; returns warm-up ms."""
        if not self.models:
            return []
        sample = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        timings = []
        for index in range(max(1, int(runs))):
            started = time.perf_counter()
            self.process(sample, None, -index - 1)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            timings.append(round((time.perf_counter() - started) * 1000.0, 2))
        print(f"[YOLO] warm-up ms: {timings}")
        return timings
