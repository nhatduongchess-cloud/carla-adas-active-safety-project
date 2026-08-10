# fmt: off
# isort: skip_file
"""
Biên dịch YOLO sang TensorRT (FP16) — hoặc ONNX (fallback) — để tăng tốc suy luận
2-4x mà GIỮ NGUYÊN độ chính xác (FP16, KHÔNG dùng INT8). Chạy MỘT LẦN:

    .venvCarLa\\Scripts\\python.exe export_models.py

Sau khi xuất xong, object_tracking.py tự động dùng file .engine/.onnx (nhờ
config.YOLO_USE_OPTIMIZED=True) — không cần sửa gì thêm.

Yêu cầu cài đặt (nếu thiếu):
  - TensorRT (nhanh nhất):  pip install tensorrt
  - Hoặc ONNX Runtime GPU:  pip install onnx onnxruntime-gpu
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ultralytics import YOLO

MODELS = ["yolov8n.pt", "yolov10n.pt"]   # khớp config.YOLO_MODEL_A / _B
IMGSZ = 640


def export(pt, fmt):
    print(f"\n[Export] {pt}  ->  {fmt}  (FP16, imgsz={IMGSZ}) ...")
    # Ultralytics 8.4: 'quantize=16' (FP16) thay cho 'half=True' đã deprecated.
    YOLO(pt).export(format=fmt, quantize=16, imgsz=IMGSZ, device=0)


def main():
    for pt in MODELS:
        if not os.path.exists(pt):
            print(f"[Bỏ qua] Không thấy {pt}")
            continue
        base = os.path.splitext(pt)[0]
        # 1) Thử TensorRT (nhanh nhất)
        try:
            export(pt, "engine")
            print(f"[OK] {pt}  ->  {base}.engine  (TensorRT FP16)")
            continue
        except Exception as e:
            print(f"[TensorRT thất bại] {e}\n     -> thử ONNX...")
        # 2) Fallback ONNX
        try:
            export(pt, "onnx")
            print(f"[OK] {pt}  ->  {base}.onnx  (ONNX FP16)")
        except Exception as e2:
            print(f"[ONNX cũng thất bại] {e2}")
            print("     Cài đặt:  pip install tensorrt   HOẶC   pip install onnx onnxruntime-gpu")

    print("\n[Xong] Chạy lại demo bình thường — pipeline tự dùng mô hình tối ưu nếu có.")


if __name__ == "__main__":
    main()
