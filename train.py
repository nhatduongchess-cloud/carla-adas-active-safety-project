from ultralytics import YOLO
import torch


def main():
    # 1. Kiểm tra tài nguyên tính toán
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Hệ Thống] Bắt đầu huấn luyện trên: {device}")

    # 2. Khởi tạo mô hình pre-trained YOLOv8 Nano (tối ưu cho FPS cao)
    model = YOLO('yolov8n.pt')

    # 3. Khởi chạy quá trình huấn luyện
    # Lưu ý: Các tham số dưới đây đã được tinh chỉnh cho phần cứng cá nhân
    results = model.train(
        data='./dataset/data.yaml',  # Trỏ tới tệp cấu hình vừa sửa
        # Chạy 50 chu kỳ học (có thể tăng lên 100 nếu cần)
        epochs=50,
        imgsz=640,                   # Kích thước ảnh resize
        # Kích thước lô (Giảm xuống 8 hoặc 4 nếu GPU báo lỗi Out of Memory)
        batch=16,
        device=device,
        workers=2                    # Số luồng CPU hỗ trợ tải dữ liệu
    )

    print("[Hệ Thống] Hoàn tất quá trình huấn luyện.")


if __name__ == '__main__':
    # Fix lỗi multiprocessing trên Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
