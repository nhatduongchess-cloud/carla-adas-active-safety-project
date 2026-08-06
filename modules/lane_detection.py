import cv2
import numpy as np

class LaneDetector:
    """Mô-đun trích xuất và làm mượt làn đường thông minh, loại bỏ nhiễu thị giác."""

    def __init__(self):
        self.blur_ksize = (5, 5)
        self.canny_low = 50
        self.canny_high = 150

    def process(self, frame):
        """Trả về khung đã chồng vạch làn (blend)."""
        line_image = self.line_overlay(frame)
        return cv2.addWeighted(frame, 0.85, line_image, 0.85, 0)

    def line_overlay(self, frame):
        """Chỉ trả về LỚP vạch làn (ảnh đen + các vạch) để tái sử dụng nhiều khung.

        Tách riêng phần Hough tốn kém: chinh.py dò làn mỗi N khung rồi CHỒNG lớp
        này lên mọi khung -> video vẫn mượt mà giảm tải Hough."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Định nghĩa vùng quan tâm (ROI hình thang tập trung vào mặt đường phía trước)
        polygons = np.array([
            [(int(width * 0.15), height), 
             (int(width * 0.85), height), 
             (int(width * 0.52), int(height * 0.62)), 
             (int(width * 0.48), int(height * 0.62))]
        ], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygons, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Lọc Hough Lines với giới hạn khoảng cách và góc hợp lý
        lines = cv2.HoughLinesP(
            masked_edges, rho=1, theta=np.pi/180, threshold=40,
            minLineLength=60, maxLineGap=120
        )

        line_image = np.zeros_like(frame)
        drivable_polygon = np.zeros_like(frame)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                # Lọc bỏ các đường nằm ngang hoặc đứng quá mức
                if abs(slope) < 0.3 or abs(slope) > 3.5:
                    continue
                
                if slope < 0:  # Làn bên trái
                    left_lines.append((x1, y1, x2, y2))
                else:          # Làn bên phải
                    right_lines.append((x1, y1, x2, y2))

            # Vẽ đường biên làn đường thanh lịch (Màu xanh lá sáng)
            for x1, y1, x2, y2 in left_lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 120), 4)
            for x1, y1, x2, y2 in right_lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 120), 4)

        return line_image