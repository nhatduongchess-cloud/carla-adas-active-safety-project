"""Dò làn đường: Canny + Hough, nhưng KHÔNG chỉ vẽ đoạn thô.

Nâng cấp so với bản Hough-thuần:
  1) GỘP nhiều đoạn Hough thành MỘT làn trái + MỘT làn phải (bình phương tối thiểu
     theo x = m·y + b -> ổn định với đường gần thẳng đứng).
  2) NGOẠI SUY mỗi làn từ đáy khung tới đỉnh ROI -> vạch liền, không đứt đoạn.
  3) LÀM MƯỢT THEO THỜI GIAN (EMA) tham số (m, b) -> vạch không "nhảy" giữa các
     khung khi Hough chập chờn.
  4) Tô VÙNG LÁI ĐƯỢC (drivable area) giữa hai làn + tính ĐỘ LỆCH TÂM LÀN
     (lane_offset_m) để lớp điều khiển/HUD dùng (cảnh báo chệch làn).

Giữ nguyên giao diện cũ: process(frame) và line_overlay(frame).
"""
import cv2
import numpy as np


class LaneDetector:
    """Trích xuất, ngoại suy và làm mượt làn đường; kèm ước lượng lệch tâm làn."""

    def __init__(self, ema: float = 0.35, lane_width_m: float = 3.5):
        self.blur_ksize = (5, 5)
        self.canny_low = 50
        self.canny_high = 150
        # Hệ số làm mượt EMA cho (slope, intercept) của mỗi làn (0=đóng băng, 1=không mượt).
        self.ema = ema
        self.lane_width_m = lane_width_m

        # Tham số làn đã làm mượt: (m, b) với x = m·y + b (toạ độ ảnh).
        self._left = None
        self._right = None

        # Kết quả suy ra cho lớp trên dùng.
        self.lane_offset_m = 0.0     # + = xe LỆCH PHẢI so với tâm làn, - = lệch trái
        self.lane_detected = False

    # ---------------------------------------------------------------------- #
    def process(self, frame):
        """Trả về khung đã CHỒNG vạch làn + vùng lái được (blend)."""
        overlay = self.line_overlay(frame)
        return cv2.addWeighted(frame, 0.85, overlay, 0.85, 0)

    def _roi_polygon(self, width, height):
        return np.array([[
            (int(width * 0.10), height),
            (int(width * 0.90), height),
            (int(width * 0.55), int(height * 0.60)),
            (int(width * 0.45), int(height * 0.60)),
        ]], dtype=np.int32)

    @staticmethod
    def _fit_side(segments):
        """Bình phương tối thiểu x = m·y + b từ các điểm đầu-cuối đoạn Hough.

        Fit theo y (không theo x) để bền với làn gần thẳng đứng. Trả (m, b) hoặc None.
        """
        if not segments:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in segments:
            xs += [x1, x2]
            ys += [y1, y2]
        if len(set(ys)) < 2:
            return None
        m, b = np.polyfit(ys, xs, 1)   # x = m·y + b
        return float(m), float(b)

    def _smooth(self, prev, new):
        """EMA cho (m, b). Nếu khung này mất làn -> giữ giá trị cũ (không giật)."""
        if new is None:
            return prev
        if prev is None:
            return new
        a = self.ema
        return ((1 - a) * prev[0] + a * new[0], (1 - a) * prev[1] + a * new[1])

    # ---------------------------------------------------------------------- #
    def line_overlay(self, frame):
        """Trả về LỚP overlay (nền đen + vạch làn + vùng lái được) cùng kích thước.

        Đồng thời CẬP NHẬT self.lane_offset_m / self.lane_detected để lớp trên đọc.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self._roi_polygon(width, height), 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, rho=1, theta=np.pi / 180, threshold=40,
                                minLineLength=60, maxLineGap=120)

        left_seg, right_seg = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3 or abs(slope) > 3.5:   # bỏ đoạn ngang/dọc bất thường
                    continue
                (left_seg if slope < 0 else right_seg).append((x1, y1, x2, y2))

        # Fit + làm mượt EMA từng làn.
        self._left = self._smooth(self._left, self._fit_side(left_seg))
        self._right = self._smooth(self._right, self._fit_side(right_seg))

        overlay = np.zeros_like(frame)
        y_bottom, y_top = height, int(height * 0.62)

        def _pts(side):
            if side is None:
                return None
            m, b = side
            return (int(m * y_bottom + b), y_bottom), (int(m * y_top + b), y_top)

        lp, rp = _pts(self._left), _pts(self._right)

        # Vùng lái được (tô xanh mờ giữa hai làn) khi có ĐỦ cả hai làn.
        self.lane_detected = lp is not None and rp is not None
        if self.lane_detected:
            poly = np.array([[lp[0], lp[1], rp[1], rp[0]]], dtype=np.int32)
            drivable = np.zeros_like(frame)
            cv2.fillPoly(drivable, poly, (0, 120, 0))
            overlay = cv2.addWeighted(overlay, 1.0, drivable, 0.35, 0)

            # Lệch tâm làn: so tâm hai làn ở ĐÁY khung với tâm ảnh, quy sang mét.
            lane_center_px = (lp[0][0] + rp[0][0]) / 2.0
            lane_px_width = max(1.0, abs(rp[0][0] - lp[0][0]))
            m_per_px = self.lane_width_m / lane_px_width
            self.lane_offset_m = round((width / 2.0 - lane_center_px) * m_per_px, 2)

        # Vẽ vạch làn (xanh lá sáng).
        for p in (lp, rp):
            if p is not None:
                cv2.line(overlay, p[0], p[1], (0, 255, 120), 6)

        return overlay
