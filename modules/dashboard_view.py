"""Bảng điều khiển (Dashboard) trực quan ghép cạnh khung hình camera.

Trái: camera + bounding box + làn. Phải: panel thông số + bản đồ BEV.
Hiển thị: tốc độ, trạng thái FSM, trạng thái Fallback L3 + ODD + thời tiết, nguy
cơ phía trước (khoảng cách/TTC), số phương tiện, số va chạm, FPS, và bản đồ
nhìn-từ-trên có VẼ QUỸ ĐẠO DỰ ĐOÁN của từng track.
"""
# fmt: off
# isort: skip_file
import cv2
import numpy as np

# Bảng màu (BGR)
C_PANEL = (22, 22, 26)
C_ACCENT = (0, 200, 255)
C_WHITE = (240, 240, 240)
C_GRAY = (150, 150, 150)
C_GREEN = (90, 220, 120)
C_ORANGE = (0, 165, 255)
C_RED = (40, 40, 255)
C_CYAN = (230, 230, 60)


class DashboardView:
    def __init__(self, panel_width=380, speed_max_kmh=80.0, lane_half_m=1.75,
                 bev_range_m=40.0, bev_lat_m=12.0):
        self.pw = panel_width
        self.speed_max = speed_max_kmh
        self.lane_half = lane_half_m
        self.bev_range = bev_range_m
        self.bev_lat = bev_lat_m

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fsm_color(state):
        if 'BRAKE' in state:
            return C_RED
        if state in ('NORMAL', 'AUTOPILOT RUNNING', ''):
            return C_GREEN
        return C_ORANGE

    @staticmethod
    def _l3_color(state):
        if state in ('MRM_EXECUTING', 'SAFE_STOP'):
            return C_RED
        if state in ('TAKEOVER_REQUEST', 'DEGRADED'):
            return C_ORANGE
        return C_GREEN  # L3_ACTIVE

    @staticmethod
    def _bar(canvas, x, y, w, h, frac, color, bg=(60, 60, 60)):
        frac = max(0.0, min(1.0, frac))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + int(w * frac), y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 90), 1)

    @staticmethod
    def _chip(canvas, x, y, w, text, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + 30), color, -1)
        cv2.putText(canvas, text, (x + 10, y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    # ------------------------------------------------------------------ #
    def render(self, frame, metrics, obstacles=None, tracks=None):
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + self.pw, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = C_PANEL
        cv2.line(canvas, (w, 0), (w, h), C_ACCENT, 2)

        x0 = w + 24
        inner = self.pw - 48
        cv2.putText(canvas, "ADAS DASHBOARD", (x0, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_ACCENT, 2)

        # --- Tốc độ ---
        speed = float(metrics.get('speed', 0.0))
        cv2.putText(canvas, "EGO SPEED", (x0, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)
        cv2.putText(canvas, f"{speed:4.0f}", (x0, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.15, C_WHITE, 3)
        cv2.putText(canvas, "km/h", (x0 + 120, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)
        self._bar(canvas, x0, 114, inner, 10, speed / self.speed_max, C_ACCENT)

        # --- Chip FSM + chip L3 ---
        fsm = metrics.get('fsm_state', 'NORMAL')
        self._chip(canvas, x0, 132, inner, f"AEB: {fsm}", self._fsm_color(fsm))
        l3 = metrics.get('l3_state', 'L3_ACTIVE')
        self._chip(canvas, x0, 168, inner, f"L3: {l3}", self._l3_color(l3))

        # --- ODD + thời tiết ---
        odd = metrics.get('odd_state', 'NORMAL')
        vis = metrics.get('visibility_m')
        mu = metrics.get('mu')
        wline = f"ODD:{odd}"
        if vis is not None:
            wline += f"  vis {vis:.0f}m"
        if mu is not None:
            wline += f"  u{mu:.2f}"
        cv2.putText(canvas, wline, (x0, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self._l3_color(odd if odd in ('DEGRADED',) else l3), 1)

        # --- Nguy cơ phía trước ---
        cv2.putText(canvas, "DANGER AHEAD", (x0, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)
        threat = metrics.get('threat') or {}
        dist = threat.get('distance_m')
        if dist is not None:
            ttc = threat.get('ttc_s')
            ttc_txt = f"{ttc:.1f}s" if ttc is not None else "--"
            cv2.putText(canvas, f"{threat.get('label', 'obstacle')}  {dist:.1f} m",
                        (x0, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 2)
            cv2.putText(canvas, f"TTC: {ttc_txt}", (x0, 296),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self._fsm_color(fsm), 2)
            self._bar(canvas, x0, 306, inner, 10, 1.0 - min(1.0, dist / self.bev_range),
                      self._fsm_color(fsm))
        else:
            cv2.putText(canvas, "clear", (x0, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_GREEN, 2)

        # --- Số liệu đếm ---
        y = 344
        collisions = metrics.get('collisions', 0)
        rows = [
            ("VEHICLES", str(metrics.get('vehicles', 0)), C_WHITE),
            ("COLLISIONS", str(collisions), C_RED if collisions > 0 else C_GREEN),
            ("FPS", f"{metrics.get('fps', 0.0):.1f}", C_WHITE),
        ]
        if metrics.get('rl_target_kmh') is not None:
            rows.insert(2, (f"RL CRUISE (rho {metrics.get('density', 0.0):.1f})",
                            f"{metrics['rl_target_kmh']:.0f} km/h", C_ACCENT))
        for lbl, val, col in rows:
            cv2.putText(canvas, lbl, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GRAY, 1)
            cv2.putText(canvas, val, (x0 + 250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)
            y += 28

        # --- BEV ---
        bev_top = y + 8
        bev_size = min(inner, h - bev_top - 18)
        if bev_size > 80:
            bx = w + (self.pw - bev_size) // 2
            by = h - bev_size - 18
            self._draw_bev(canvas, bx, by, bev_size, obstacles or [], tracks or [])

        return canvas

    # ------------------------------------------------------------------ #
    def _draw_bev(self, canvas, bx, by, size, obstacles, tracks):
        cv2.rectangle(canvas, (bx, by), (bx + size, by + size), (12, 12, 14), -1)
        cv2.rectangle(canvas, (bx, by), (bx + size, by + size), (70, 70, 70), 1)
        cv2.putText(canvas, "BEV + prediction", (bx + 6, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

        cx = bx + size // 2
        ego_y = by + size - 16
        scale_fwd = (size - 34) / self.bev_range
        scale_lat = (size / 2 - 8) / self.bev_lat

        def to_px(x_fwd, y_lat):
            return int(cx + y_lat * scale_lat), int(ego_y - x_fwd * scale_fwd)

        # Hành lang làn.
        lane_px = int(self.lane_half * scale_lat)
        cv2.line(canvas, (cx - lane_px, by + 22), (cx - lane_px, ego_y), (60, 60, 70), 1)
        cv2.line(canvas, (cx + lane_px, by + 22), (cx + lane_px, ego_y), (60, 60, 70), 1)
        for r in (10, 20, 30):
            ry = ego_y - int(r * scale_fwd)
            if ry > by + 20:
                cv2.line(canvas, (bx + 4, ry), (bx + size - 4, ry), (40, 40, 46), 1)
                cv2.putText(canvas, f"{r}m", (bx + 6, ry - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 90), 1)

        # Chấm cụm LiDAR (mờ).
        for obs in obstacles:
            x_fwd, y_lat = obs["centroid"][0], obs["centroid"][1]
            if 0 < x_fwd <= self.bev_range and abs(y_lat) <= self.bev_lat:
                px, py = to_px(x_fwd, y_lat)
                cv2.circle(canvas, (px, py), 2, (110, 110, 110), -1)

        # Track: chấm + ID + quỹ đạo dự đoán (2s).
        for tr in tracks:
            x_fwd, y_lat = tr.pos
            if not (0 < x_fwd <= self.bev_range and abs(y_lat) <= self.bev_lat):
                continue
            px, py = to_px(x_fwd, y_lat)
            in_path = abs(y_lat) < self.lane_half
            color = C_ORANGE if in_path else C_GREEN
            if in_path and x_fwd < 8:
                color = C_RED
            cv2.circle(canvas, (px, py), 4, color, -1)
            cv2.putText(canvas, f"{tr.id}", (px + 5, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)
            pts = [(px, py)]
            for (fx, fy) in tr.predict_path(horizon_s=2.0, step_s=0.5):
                if abs(fy) <= self.bev_lat:
                    pts.append(to_px(fx, fy))
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, C_CYAN, 1)

        cv2.drawMarker(canvas, (cx, ego_y), C_ACCENT, cv2.MARKER_TRIANGLE_UP, 14, 2)
