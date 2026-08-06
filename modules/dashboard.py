import cv2


def draw_dashboard(canvas, metrics, im_width, total_width, im_height):
    cv2.rectangle(canvas, (im_width, 0),
                  (total_width, im_height), (15, 15, 15), -1)
    cv2.line(canvas, (im_width, 0), (im_width, im_height), (0, 200, 255), 2)

    cv2.putText(canvas, "PERCEPTION DASHBOARD", (im_width + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Thêm thông số Speed (km/h) vào danh sách hiển thị
    labels = [
        "Frame", f"{metrics.get('frame', 0)} / 500",
        "FPS", f"{metrics.get('fps', 0.0):.1f}",
        "Speed (km/h)", f"{metrics.get('speed', 0.0):.1f}",
        "Vehicles (now)", f"{metrics.get('vehicles_now', 0)}",
        "Pedestrians (now)", f"{metrics.get('pedestrians_now', 0)}",
        "Tracked objects", f"{metrics.get('tracked_objects', 0)}",
        "Unique vehicles", f"{metrics.get('unique_vehicles', 0)}",
        "Traffic light", f"{metrics.get('traffic_light', 'None')}",
        "Signs seen", f"{metrics.get('signs_seen', 0)}",
        "Lane status", f"{metrics.get('lane_status', 'Normal')}",
        "Nearest object", f"{metrics.get('nearest_object', 'None')}"
    ]

    start_y = 80
    for i in range(0, len(labels), 2):
        cv2.putText(canvas, labels[i], (im_width + 20, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, labels[i+1], (im_width + 20, start_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        start_y += 45  # Điều chỉnh bước nhảy dọc để vừa vặn khi tăng số lượng trường hiển thị

    if metrics.get('collision_risk', False):
        cv2.rectangle(canvas, (im_width + 15, im_height - 70),
                      (total_width - 15, im_height - 20), (0, 0, 255), -1)
        cv2.putText(canvas, "! COLLISION RISK !", (im_width + 45, im_height - 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
