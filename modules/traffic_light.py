import cv2
import numpy as np

def classify_traffic_light(box_crop):
    """Phân loại trạng thái đèn giao thông (Đỏ, Vàng, Xanh) bằng heuristic HSV."""
    if box_crop.size == 0:
        return "unknown"
        
    hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    third = h // 3
    
    if third == 0:
        return "unknown"

    # Chia hộp thành 3 phần: Đỏ (trên), Vàng (giữa), Xanh (dưới)
    red_crop = hsv[0:third, :]
    yellow_crop = hsv[third:2*third, :]
    green_crop = hsv[2*third:, :]

    def get_brightness(crop):
        if crop.size == 0:
            return 0
        return np.mean(crop[:, :, 1].astype(float) * crop[:, :, 2].astype(float))

    scores = {
        "red": get_brightness(red_crop),
        "yellow": get_brightness(yellow_crop),
        "green": get_brightness(green_crop)
    }
    
    best_color = max(scores, key=scores.get)
    return best_color if scores[best_color] > 1000 else "off"