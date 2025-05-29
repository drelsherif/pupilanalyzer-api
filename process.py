import cv2
import numpy as np

def analyze_pupil_response(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_ellipse = None
    best_area = 0
    best_circularity = 0

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            area = np.pi * (MA / 2) * (ma / 2)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

            if 100 < area < 3000 and circularity > 0.7:
                if area > best_area:
                    best_ellipse = ellipse
                    best_area = area
                    best_circularity = circularity

    if best_ellipse:
        (x, y), (MA, ma), angle = best_ellipse
        return {
            "pupil_area": round(best_area, 2),
            "center": [round(x, 2), round(y, 2)],
            "axes": [round(MA, 2), round(ma, 2)],
            "angle": round(angle, 2),
            "circularity": round(best_circularity, 3)
        }
    else:
        return {
            "pupil_area": 0,
            "center": None,
            "axes": None,
            "angle": None,
            "circularity": 0
        }
