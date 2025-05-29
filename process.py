import cv2
import numpy as np

def analyze_pupil_response(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pupil_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 3000:
            pupil_area = area
            break

    return {
        "pupil_area": pupil_area
    }
