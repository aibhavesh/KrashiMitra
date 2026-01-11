import cv2
import numpy as np

def detect_salinity(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white = np.sum(gray > 200)
    total = gray.size

    ratio = white / total

    if ratio > 0.15:
        return "high", ratio
    elif ratio > 0.05:
        return "medium", ratio
    else:
        return "low", ratio
