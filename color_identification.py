import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("test_image1.jpg")
cv2.waitKey(0)

img = np.copy(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_boundaries = {
    "red": ([161, 155, 84], [179, 255, 255]),
    "blue": ([94, 80, 2], [126, 255, 255]),
    "yellow": ([22, 60, 200], [60, 255, 255]),
}

for color_name, (lower, upper) in color_boundaries.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    #    mask = cv2.erode(mask, None, iterations=2)
    #    mask = cv2.dilate(mask, None, iterations=2)

    output = cv2.bitwise_and(hsv, hsv, mask=mask)
    if mask.any():
        print(f"{color_name}")
