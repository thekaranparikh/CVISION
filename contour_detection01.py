import cv2
import numpy as np

img = cv2.imread("sipi-dataset/sequences/6.1.01.tiff")   # any image with one clear foreground object
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segment foreground from background
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Clean up the mask: opening removes small noise, closing seals small gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours on the cleaned mask
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

result = img.copy()
cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

# Polygon approximation: simplify the contour to fewer vertices
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
cv2.drawContours(result, [approx], -1, (0, 0, 255), 2)

cv2.imshow("Contour (green) vs Approximated Polygon (red)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()