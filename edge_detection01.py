import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")  # pick any image with clear edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)  # mild denoising before edge detection

# --- Sobel ---
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# --- Laplacian of Gaussian (LoG): Gaussian blur, then Laplacian ---
blurred_for_log = cv2.GaussianBlur(gray, (5, 5), 0)
log = cv2.Laplacian(blurred_for_log, cv2.CV_64F, ksize=3)
log = cv2.convertScaleAbs(log)

# --- Canny ---
canny = cv2.Canny(gray, 100, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
titles = ["Original (Gray)", "Sobel X", "Sobel Y", "Sobel Magnitude", "Laplacian of Gaussian", "Canny"]
images = [gray, cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y), sobel_magnitude, log, canny]

for ax, title, im in zip(axes.ravel(), titles, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()