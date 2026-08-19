import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=2)
dilated = cv2.dilate(binary, kernel, iterations=2)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titles = ["Binary (Otsu)", "Erosion", "Dilation", "Opening", "Closing"]
images = [binary, eroded, dilated, opened, closed]

for ax, title, im in zip(axes, titles, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()