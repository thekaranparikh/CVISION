import cv2

img = cv2.imread("sipi-dataset/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

def nothing(x):
    pass

cv2.namedWindow("Canny Threshold Explorer", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Low Threshold", "Canny Threshold Explorer", 50, 500, nothing)
cv2.createTrackbar("High Threshold", "Canny Threshold Explorer", 150, 500, nothing)

while True:
    low = cv2.getTrackbarPos("Low Threshold", "Canny Threshold Explorer")
    high = cv2.getTrackbarPos("High Threshold", "Canny Threshold Explorer")

    edges = cv2.Canny(gray, low, high)
    cv2.imshow("Canny Threshold Explorer", edges)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()