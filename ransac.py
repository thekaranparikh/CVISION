import cv2
import numpy as np

img1 = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")
img2 = cv2.imread("hpatches-sequences-release/v_wall/2.ppm")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
mask = mask.ravel().tolist()

num_inliers = sum(mask)
print(f"{num_inliers} inliers out of {len(good_matches)} matches "
      f"({num_inliers / len(good_matches) * 100:.1f}%)")


def draw_matches_colored(img_a, kp_a, img_b, kp_b, matches, inlier_mask):
    """Place img_a and img_b side by side and draw each match:
    green if it's an inlier, red if it's an outlier."""
    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_a
    canvas[:h2, w1:w1 + w2] = img_b

    for m, is_inlier in zip(matches, inlier_mask):
        pt1 = tuple(np.round(kp_a[m.queryIdx].pt).astype(int))
        pt2 = tuple((np.round(kp_b[m.trainIdx].pt).astype(int) + np.array([w1, 0])))
        color = (0, 255, 0) if is_inlier else (0, 0, 255)  # green = inlier, red = outlier
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    return canvas


result = draw_matches_colored(img1, kp1, img2, kp2, good_matches, mask)

cv2.namedWindow("RANSAC: Inliers (green) vs Outliers (red)", cv2.WINDOW_NORMAL)
cv2.imshow("RANSAC: Inliers (green) vs Outliers (red)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()