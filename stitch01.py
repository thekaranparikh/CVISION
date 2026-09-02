import cv2
import numpy as np

# Configurable Parameters
RATIO_THRESH = 0.75   # Lowe's ratio test threshold
RANSAC_REPROJ_THRESH = 5.0 # Max allowed reprojection error in pixels
MIN_INLIERS = 10      # Minimum inliers required to accept stitching

def stitch_two_images(img1_path, img2_path):
    # 1. Load Images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both input images.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. Extract SIFT Features
    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. Match Features using FLANN + Lowe's Ratio Test
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < RATIO_THRESH * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")
    if len(good_matches) < MIN_INLIERS:
        print(f"REJECTED: Only {len(good_matches)} good matches (Minimum: {MIN_INLIERS}).")
        return None

    # 4. Compute Homography via RANSAC (transforms img2 to img1's coordinate system)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    inlier_count = np.sum(mask) if mask is not None else 0

    print(f"RANSAC Inliers: {inlier_count} / {len(good_matches)}")
    if inlier_count < MIN_INLIERS or H is None:
        print(f"REJECTED: Only {inlier_count} RANSAC inliers (Minimum: {MIN_INLIERS}).")
        return None

    # 5. Compute Dynamic Canvas Size to Prevent Clipping
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corner points of img2
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

    # Corner points of img1 (reference canvas)
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Find total bounding box
    all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift output to positive pixel coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float32)

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # 6. Warp and Composite Images
    # Warp img2 into translated coordinate space
    result = cv2.warpPerspective(img2, translation @ H, (canvas_width, canvas_height))
    
    # Copy reference img1 into its translated position
    result[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

    return result

if __name__ == "__main__":
    panorama = stitch_two_images("VPG/S01/01.jpg", "VPG/S01/02.jpg")
    if panorama is not None:
        cv2.imwrite("stitched_result.jpg", panorama)
        cv2.imshow("Stitched Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()