import os
import glob
import cv2
import numpy as np

# Configurable Parameters (Lab Constants)
RATIO_THRESH = 0.75         # Lowe's ratio test threshold
RANSAC_REPROJ_THRESH = 3.0   # Max allowed reprojection error in pixels
MIN_INLIERS = 10             # Minimum inliers required to accept match

def load_images_from_folder(folder_path):
    extensions = ('*.jpg', '*.png', '*.ppm', '*.jpeg', '*.JPG')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files = sorted(files)
    
    images = []
    filenames = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(f))
    return images, filenames

def get_pairwise_transform(img1, img2):
    """Computes a 3x3 rigid/affine transform mapping img2 points to img1 frame."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m_tuple in matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < RATIO_THRESH * n.distance:
                good.append(m)

    if len(good) < MIN_INLIERS:
        return None, len(good)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate 2D Similarity/Affine transform (Translation + Rotation + Uniform Scale)
    # This prevents perspective foreshortening/blurring at far ends
    M, mask = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESH)
    inliers = int(np.sum(mask)) if mask is not None else 0

    if M is None or inliers < MIN_INLIERS:
        return None, inliers

    # Convert 2x3 Affine matrix to 3x3 Homogeneous transformation matrix
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M

    return H, inliers

def stitch_n_images(folder_path):
    images, filenames = load_images_from_folder(folder_path)
    num_imgs = len(images)
    
    if num_imgs == 0:
        print("No images found in the specified folder.")
        return

    print(f"Loaded {num_imgs} images.")

    # Step 1: Compute adjacent transforms H_{i, i+1} (transforms i+1 -> i)
    H_adjacent = {}
    valid_mask = [True] * num_imgs

    for i in range(num_imgs - 1):
        name1, name2 = filenames[i], filenames[i+1]
        H, inliers = get_pairwise_transform(images[i], images[i+1])
        
        if H is not None:
            H_adjacent[i] = H
            print(f"[MATCH OK] ({name1} <-> {name2}) connected with {inliers} inliers.")
        else:
            print(f"[REJECT IMAGE] {name2} rejected — insufficient inliers ({inliers}).")
            valid_mask[i+1] = False

    accepted_images = [img for idx, img in enumerate(images) if valid_mask[idx]]
    accepted_names = [name for idx, name in enumerate(filenames) if valid_mask[idx]]
    N = len(accepted_images)

    if N == 0:
        print("All images were rejected!")
        return

    # Step 2: Choose Anchor (Middle image)
    anchor_idx = N // 2
    print(f"\n---> Anchor Image: {accepted_names[anchor_idx]} (Index {anchor_idx})")

    # Step 3: Chain Transforms relative to Anchor
    H_global = [None] * N
    H_global[anchor_idx] = np.eye(3, dtype=np.float32)

    # Chain backwards from anchor to index 0
    for i in range(anchor_idx - 1, -1, -1):
        H_global[i] = H_global[i+1] @ np.linalg.inv(H_adjacent[i])

    # Chain forwards from anchor to index N-1
    for i in range(anchor_idx + 1, N):
        H_global[i] = H_global[i-1] @ H_adjacent[i-1]

    # Step 4: Compute Bounding Box across all projected corners
    all_corners = []
    for i in range(N):
        h, w = accepted_images[i].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H_global[i])
        all_corners.append(projected)

    all_corners_stacked = np.concatenate(all_corners, axis=0)
    [x_min, y_min] = np.int32(all_corners_stacked.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners_stacked.max(axis=0).ravel() + 0.5)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Translation matrix to shift canvas into positive coordinates
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float32)

    # Step 5: Compositing onto single canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i in range(N):
        final_H = T @ H_global[i]
        warped = cv2.warpPerspective(accepted_images[i], final_H, (canvas_w, canvas_h))
        
        # Overlay pixels
        mask = (warped > 0)
        canvas[mask] = warped[mask]

    return canvas

if __name__ == "__main__":
    FOLDER_PATH = "VPG/S01"  # Update this path to your image folder
    
    panorama = stitch_n_images(FOLDER_PATH)

    if panorama is not None:
        cv2.namedWindow("Crisp Panorama (No Blur)", cv2.WINDOW_NORMAL)
        cv2.imshow("Crisp Panorama (No Blur)", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()