import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

POSITIVE_DIR = "surface-crack-detection/Positive"
NEGATIVE_DIR = "surface-crack-detection/Negative"
SAMPLE_SIZE_PER_CLASS = 300  # Adjust as needed


def is_crack(image_path, thresh_type="adaptive", block_size=15, C=4, kernel_size=3, aspect_ratio_thresh=2.5, min_area=30) -> bool:
    """
    Classical Crack Detection Pipeline:
    1. Grayscale + Gaussian Blur
    2. Thresholding (Adaptive, Otsu, or Fixed)
    3. Morphological Cleaning (Opening)
    4. Contour Analysis (Aspect Ratio of bounding boxes)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # 1. Blur to smooth out surface noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Thresholding (Cracks are darker than surrounding concrete)
    if thresh_type == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
    elif thresh_type == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Morphological Cleanup (Opening removes isolated noise specks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. Contour Detection and Aspect Ratio filtering
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                # If a contour is sufficiently thin and long, flag as a crack
                if aspect_ratio >= aspect_ratio_thresh:
                    return True

    return False


def visualize_detection(image_path, thresh_type="adaptive", block_size=15, C=4, kernel_size=3, aspect_ratio_thresh=2.5, min_area=30):
    """Visualizes each stage of the classical computer vision pipeline in a GUI window."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. Thresholding
    if thresh_type == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
    elif thresh_type == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # 2. Morphological Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Contour Detection & Overlay Drawing
    output_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio >= aspect_ratio_thresh:
                    detected = True
                    # Draw detected crack bounding box in red
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(output_bgr, [box], 0, (0, 0, 255), 2)

    # Plot pipeline stages via Matplotlib window
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Original", "Thresholded", "Morph Cleaned", f"Detected: {detected}"]
    imgs = [img, binary, cleaned, cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)]

    for ax, t, im in zip(axes, titles, imgs):
        ax.imshow(im, cmap="gray" if len(im.shape) == 2 else None)
        ax.set_title(t)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate(positive_dir, negative_dir, sample_size, **params):
    random.seed(42)  # Fixed seed for reproducible sample selection
    positive_files = random.sample(os.listdir(positive_dir), sample_size)
    negative_files = random.sample(os.listdir(negative_dir), sample_size)

    tp = fp = tn = fn = 0

    for fname in positive_files:
        predicted = is_crack(os.path.join(positive_dir, fname), **params)
        if predicted:
            tp += 1
        else:
            fn += 1

    for fname in negative_files:
        predicted = is_crack(os.path.join(negative_dir, fname), **params)
        if predicted:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP={tp:3d} | FP={fp:3d} | TN={tn:3d} | FN={fn:3d}")
    print(f"Accuracy={accuracy:.3f} | Precision={precision:.3f} | Recall={recall:.3f}\n")
    return tp, fp, tn, fn


if __name__ == "__main__":
    print("--- Experiment 1: Global Otsu Threshold + Small Kernel (3x3) ---")
    evaluate(POSITIVE_DIR, NEGATIVE_DIR, SAMPLE_SIZE_PER_CLASS, 
             thresh_type="otsu", kernel_size=3, aspect_ratio_thresh=2.0)

    print("--- Experiment 2: Adaptive Threshold + Larger Morph Kernel (5x5) ---")
    evaluate(POSITIVE_DIR, NEGATIVE_DIR, SAMPLE_SIZE_PER_CLASS, 
             thresh_type="adaptive", block_size=25, C=5, kernel_size=5, aspect_ratio_thresh=3.0)

    print("--- Experiment 3: Adaptive Threshold + Fine Tuning ---")
    evaluate(POSITIVE_DIR, NEGATIVE_DIR, SAMPLE_SIZE_PER_CLASS, 
             thresh_type="adaptive", block_size=15, C=4, kernel_size=3, aspect_ratio_thresh=2.5)

    # Display GUI window for sample positive and negative images
    pos_sample = os.path.join(POSITIVE_DIR, os.listdir(POSITIVE_DIR)[0])
    neg_sample = os.path.join(NEGATIVE_DIR, os.listdir(NEGATIVE_DIR)[0])

    print("Opening GUI visualization for a POSITIVE sample...")
    visualize_detection(pos_sample, thresh_type="adaptive", block_size=25, C=5, kernel_size=5, aspect_ratio_thresh=3.0)

    print("Opening GUI visualization for a NEGATIVE sample...")
    visualize_detection(neg_sample, thresh_type="adaptive", block_size=25, C=5, kernel_size=5, aspect_ratio_thresh=3.0)