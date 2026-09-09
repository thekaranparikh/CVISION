"""
Lab 4 - HSV Color Histogram Baseline
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------------- Configuration ----------------
CALTECH_DIR = "256_ObjectCategories"
RESIZE_DIM = 128
MAX_IMAGES_PER_CLASS = 50
NUM_EPOCHS = 30
RANDOM_STATE = 42

# ---------------- 1) Data Loader ----------------
def load_caltech256_paths(root_dir, max_per_class):
    class_folders = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    label_names = [name.split(".", 1)[1] if "." in name else name for name in class_folders]

    rng = random.Random(RANDOM_STATE)
    filepaths, labels = [], []

    for label_id, folder in enumerate(class_folders):
        folder_path = os.path.join(root_dir, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(images)
        for fname in images[:max_per_class]:
            filepaths.append(os.path.join(folder_path, fname))
            labels.append(label_id)

    return filepaths, np.array(labels), label_names

print("Loading dataset paths...")
filepaths, labels, label_names = load_caltech256_paths(CALTECH_DIR, MAX_IMAGES_PER_CLASS)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, train_size=0.70, stratify=labels, random_state=RANDOM_STATE
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_STATE
)

# ---------------- 2) Extract HSV Color Histograms ----------------
def extract_hsv_histograms(paths):
    hsv_hists = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            # 4 HUE bins * 4 SATURATION bins * 3 VALUE bins = 48 total bins
            hsv_hists.append(np.zeros(48, dtype=np.float32))
            continue
        
        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Calculate 3D Histogram: 4 Hue, 4 Saturation, 3 Value
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 3], [0, 180, 0, 256, 0, 256]).flatten()
        
        # L2 Normalize
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
            
        hsv_hists.append(hist)
        
    return np.array(hsv_hists)

print("Extracting HSV Color Histograms...")
train_hsv = extract_hsv_histograms(train_paths)
val_hsv = extract_hsv_histograms(val_paths)
test_hsv = extract_hsv_histograms(test_paths)

# ---------------- 3) Standardize Features ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_hsv)
X_val = scaler.transform(val_hsv)
X_test = scaler.transform(test_hsv)

# ---------------- 4) Train SGD-SVM & Track Loss ----------------
def multiclass_hinge_loss(decision_values, true_labels, classes):
    y_binarized = np.array([[1 if c == label else -1 for c in classes] for label in true_labels])
    margins = np.clip(1 - y_binarized * decision_values, 0, None)
    return float(np.mean(margins))

svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
classes = np.unique(train_labels)

train_losses, val_losses = [], []

print("Training SGD-SVM classifier on HSV Histograms...")
for epoch in range(NUM_EPOCHS):
    svm.partial_fit(X_train, train_labels, classes=classes)
    t_loss = multiclass_hinge_loss(svm.decision_function(X_train), train_labels, classes)
    v_loss = multiclass_hinge_loss(svm.decision_function(X_val), val_labels, classes)
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} - Train Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f}")

# Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Hinge Loss")
plt.title("HSV Color Histogram Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("hsv_loss_curve.png")
plt.show()

# ---------------- 5) Evaluate & Save Artifacts ----------------
test_preds = svm.predict(X_test)
acc = accuracy_score(test_labels, test_preds)
print(f"\n==========================================")
print(f"Overall HSV Color Test Accuracy: {acc * 100:.2f}%")
print(f"==========================================\n")
print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

# Save artifacts
joblib.dump(svm, "hsv_svm.joblib")
joblib.dump(scaler, "hsv_scaler.joblib")
joblib.dump(label_names, "hsv_label_names.joblib")
print("Saved HSV pipeline artifacts successfully.")