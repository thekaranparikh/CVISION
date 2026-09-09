"""
Lab 4 - Task A: Complete Feature Fusion (SIFT-BoVW + HSV + LBP)
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

# ---------------- Configuration ----------------
CALTECH_DIR = "256_ObjectCategories"
VOCAB_SIZE = 64
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

# ---------------- 2) Extract Features ----------------
sift = cv2.SIFT_create()

def extract_all_features(paths):
    sift_list, hsv_list, lbp_list = [], [], []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            sift_list.append(None)
            hsv_list.append(np.zeros(48, dtype=np.float32))
            lbp_list.append(np.zeros(26, dtype=np.float32))
            continue

        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)

        # Feature 1: SIFT
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        sift_list.append(des)

        # Feature 2: HSV Color
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 3], [0, 180, 0, 256, 0, 256]).flatten()
        norm_hsv = np.linalg.norm(hsv_hist)
        if norm_hsv > 0:
            hsv_hist /= norm_hsv
        hsv_list.append(hsv_hist)

        # Feature 3: LBP Texture
        lbp = local_binary_pattern(gray, P=8, R=1, method="nri_uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27))
        lbp_hist = lbp_hist.astype(np.float32)
        norm_lbp = np.linalg.norm(lbp_hist)
        if norm_lbp > 0:
            lbp_hist /= norm_lbp
        lbp_list.append(lbp_hist)

    return sift_list, np.array(hsv_list), np.array(lbp_list)

print("Extracting features (Train)...")
train_sift, train_hsv, train_lbp = extract_all_features(train_paths)
print("Extracting features (Val)...")
val_sift, val_hsv, val_lbp = extract_all_features(val_paths)
print("Extracting features (Test)...")
test_sift, test_hsv, test_lbp = extract_all_features(test_paths)

# ---------------- 3) SIFT Visual Vocabulary & BoVW ----------------
all_train_des = np.vstack([d for d in train_sift if d is not None])
print(f"Fitting SIFT vocabulary (k={VOCAB_SIZE})...")
vocabulary = KMeans(n_clusters=VOCAB_SIZE, n_init=4, random_state=RANDOM_STATE)
vocabulary.fit(all_train_des)

def build_bovw_matrix(sift_list, vocab, vocab_size):
    bovw = []
    for des in sift_list:
        if des is None or len(des) == 0:
            bovw.append(np.zeros(vocab_size, dtype=np.float32))
        else:
            words = vocab.predict(des)
            hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            bovw.append(hist)
    return np.array(bovw)

train_bovw = build_bovw_matrix(train_sift, vocabulary, VOCAB_SIZE)
val_bovw = build_bovw_matrix(val_sift, vocabulary, VOCAB_SIZE)
test_bovw = build_bovw_matrix(test_sift, vocabulary, VOCAB_SIZE)

# ---------------- 4) Independent Scaling & Concatenation ----------------
scaler_bovw = StandardScaler()
scaler_hsv = StandardScaler()
scaler_lbp = StandardScaler()

X_train_bovw = scaler_bovw.fit_transform(train_bovw)
X_val_bovw = scaler_bovw.transform(val_bovw)
X_test_bovw = scaler_bovw.transform(test_bovw)

X_train_hsv = scaler_hsv.fit_transform(train_hsv)
X_val_hsv = scaler_hsv.transform(val_hsv)
X_test_hsv = scaler_hsv.transform(test_hsv)

X_train_lbp = scaler_lbp.fit_transform(train_lbp)
X_val_lbp = scaler_lbp.transform(val_lbp)
X_test_lbp = scaler_lbp.transform(test_lbp)

# Concatenate into unified feature matrices
X_train = np.hstack([X_train_bovw, X_train_hsv, X_train_lbp])
X_val = np.hstack([X_val_bovw, X_val_hsv, X_val_lbp])
X_test = np.hstack([X_test_bovw, X_test_hsv, X_test_lbp])

print(f"Total Combined Feature Vector Size: {X_train.shape[1]} dimensions")

# ---------------- 5) Classifier Training & Loss Tracking ----------------
def multiclass_hinge_loss(decision_values, true_labels, classes):
    y_binarized = np.array([[1 if c == label else -1 for c in classes] for label in true_labels])
    margins = np.clip(1 - y_binarized * decision_values, 0, None)
    return float(np.mean(margins))

svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
classes = np.unique(train_labels)

train_losses, val_losses = [], []

print("Training Fused Feature Classifier...")
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
plt.title("Fused Features (SIFT + HSV + LBP) Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("fused_loss_curve.png")
plt.show() # Remember to close this plot window to view accuracy output!

# ---------------- 6) Evaluation ----------------
test_preds = svm.predict(X_test)
acc = accuracy_score(test_labels, test_preds)
print(f"\n==========================================")
print(f"Overall Fused Features Test Accuracy: {acc * 100:.2f}%")
print(f"==========================================\n")
print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

# Save artifacts
joblib.dump(svm, "fused_svm.joblib")
joblib.dump(vocabulary, "fused_vocab.joblib")
joblib.dump(scaler_bovw, "scaler_bovw.joblib")
joblib.dump(scaler_hsv, "scaler_hsv.joblib")
joblib.dump(scaler_lbp, "scaler_lbp.joblib")
joblib.dump(label_names, "fused_label_names.joblib")
print("Saved all fused pipeline artifacts successfully.")