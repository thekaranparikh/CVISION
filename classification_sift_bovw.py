"""
Lab 4 - SIFT + Bag-of-Visual-Words (BoVW) Baseline
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

# 70% Train, 15% Val, 15% Test Split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, train_size=0.70, stratify=labels, random_state=RANDOM_STATE
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_STATE
)

# ---------------- 2) Extract SIFT Keypoint Descriptors ----------------
sift = cv2.SIFT_create()

def extract_sift_descriptors(paths):
    sift_list = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            sift_list.append(None)
            continue
        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        sift_list.append(des)
    return sift_list

print("Extracting SIFT descriptors...")
train_sift = extract_sift_descriptors(train_paths)
val_sift = extract_sift_descriptors(val_paths)
test_sift = extract_sift_descriptors(test_paths)

# ---------------- 3) Build Visual Vocabulary via K-Means ----------------
all_train_des = np.vstack([d for d in train_sift if d is not None])
print(f"Fitting K-Means vocabulary with K={VOCAB_SIZE} on {all_train_des.shape[0]} descriptors...")

vocabulary = KMeans(n_clusters=VOCAB_SIZE, n_init=4, random_state=RANDOM_STATE)
vocabulary.fit(all_train_des)

# ---------------- 4) Construct BoVW Histograms ----------------
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

print("Building BoVW histograms...")
train_bovw = build_bovw_matrix(train_sift, vocabulary, VOCAB_SIZE)
val_bovw = build_bovw_matrix(val_sift, vocabulary, VOCAB_SIZE)
test_bovw = build_bovw_matrix(test_sift, vocabulary, VOCAB_SIZE)

# ---------------- 5) Standardize Features ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_bovw)
X_val = scaler.transform(val_bovw)
X_test = scaler.transform(test_bovw)

# ---------------- 6) Train Linear Classifier & Track Loss ----------------
def multiclass_hinge_loss(decision_values, true_labels, classes):
    y_binarized = np.array([[1 if c == label else -1 for c in classes] for label in true_labels])
    margins = np.clip(1 - y_binarized * decision_values, 0, None)
    return float(np.mean(margins))

svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
classes = np.unique(train_labels)

train_losses, val_losses = [], []

print("Training SGD-SVM classifier...")
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
plt.title("SIFT + BoVW Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("sift_bovw_loss_curve.png")
plt.show()

# ---------------- 7) Evaluate & Save Artifacts ----------------
test_preds = svm.predict(X_test)
acc = accuracy_score(test_labels, test_preds)
print(f"\n==========================================")
print(f"Overall SIFT-BoVW Test Accuracy: {acc * 100:.2f}%")
print(f"==========================================\n")
print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

# Save artifacts for single-image testing
joblib.dump(svm, "sift_svm.joblib")
joblib.dump(vocabulary, "sift_vocab.joblib")
joblib.dump(scaler, "sift_scaler.joblib")
joblib.dump(label_names, "sift_label_names.joblib")
print("Saved all SIFT-BoVW model artifacts successfully.")