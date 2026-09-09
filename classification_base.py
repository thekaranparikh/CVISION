"""
Lab 4 - Caltech-256 classification with SIFT + Bag-of-Visual-Words + SVM

Pipeline:
  1. DataLoader           -> walk the Caltech-256 folder-per-class structure
  2. Train / val / test split
  3. SIFT features on grayscale images
  4. K-Means (k=64) on ALL training SIFT descriptors -> visual vocabulary
  5. Bag-of-Visual-Words (BoVW) histogram per image
  6. Multiclass SVM (linear, trained via SGD so we can plot a loss curve)
  7. Test-set evaluation: precision / recall / F1
  8. Save the trained SVM + vocabulary + label names for later reuse
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
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ---------------- Configuration ----------------
CALTECH_DIR = "256_ObjectCategories"   # folder containing one subfolder per class, e.g. "001.ak47"
VOCAB_SIZE = 128                        # k for the visual-vocabulary K-Means
RESIZE_DIM = 128                       # resize every image to a consistent size before feature extraction
MAX_IMAGES_PER_CLASS = 100              # subsample for a lab-friendly runtime; raise this for a stronger model
NUM_EPOCHS = 50                        # SGD-SVM training epochs
RANDOM_STATE = 42


# ---------------- 1) DataLoader ----------------
def load_caltech256_paths(root_dir, max_per_class):
    """Walk the one-folder-per-class structure and return (filepath, label_id) pairs,
    subsampled to at most `max_per_class` images per class. We keep file PATHS, not
    loaded images, since Caltech-256 is too large to hold entirely in memory."""
    class_folders = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    label_names = [name.split(".", 1)[1] if "." in name else name for name in class_folders]

    rng = random.Random(RANDOM_STATE)
    filepaths, labels = [], []

    for label_id, folder in enumerate(class_folders):
        folder_path = os.path.join(root_dir, folder)
        images_in_class = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rng.shuffle(images_in_class)
        for fname in images_in_class[:max_per_class]:
            filepaths.append(os.path.join(folder_path, fname))
            labels.append(label_id)

    return filepaths, np.array(labels), label_names


filepaths, labels, label_names = load_caltech256_paths(CALTECH_DIR, MAX_IMAGES_PER_CLASS)
print(f"Loaded {len(filepaths)} images across {len(label_names)} classes.")


# ---------------- 2) Train / val / test split ----------------
# Caltech-256 doesn't ship a fixed split, so we carve out all three ourselves (stratified,
# so every class is represented proportionally in each split).
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filepaths, labels, test_size=0.30, stratify=labels, random_state=RANDOM_STATE
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_STATE
)

print(f"Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")


# ---------------- 3) SIFT features on grayscale images ----------------
sift = cv2.SIFT_create()


def extract_sift_descriptors(paths):
    """Returns a list of per-image descriptor arrays (each shape [n_keypoints, 128], or None).
    Images are read from disk one at a time here, rather than preloaded, since Caltech-256's
    full-resolution JPEGs are much larger than a toy dataset's packed arrays."""
    all_descriptors = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            all_descriptors.append(None)
            continue
        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        all_descriptors.append(descriptors)  # may be None if literally no keypoints were found
    return all_descriptors


print("Extracting SIFT descriptors (train)...")
train_descriptors = extract_sift_descriptors(train_paths)
print("Extracting SIFT descriptors (val)...")
val_descriptors = extract_sift_descriptors(val_paths)
print("Extracting SIFT descriptors (test)...")
test_descriptors = extract_sift_descriptors(test_paths)


# ---------------- 4) K-Means visual vocabulary (k=64) on ALL training descriptors ----------------
# Fit the vocabulary on the TRAINING set's descriptors ONLY - fitting it on val/test
# descriptors too would leak information about those images into the vocabulary.
all_train_descriptors = np.vstack([d for d in train_descriptors if d is not None])
print(f"Total training SIFT descriptors: {all_train_descriptors.shape[0]}")

print(f"Clustering into a {VOCAB_SIZE}-word visual vocabulary...")
vocabulary = KMeans(n_clusters=VOCAB_SIZE, n_init=4, random_state=RANDOM_STATE)
vocabulary.fit(all_train_descriptors)


# ---------------- 5) Bag-of-Visual-Words histogram per image ----------------
def to_bovw_histogram(descriptors, vocabulary, vocab_size):
    """Assign each descriptor to its nearest visual word, and build a
    normalized histogram of word occurrences for the image."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size, dtype=np.float32)  # no features found -> empty histogram

    word_ids = vocabulary.predict(descriptors)
    histogram, _ = np.histogram(word_ids, bins=np.arange(vocab_size + 1))
    histogram = histogram.astype(np.float32)

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm  # L2-normalize so keypoint COUNT doesn't dominate over word DISTRIBUTION

    return histogram


def build_feature_matrix(descriptor_list, vocabulary, vocab_size):
    return np.array([to_bovw_histogram(d, vocabulary, vocab_size) for d in descriptor_list])


X_train = build_feature_matrix(train_descriptors, vocabulary, VOCAB_SIZE)
X_val = build_feature_matrix(val_descriptors, vocabulary, VOCAB_SIZE)
X_test = build_feature_matrix(test_descriptors, vocabulary, VOCAB_SIZE)

# Standardize features (zero mean, unit variance) - helps SGD converge faster and more stably
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ---------------- 6) Multiclass SVM via SGD (so we can track a loss curve) ----------------
def multiclass_hinge_loss(decision_values, true_labels, classes):
    """Approximate one-vs-rest multiclass hinge loss from decision_function output."""
    y_binarized = np.array([[1 if c == label else -1 for c in classes] for label in true_labels])
    margins = np.clip(1 - y_binarized * decision_values, 0, None)
    return float(np.mean(margins))


svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE, learning_rate="optimal")
classes = np.unique(train_labels)

train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    svm.partial_fit(X_train, train_labels, classes=classes)

    train_loss = multiclass_hinge_loss(svm.decision_function(X_train), train_labels, classes)
    val_loss = multiclass_hinge_loss(svm.decision_function(X_val), val_labels, classes)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Hinge loss")
plt.title("SVM (SGD) Training/Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()


# ---------------- 7) Test-set evaluation ----------------
test_predictions = svm.predict(X_test)
report = classification_report(
    test_labels, test_predictions, target_names=label_names, zero_division=0
)
print("\nTest set performance:\n")
print(report)


# ---------------- 8) Save the trained pipeline for later reuse ----------------
joblib.dump(svm, "caltech256_svm.joblib")
joblib.dump(vocabulary, "caltech256_vocabulary.joblib")
joblib.dump(scaler, "caltech256_scaler.joblib")
joblib.dump(label_names, "caltech256_label_names.joblib")
print("Saved model, vocabulary, scaler, and label names to disk.")