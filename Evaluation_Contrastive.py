"""
====================================================================
Signature Verification Evaluation (Contrastive Siamese)
====================================================================
Author: Gautham Ganesh
Description:
    Evaluates trained contrastive Siamese model.
    Computes ROC-AUC, Precision-Recall, Optimal Threshold,
    and Classification Metrics.

    ⚙️ Requirements:
        - Model trained using EuclideanDistanceLayer (contrastive)
        - Uses unseen test pairs for evaluation
====================================================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, classification_report, auc
)
import matplotlib.pyplot as plt

# ======================================================
# ⚙️ CONFIG
# ======================================================
BASE_DIR = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\signatures"
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
IMG_SIZE = (128, 128)
SAVE_DIR = os.path.dirname(MODEL_PATH)

# ======================================================
# 🧩 IMPORT CUSTOM LAYER (Used in training)
# ======================================================
class EuclideanDistanceLayer(tf.keras.layers.Layer):
    """Computes Euclidean distance between two embeddings."""
    def call(self, inputs):
        a, b = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# ======================================================
# 🧠 LOAD TRAINED MODEL SAFELY
# ======================================================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'EuclideanDistanceLayer': EuclideanDistanceLayer},
    compile=False
)
print("✅ Model loaded successfully.")

# ======================================================
# 🧼 IMAGE PREPROCESSING (same as training)
# ======================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, -1)

# ======================================================
# 📁 LOAD TEST SET
# ======================================================
def numeric_sort(folder_name):
    import re
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else 0

def load_pairs_subset(base_dir, person_list):
    pairs, labels = [], []
    for person in person_list: 
        person_path = os.path.join(base_dir, person)
        orig_dir = os.path.join(person_path, "originals")
        forg_dir = os.path.join(person_path, "forgeries")
        if not os.path.exists(orig_dir) or not os.path.exists(forg_dir):
            continue

        originals = [os.path.join(orig_dir, f) for f in os.listdir(orig_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        forgeries = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(originals) < 2:
            continue

        for i in range(len(originals)):
            for j in range(i + 1, len(originals)):
                pairs.append([preprocess_image(originals[i]), preprocess_image(originals[j])])
                labels.append(1)
        for orig in originals:
            img1 = preprocess_image(orig)
            for forg in forgeries:
                img2 = preprocess_image(forg)
                pairs.append([img1, img2])
                labels.append(0)
    return np.array(pairs), np.array(labels, dtype=np.float32)

# Get test folders (same split logic)
all_people = sorted(
    [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))],
    key=numeric_sort
)[:10]
test_people = all_people[8:]

X_test, y_test = load_pairs_subset(BASE_DIR, test_people)
X1_test, X2_test = X_test[:, 0], X_test[:, 1]

print(f"📂 Loaded {len(X_test)} test pairs from unseen persons.")

# ======================================================
# 🔍 PREDICT DISTANCES
# ======================================================
distances = model.predict([X1_test, X2_test], batch_size=16)
distances = distances.flatten()

# Smaller distance → More similar (genuine)
# Larger distance → More different (forgery)

# ======================================================
# 📈 ROC / PR CURVES + METRICS
# ======================================================
fpr, tpr, roc_thresholds = roc_curve(y_test, -distances)  # invert for similarity
roc_auc = auc(fpr, tpr)

prec, rec, pr_thresholds = precision_recall_curve(y_test, -distances)
pr_auc = auc(rec, prec)

# Optimal thresholds
youden_index = np.argmax(tpr - fpr)
optimal_thresh_youden = roc_thresholds[youden_index]

f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
optimal_thresh_f1 = pr_thresholds[np.argmax(f1_scores)]

# ======================================================
# 🧾 CLASSIFICATION REPORT
# ======================================================
y_pred_opt = (distances < abs(optimal_thresh_f1)).astype(int)

print("\n🔍 Evaluation Results")
print("──────────────────────────────────────────────")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")
print(f"Optimal Threshold (Youden's J): {optimal_thresh_youden:.4f}")
print(f"Optimal Threshold (F1): {optimal_thresh_f1:.4f}\n")

from sklearn.metrics import classification_report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_opt, target_names=["Forgery (0)", "Genuine (1)"]))

# ======================================================
# 📉 PLOT & SAVE
# ======================================================
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Contrastive Siamese")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(rec, prec, label=f"PR (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Contrastive Siamese")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
plt.close()

print(f"✅ Plots saved in: {SAVE_DIR}")
