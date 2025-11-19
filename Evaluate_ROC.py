"""
====================================================================
Evaluation: ROC + Precision-Recall for Contrastive Siamese Network
====================================================================
Author: Gautham Ganesh
Description:
    Evaluates a trained contrastive Siamese signature verification model.
    Plots ROC, Precision-Recall, computes optimal thresholds.

    ✅ Works with contrastive loss models
    ✅ Uses EuclideanDistanceLayer & L2NormalizeLayer (safe custom layers)
    ✅ Automatically saves ROC & PR plots
====================================================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report

# ======================================================
# 🧱 CUSTOM LAYERS (Registered)
# ======================================================
@tf.keras.utils.register_keras_serializable()
class L2NormalizeLayer(tf.keras.layers.Layer):
    """Normalizes embedding vectors."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)


@tf.keras.utils.register_keras_serializable()
class EuclideanDistanceLayer(tf.keras.layers.Layer):
    """Computes Euclidean distance between two embeddings."""
    def call(self, inputs):
        a, b = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# ======================================================
# 🧩 HELPER FUNCTIONS
# ======================================================
def numeric_sort(folder_name):
    """Sorts folders numerically (signatures_1, signatures_2...)."""
    import re
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else 0

def preprocess_image(image_path):
    """Reads and normalizes grayscale signature."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, -1)

def load_pairs_subset(base_dir, person_list):
    """Loads signature image pairs (originals vs forgeries)."""
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

        # Genuine (positive) pairs
        for i in range(len(originals)):
            for j in range(i + 1, len(originals)):
                pairs.append([preprocess_image(originals[i]), preprocess_image(originals[j])])
                labels.append(1)

        # Forgery (negative) pairs
        for orig in originals:
            img1 = preprocess_image(orig)
            for forg in forgeries:
                img2 = preprocess_image(forg)
                pairs.append([img1, img2])
                labels.append(0)

    return np.array(pairs), np.array(labels, dtype=np.float32)

# ======================================================
# ⚙️ CONFIGURATION
# ======================================================
BASE_DIR = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\signatures"
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
SAVE_DIR = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2"

# ======================================================
# 🔁 LOAD MODEL (SAFE)
# ======================================================
print("🔁 Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'L2NormalizeLayer': L2NormalizeLayer,
        'EuclideanDistanceLayer': EuclideanDistanceLayer
    },
    compile=False
)
print("✅ Model loaded successfully.")

# ======================================================
# 📁 LOAD TEST DATA (UNSEEN SIGNERS)
# ======================================================
all_people = sorted(
    [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))],
    key=numeric_sort
)[:14]
test_people = all_people[12:]
print(f"🧩 Loading test data from unseen persons: {test_people}")

X_test, y_test = load_pairs_subset(BASE_DIR, test_people)
X1_test, X2_test = X_test[:, 0], X_test[:, 1]

# ======================================================
# 🔮 MODEL PREDICTIONS
# ======================================================
print("🔮 Computing distances...")
distances = model.predict([X1_test, X2_test], batch_size=8)
similarity_scores = 1 - distances  # invert: smaller distance → more genuine

# ======================================================
# 📈 ROC & PR CURVES
# ======================================================
fpr, tpr, roc_thresholds = roc_curve(y_test, similarity_scores)
precision, recall, pr_thresholds = precision_recall_curve(y_test, similarity_scores)
roc_auc = auc(fpr, tpr)

# ======================================================
# 🧮 OPTIMAL THRESHOLDS
# ======================================================
youdenJ = tpr - fpr
optimal_idx_J = np.argmax(youdenJ)
optimal_thresh_J = roc_thresholds[optimal_idx_J]

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx_F1 = np.argmax(f1_scores)
optimal_thresh_F1 = pr_thresholds[min(optimal_idx_F1, len(pr_thresholds)-1)]

# ======================================================
# 🧾 CLASSIFICATION REPORT
# ======================================================
y_pred_opt = (similarity_scores >= optimal_thresh_F1).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_opt, target_names=["Forgery (0)", "Genuine (1)"]))
accuracy = np.mean(y_pred_opt.flatten() == y_test.flatten()) * 100
print(f"✅ Overall Accuracy: {accuracy:.2f}%")
print(f"🔍 Optimal threshold (ROC Youden's J): {optimal_thresh_J:.3f}")
print(f"🔍 Optimal threshold (Best F1): {optimal_thresh_F1:.3f}")
print(f"📈 ROC-AUC = {roc_auc:.3f}")

# ======================================================
# 📉 PLOTTING
# ======================================================
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.scatter(fpr[optimal_idx_J], tpr[optimal_idx_J], color='red', label=f'Optimal J={optimal_thresh_J:.2f}')
plt.title("ROC Curve - Contrastive Siamese Network")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ROC_curve.png"))
print("📊 ROC curve saved as ROC_curve.png")

plt.figure(figsize=(7,6))
plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
plt.scatter(recall[optimal_idx_F1], precision[optimal_idx_F1],
            color='red', label=f'Optimal F1={optimal_thresh_F1:.2f}')
plt.title("Precision-Recall Curve - Contrastive Siamese Network")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "PR_curve.png"))
print("📊 Precision-Recall curve saved as PR_curve.png")

print("\n✅ Evaluation completed successfully.")
