import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# CONFIGURATION
# ======================================================
BASE_DIR = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\signatures"
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
IMG_SIZE = (128, 128)

# ======================================================
# CUSTOM LAYERS
# ======================================================
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

class AbsDiffLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

# ======================================================
# LOAD MODEL
# ======================================================
with tf.keras.utils.custom_object_scope({'L2NormalizeLayer': L2NormalizeLayer, 'AbsDiffLayer': AbsDiffLayer}):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print(f"✅ Model loaded from: {MODEL_PATH}")

# ======================================================
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

# ======================================================
# LOAD UNSEEN TEST SET
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
            print(f"⚠️ Skipping {person}: missing folders.")
            continue

        originals = [os.path.join(orig_dir, f) for f in os.listdir(orig_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        forgeries = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(originals) < 2 or len(forgeries) == 0:
            continue

        # Genuine pairs (positive)
        for i in range(len(originals)):
            for j in range(i + 1, len(originals)):
                pairs.append([preprocess_image(originals[i]), preprocess_image(originals[j])])
                labels.append(1)

        # Forged pairs (negative)
        for orig in originals:
            for forg in forgeries:
                pairs.append([preprocess_image(orig), preprocess_image(forg)])
                labels.append(0)
    return np.array(pairs), np.array(labels, dtype=np.float32)

# --- Select test folders (unseen people)
all_people = sorted([p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))], key=numeric_sort)
test_people = all_people[8:10]
print(f"📂 Evaluating on unseen folders: {test_people}")

X_test, y_test = load_pairs_subset(BASE_DIR, test_people)
print(f"✅ Loaded {len(X_test)} test pairs from unseen people.")

# ======================================================
# PREDICT & EVALUATE
# ======================================================
X1_test, X2_test = X_test[:, 0], X_test[:, 1]
preds = model.predict([X1_test, X2_test], batch_size=16).ravel()

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, preds)

# Optimal threshold (Youden’s J index)
j_scores = tpr - fpr
j_best_idx = np.argmax(j_scores)
optimal_thresh_roc = roc_thresholds[j_best_idx]

# Optimal F1 threshold (from Precision-Recall)
f1_scores = 2 * precision * recall / (precision + recall + 1e-7)
best_f1_idx = np.argmax(f1_scores)
optimal_thresh_f1 = pr_thresholds[best_f1_idx]

print(f"\n🔍 Optimal threshold (ROC Youden's J): {optimal_thresh_roc:.3f}")
print(f"🔍 Optimal threshold (Best F1): {optimal_thresh_f1:.3f}")
print(f"📈 ROC-AUC = {roc_auc:.3f}")

# ======================================================
# APPLY OPTIMAL THRESHOLD
# ======================================================
THRESHOLD = optimal_thresh_f1  # or optimal_thresh_roc
pred_labels = (preds >= THRESHOLD).astype(int)

# Classification metrics
acc = accuracy_score(y_test, pred_labels)
print("\n📊 Classification Report:")
print(classification_report(y_test, pred_labels, target_names=["Forgery (0)", "Genuine (1)"]))
print(f"✅ Overall Accuracy: {acc*100:.2f}% at threshold={THRESHOLD:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Forged", "Pred Genuine"],
            yticklabels=["True Forged", "True Genuine"])
plt.title("Confusion Matrix - Signature Verification")
plt.show()

# ======================================================
# ROC + PR CURVES
# ======================================================
plt.figure(figsize=(12,5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.scatter(fpr[j_best_idx], tpr[j_best_idx], color='red', label=f"Best Th={optimal_thresh_roc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="PR Curve")
plt.scatter(recall[best_f1_idx], precision[best_f1_idx], color='green',
            label=f"Best F1 Th={optimal_thresh_f1:.2f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()

# ======================================================
# VISUALIZE SAMPLE PREDICTIONS
# ======================================================
import random
indices = random.sample(range(len(X_test)), 5)
plt.figure(figsize=(12,8))
for idx, i in enumerate(indices):
    plt.subplot(2, 3, idx + 1)
    img1 = np.squeeze(X_test[i, 0])
    img2 = np.squeeze(X_test[i, 1])
    combined = np.hstack((img1, img2))
    plt.imshow(combined, cmap="gray")
    plt.title(f"True: {'Genuine' if y_test[i]==1 else 'Forgery'} | Pred: {'Genuine' if pred_labels[i]==1 else 'Forgery'}\nScore: {preds[i]:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()
