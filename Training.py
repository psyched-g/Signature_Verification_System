"""
====================================================================
Signature Verification using Siamese Network + Contrastive Loss
====================================================================
Author: Gautham Ganesh
Description:
    This version replaces Binary Cross-Entropy with Contrastive Loss.
    It learns a true similarity metric instead of just classification.
    Same dataset structure, same preprocessing — much stronger results.

    Works with folders:
        signatures_1/... signatures_10/
            ├── originals/
            ├── forgeries/
====================================================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================================
# ⚙️ CONFIGURATION
# ======================================================
BASE_DIR = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\signatures"
IMG_SIZE = (128, 128)
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
MARGIN = 1.0  # margin for contrastive loss separation

# ======================================================
# 🧱 CUSTOM LAYERS
# ======================================================
class L2NormalizeLayer(layers.Layer):
    """Normalizes embedding vectors."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

# ======================================================
# 📊 CONTRASTIVE LOSS
# ======================================================
def contrastive_loss(y_true, y_pred):
    """
    y_true: 1 = genuine, 0 = forgery
    y_pred: distance between embeddings
    """
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(MARGIN - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# ======================================================
# 🧼 IMAGE PREPROCESSING
# ======================================================
def preprocess_image(image_path):
    """Reads, resizes, enhances, and normalizes grayscale signature."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, -1)

# ======================================================
# 📁 DATASET LOADER
# ======================================================
def numeric_sort(folder_name):
    """Ensures folder sorting numerically (signatures_1, signatures_2, ...)."""
    import re
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else 0

def load_pairs_subset(base_dir, person_list):
    """Loads (img1, img2) pairs and labels (1=genuine, 0=forgery)."""
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

        # Positive pairs (same person)
        for i in range(len(originals)):
            for j in range(i + 1, len(originals)):
                pairs.append([preprocess_image(originals[i]), preprocess_image(originals[j])])
                labels.append(1)

        # Negative pairs (original vs forgery)
        for orig in originals:
            img1 = preprocess_image(orig)
            for forg in forgeries:
                img2 = preprocess_image(forg)
                pairs.append([img1, img2])
                labels.append(0)

    return np.array(pairs), np.array(labels, dtype=np.float32)

# ======================================================
# 🧩 DATASET SPLIT (Train/Test by Person)
# ======================================================
all_people = sorted(
    [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))],
    key=numeric_sort
)[:10]

train_people = all_people[:8]   # 8 persons for training
test_people = all_people[8:]    # 2 persons for unseen testing

X_train, y_train = load_pairs_subset(BASE_DIR, train_people)
X_test, y_test = load_pairs_subset(BASE_DIR, test_people)

print(f"✅ Train pairs: {len(X_train)}, Test pairs (unseen people): {len(X_test)}")

# ======================================================
# 🔄 DATA AUGMENTATION
# ======================================================
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    shear_range=0.05,
    brightness_range=[0.8, 1.2]
)

def augment_batch(imgs1, imgs2, labels):
    """Augment both images in each pair."""
    aug_imgs1, aug_imgs2, aug_labels = [], [], []
    for i in range(len(imgs1)):
        aug_imgs1.append(datagen.random_transform(imgs1[i]))
        aug_imgs2.append(datagen.random_transform(imgs2[i]))
        aug_labels.append(labels[i])
    return np.array(aug_imgs1), np.array(aug_imgs2), np.array(aug_labels)

# ======================================================
# 🧠 BUILD EMBEDDING NETWORK
# ======================================================
def build_embedding_network(input_shape):
    """Defines CNN feature extractor for signatures."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = L2NormalizeLayer()(x)
    return Model(inp, x, name="EmbeddingNet")

embedding_net = build_embedding_network((*IMG_SIZE, 1))

# ======================================================
# 🔗 BUILD SIAMESE NETWORK (DISTANCE-BASED)
# ======================================================
input_A = layers.Input((*IMG_SIZE, 1))
input_B = layers.Input((*IMG_SIZE, 1))

feat_A = embedding_net(input_A)
feat_B = embedding_net(input_B)

# Compute Euclidean distance (instead of sigmoid probability)
distance = layers.Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([feat_A, feat_B])

siamese_model = Model(inputs=[input_A, input_B], outputs=distance)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=contrastive_loss)

siamese_model.summary()

# ======================================================
# 🚀 TRAINING
# ======================================================
X1_train, X2_train = X_train[:, 0], X_train[:, 1]
X1_test, X2_test = X_test[:, 0], X_test[:, 1]

# Apply augmentation
aug_X1, aug_X2, aug_y = augment_batch(X1_train, X2_train, y_train)
X1_train = np.concatenate([X1_train, aug_X1])
X2_train = np.concatenate([X2_train, aug_X2])
y_train = np.concatenate([y_train, aug_y])

# Train model
history = siamese_model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_test, X2_test], y_test),
    batch_size=8,
    epochs=25
)

# ======================================================
# 💾 SAVE MODEL
# ======================================================
siamese_model.save(MODEL_PATH)
print(f"✅ Contrastive model trained and saved at: {MODEL_PATH}")
