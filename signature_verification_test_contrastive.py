"""
====================================================================
Signature Verification GUI – Contrastive Siamese Network
====================================================================
Author: Gautham Ganesh
Description:
    Loads the trained contrastive Siamese model and provides
    an interactive GUI for comparing two signatures.

    ✅ Select Reference & Test images
    ✅ Displays both side-by-side
    ✅ Predicts Genuine / Forged with confidence
====================================================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ======================================================
# 🧱 Custom Layers (must match your training)
# ======================================================
@tf.keras.utils.register_keras_serializable()
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

@tf.keras.utils.register_keras_serializable()
class EuclideanDistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# ======================================================
# ⚙️ Configurationddd
# ======================================================
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.91  # From your evaluation best F1 threshold

# ======================================================
# 🧠 Load Model
# ======================================================
print("🔁 Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L2NormalizeLayer": L2NormalizeLayer,
        "EuclideanDistanceLayer": EuclideanDistanceLayer,
    },
    compile=False,
)
print("✅ Model loaded successfully.")

# ======================================================
# 🧼 Preprocessing
# ======================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=(0, -1))  # shape (1,128,128,1)

# ======================================================
# 🧮 Prediction Logic
# ======================================================
def predict_similarity(img_path1, img_path2):
    img1 = preprocess_image(img_path1)
    img2 = preprocess_image(img_path2)
    distance = model.predict([img1, img2])[0][0]
    similarity = 1 - distance
    label = "Genuine ✅" if similarity >= THRESHOLD else "Forged ❌"
    return label, similarity, distance

# ======================================================
# 🎨 GUI Setup
# ======================================================
root = tk.Tk()
root.title("Signature Verification System")
root.geometry("800x600")
root.configure(bg="#101820")

font_title = ("Arial", 20, "bold")
font_label = ("Arial", 14)
font_result = ("Arial", 18, "bold")

reference_path = None
test_path = None

# ======================================================
# 🖼️ Image Display Function
# ======================================================
def show_image(img_path, label_widget):
    img = Image.open(img_path)
    img = img.resize((300, 150))
    img = ImageTk.PhotoImage(img)
    label_widget.configure(image=img)
    label_widget.image = img

# ======================================================
# 📂 Browse Functions
# ======================================================
def browse_reference():
    global reference_path
    reference_path = filedialog.askopenfilename(
        title="Select Reference Signature", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if reference_path:
        show_image(reference_path, label_ref)
        ref_label_text.set(f"Reference Loaded: {os.path.basename(reference_path)}")

def browse_test():
    global test_path
    test_path = filedialog.askopenfilename(
        title="Select Test Signature", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if test_path:
        show_image(test_path, label_test)
        test_label_text.set(f"Test Loaded: {os.path.basename(test_path)}")

# ======================================================
# 🔍 Verify Action
# ======================================================
def verify_signatures():
    if not reference_path or not test_path:
        result_text.set("⚠️ Please select both images first!")
        return

    try:
        label, similarity, distance = predict_similarity(reference_path, test_path)
        result_text.set(f"Prediction: {label}\n\nSimilarity: {similarity:.3f}\nDistance: {distance:.3f}")
        result_label.config(
            fg="#00FF00" if "Genuine" in label else "#FF4444"
        )
    except Exception as e:
        result_text.set(f"Error: {str(e)}")

# ======================================================
# 🧩 Layout
# ======================================================
tk.Label(root, text="Signature Verification System", font=font_title, fg="white", bg="#101820").pack(pady=15)

frame_images = tk.Frame(root, bg="#101820")
frame_images.pack()

# Reference Image
label_ref = tk.Label(frame_images, bg="#101820")
label_ref.grid(row=0, column=0, padx=20)
ref_label_text = tk.StringVar()
tk.Label(frame_images, textvariable=ref_label_text, font=font_label, fg="white", bg="#101820").grid(row=1, column=0)
tk.Button(frame_images, text="Select Reference", font=font_label, command=browse_reference).grid(row=2, column=0, pady=10)

# Test Image
label_test = tk.Label(frame_images, bg="#101820")
label_test.grid(row=0, column=1, padx=20)
test_label_text = tk.StringVar()
tk.Label(frame_images, textvariable=test_label_text, font=font_label, fg="white", bg="#101820").grid(row=1, column=1)
tk.Button(frame_images, text="Select Test", font=font_label, command=browse_test).grid(row=2, column=1, pady=10)

# Result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=font_result, fg="white", bg="#101820")
result_label.pack(pady=30)

tk.Button(root, text="🔍 Verify", font=("Arial", 16, "bold"), bg="#00ADB5", fg="white", width=15, command=verify_signatures).pack(pady=10)

tk.Label(root, text="Threshold = {:.3f}".format(THRESHOLD), font=("Arial", 12), fg="gray", bg="#101820").pack(pady=10)

root.mainloop()
