import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = (128, 128)
THRESHOLD = 0.7
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_10persons.keras"

# Load the trained model
import keras
keras.config.enable_unsafe_deserialization()
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import layers

# Define the same custom layers you used during training:
class L2NormalizeLayer(layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

class AbsDiffLayer(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

# Load model with registered custom layers
with custom_object_scope({'L2NormalizeLayer': L2NormalizeLayer, 'AbsDiffLayer': AbsDiffLayer}):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# ==============================
# IMAGE PREPROCESSING
# ==============================
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, (0, -1))

# ==============================
# GUI SETUP
# ==============================
root = tk.Tk()
root.title("🖊️ Offline Signature Verification System")
root.geometry("850x500")
root.configure(bg="#222831")

# Variables
ref_path = None
test_path = None

# ==============================
# FUNCTIONS
# ==============================
def select_reference():
    global ref_path
    path = filedialog.askopenfilename(title="Select Reference Signature",
                                      filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if path:
        ref_path = path
        display_image(path, ref_label, "Reference Signature")

def select_test():
    global test_path
    path = filedialog.askopenfilename(title="Select Test Signature",
                                      filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if path:
        test_path = path
        display_image(path, test_label, "Test Signature")

def display_image(path, label_widget, title):
    """Display selected image on the GUI."""
    img = Image.open(path).resize((200, 200))
    tk_img = ImageTk.PhotoImage(img)
    label_widget.config(image=tk_img, text=title, compound="top", fg="white", font=("Segoe UI", 11))
    label_widget.image = tk_img

def verify_signature():
    if not ref_path or not test_path:
        messagebox.showwarning("Missing Image", "Please select both reference and test signatures.")
        return

    img1 = preprocess_image(ref_path)
    img2 = preprocess_image(test_path)

    score = model.predict([img1, img2])[0][0]
    result_label.config(text=f"Similarity Score: {score:.3f}", fg="lightblue")

    if score > THRESHOLD:
        verdict_label.config(text="✅ Genuine Signature", fg="lightgreen")
    else:
        verdict_label.config(text="❌ Forged Signature", fg="red")

# ==============================
# GUI ELEMENTS
# ==============================
title_label = tk.Label(root, text="Offline Signature Verification", font=("Segoe UI", 20, "bold"), bg="#222831", fg="#EEEEEE")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#222831")
frame.pack()

# Reference column
ref_label = tk.Label(frame, text="Reference Signature", bg="#222831", fg="white", width=30, height=12)
ref_label.grid(row=0, column=0, padx=30)
ref_btn = tk.Button(frame, text="Select Reference", command=select_reference, font=("Segoe UI", 11), bg="#00ADB5", fg="white", relief="flat", width=20)
ref_btn.grid(row=1, column=0, pady=10)

# Test column
test_label = tk.Label(frame, text="Test Signature", bg="#222831", fg="white", width=30, height=12)
test_label.grid(row=0, column=1, padx=30)
test_btn = tk.Button(frame, text="Select Test", command=select_test, font=("Segoe UI", 11), bg="#00ADB5", fg="white", relief="flat", width=20)
test_btn.grid(row=1, column=1, pady=10)

# Predict button
predict_btn = tk.Button(root, text="Verify Signature", command=verify_signature, font=("Segoe UI", 13, "bold"), bg="#393E46", fg="white", relief="flat", width=25)
predict_btn.pack(pady=25)

# Results
result_label = tk.Label(root, text="Similarity Score: --", font=("Segoe UI", 13), bg="#222831", fg="white")
result_label.pack()

verdict_label = tk.Label(root, text="", font=("Segoe UI", 18, "bold"), bg="#222831", fg="white")
verdict_label.pack(pady=10)

# ==============================
# MAIN LOOP
# ==============================
root.mainloop()
