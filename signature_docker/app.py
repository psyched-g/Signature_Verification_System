"""
====================================================================
Signature Verification GUI – Contrastive Siamese Network (TFLite)
====================================================================
Author: Gautham Ganesh
Description:
    Loads a TFLite contrastive Siamese model and provides
    an interactive GUI for comparing two signatures.

    ✅ Select Reference & Test images
    ✅ Webcam Capture (preprocessed like scanned images)
    ✅ Predict Genuine / Forged with confidence
====================================================================
"""

import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ======================================================
# ⚙️ Configuration
# ======================================================
MODEL_PATH = "model.tflite" 
IMG_SIZE = (128, 128)
THRESHOLD = 0.91   # Your best F1 threshold

# ======================================================
# 🧠 Load TFLite Model
# ======================================================
print("🔁 Loading TFLite model...")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded successfully.")

# ======================================================
# 📸 Webcam Capture Function (Scanned-style)
# ======================================================
# ======================================================
# ?? Webcam Capture Function (with V4L2 + Safe Resolution)
# ======================================================
def capture_from_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        result_text.set("Cannot access webcam!")
        return

    preview = tk.Toplevel(root)
    preview.title("Live Webcam Press SPACE to capture")
    preview.geometry("500x300")

    lmain = tk.Label(preview)
    lmain.pack()

    captured_frame = None

    def show_frame():
        nonlocal captured_frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((500, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    def on_key(event):
        nonlocal captured_frame
        if event.keysym == "space":
            ret, frame = cap.read()
            if ret:
                captured_frame = frame
                preview.destroy()

    preview.bind("<Key>", on_key)
    show_frame()
    preview.grab_set()              
    root.wait_window(preview)  

    cap.release()

    if captured_frame is None:
        result_text.set("? Capture cancelled.")
        return

    # --- Your preprocessing below ---
    gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 10
    )

    inverted = cv2.bitwise_not(thresh)
    final_img = cv2.bitwise_not(inverted)

    save_path = filedialog.asksaveasfilename(
        title="Save Captured Signature",
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
    )

    if save_path:
        cv2.imwrite(save_path, final_img)
        result_text.set(f"? Saved: {os.path.basename(save_path)}")

        global test_path
        test_path = save_path
        show_image(test_path, label_test)
        test_label_text.set(f"Captured: {os.path.basename(save_path)}")


# ======================================================
# 🧼 Preprocessing
# ======================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=(0, -1))  # Shape: (1,128,128,1)

# ======================================================
# 🔮 TFLite Prediction Logic
# ======================================================
def tflite_predict(img1, img2):
    interpreter.set_tensor(input_details[0]['index'], img1)
    interpreter.set_tensor(input_details[1]['index'], img2)

    interpreter.invoke()

    distance = interpreter.get_tensor(output_details[0]['index'])[0][0]
    similarity = 1 - distance

    label = "Genuine ✅" if similarity >= THRESHOLD else "Forged ❌"
    return label, similarity, distance

# ======================================================
# 🔍 Verify Action
# ======================================================
def verify_signatures():
    if not reference_path or not test_path:
        result_text.set("⚠️ Select both images!")
        return
    try:
        img1 = preprocess_image(reference_path)
        img2 = preprocess_image(test_path)

        label, similarity, distance = tflite_predict(img1, img2)

        result_text.set(
            f"Prediction: {label}\n\n"
            f"Similarity: {similarity:.3f}\n"
            f"Distance: {distance:.3f}"
        )
        result_label.config(
            fg="#00FF00" if "Genuine" in label else "#FF4444"
        )
    except Exception as e:
        result_text.set(f"Error: {str(e)}")

# ======================================================
# 🎨 GUI Setup
# ======================================================
root = tk.Tk()
root.title("Signature Verification (TFLite)")
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
def show_image(path, widget):
    img = Image.open(path).resize((300, 150))
    img = ImageTk.PhotoImage(img)
    widget.configure(image=img)
    widget.image = img

# ======================================================
# 📂 Browse Functions
# ======================================================
def browse_reference():
    global reference_path
    reference_path = filedialog.askopenfilename(
        title="Select Reference",
        filetypes=[
    ("PNG", "*.png"),
    ("JPEG", "*.jpg"),
    ("JPEG", "*.jpeg"),
    ("All files", "*.*")
]

    )
    if reference_path:
        show_image(reference_path, label_ref)
        ref_label_text.set(f"Reference: {os.path.basename(reference_path)}")

def browse_test():
    global test_path
    test_path = filedialog.askopenfilename(
        title="Select Test",
        filetypes=[
    ("PNG", "*.png"),
    ("JPEG", "*.jpg"),
    ("JPEG", "*.jpeg"),
    ("All files", "*.*")
]

    )
    if test_path:
        show_image(test_path, label_test)
        test_label_text.set(f"Test: {os.path.basename(test_path)}")

# ======================================================
# 🧩 Layout
# ======================================================
tk.Label(root, text="Signature Verification System", font=font_title,
         fg="white", bg="#101820").pack(pady=15)

frame_images = tk.Frame(root, bg="#101820")
frame_images.pack()

label_ref = tk.Label(frame_images, bg="#101820")
label_ref.grid(row=0, column=0, padx=20)

ref_label_text = tk.StringVar()
tk.Label(frame_images, textvariable=ref_label_text,
         font=font_label, fg="white", bg="#101820").grid(row=1, column=0)

tk.Button(frame_images, text="Select Reference", font=font_label,
          command=browse_reference).grid(row=2, column=0, pady=10)

label_test = tk.Label(frame_images, bg="#101820")
label_test.grid(row=0, column=1, padx=20)

test_label_text = tk.StringVar()
tk.Label(frame_images, textvariable=test_label_text,
         font=font_label, fg="white", bg="#101820").grid(row=1, column=1)

tk.Button(frame_images, text="Select Test", font=font_label,
          command=browse_test).grid(row=2, column=1, pady=10)

tk.Button(frame_images, text="📸 Capture from Webcam", font=font_label,
          command=capture_from_webcam).grid(row=3, column=1, pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=font_result,
                        fg="white", bg="#101820")
result_label.pack(pady=30)

tk.Button(root, text="🔍 Verify", font=("Arial", 16, "bold"),
          bg="#00ADB5", fg="white", width=15,
          command=verify_signatures).pack(pady=10)

tk.Label(root, text=f"Threshold = {THRESHOLD:.3f}",
         font=("Arial", 12), fg="gray", bg="#101820").pack(pady=10)

root.mainloop()
