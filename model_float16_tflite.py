import tensorflow as tf
import numpy as np
import os

# ---------------------------
# Custom Layers
# ---------------------------
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
# ⚙️ Configuration
# ======================================================
MODEL_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive.keras"
TFLITE_PATH = r"C:\Users\gauth\OneDrive\Documents\WORK\MTech\MLES\PROJECT2\signature_verification_contrastive_fp16.tflite"

# ======================================================
# 🔁 Conversion
# ======================================================
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L2NormalizeLayer": L2NormalizeLayer,
        "EuclideanDistanceLayer": EuclideanDistanceLayer,
    },
    compile=False,
)
print("✅ Model loaded.")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

print("Converting to float16 TFLite...")
tflite_model = converter.convert()

# Save model
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Conversion complete! Saved to:\n{TFLITE_PATH}")
print(f"📦 Model size: {os.path.getsize(TFLITE_PATH) / 1024:.1f} KB")
