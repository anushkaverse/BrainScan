import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="BrainScan",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 BrainScan")
st.subheader("Unified Explainable AI for Brain MRI Classification")

st.markdown(
    "**Classes:** Healthy | Alzheimer’s Disease | Multiple Sclerosis | Brain Tumor"
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/final_brain_multi_disease_model.keras(1)"
    )

model = load_model()

# Class labels (ORDER MUST MATCH TRAINING)
class_names = [
    "Alzheimer’s Disease",
    "Brain Tumor",
    "Healthy",
    "Multiple Sclerosis"
]

# -----------------------------
# Grad-CAM Function
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load & show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown("---")
    st.success(f"### 🧠 Prediction: **{predicted_class}**")
    st.info(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

    # -----------------------------
    # Grad-CAM Visualization
    # -----------------------------
    st.markdown("### 🔍 Grad-CAM Explanation")

    # ⚠️ CHANGE THIS if your last conv layer has a different name
    LAST_CONV_LAYER = "Conv_1"   # MobileNetV2 last conv layer

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        LAST_CONV_LAYER
    )

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(
        img, 0.6, heatmap_color, 0.4, 0
    )

    # Display Grad-CAM
    st.image(
        superimposed_img,
        caption="Grad-CAM Heatmap (Model Attention Regions)",
        use_column_width=True
    )

    st.caption(
        "Red regions indicate areas that most influenced the model's decision."
    )
