try:
    import tensorflow as tf
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False
    
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("final_brain_multi_disease_model.keras")

# Class labels
class_names = ['Alzheimers_Positive', 'Healthy', 'MS_Positive', 'Tumor_Positive']

# Title
st.title("🧠 BrainScan: MRI Disease Classification")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Grad-CAM function
def make_gradcam_heatmap(img_array, model):
    last_conv_layer = model.get_layer("Conv_1")
    
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
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

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img

if not MODEL_AVAILABLE:
    st.error("Model cannot run on this server (Python version issue).")
    st.stop()

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)
    img = np.array(image)

    heatmap_img = overlay_heatmap(img, heatmap)

    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
