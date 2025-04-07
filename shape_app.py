import streamlit as st
import cv2
import numpy as np
from PIL import Image
from joblib import load
import os
import sys

# Add pyfeats path for local import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from pyfeats import zernikes_moments

# Load models and encoders
model = load("zernike_shape_svm_model.joblib")
label_encoder = load("label_encoder.joblib")
text_clf = load("text_to_shape_classifier.joblib")

st.set_page_config(page_title="Shape Classifier", layout="centered")
st.title("🔷 Shape Classifier using Zernike Moments + NLP")

# Tabs for image and text-based classification
tab1, tab2 = st.tabs(["🖼️ Image Classifier", "✍️ Text Description Classifier"])

# ----------------------------
# 🖼️ Image Classifier Tab
# ----------------------------
with tab1:
    st.subheader("Upload Shape Image")
    st.write("Supported: Circle, Square, Triangle")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        image_resized = cv2.resize(image_np, (128, 128))
        _, binary = cv2.threshold(image_resized, 127, 255, cv2.THRESH_BINARY)

        try:
            feats, _ = zernikes_moments(binary, radius=64)
        except Exception as e:
            st.error(f"❌ Error extracting Zernike moments: {e}")
            st.stop()

        # Predict
        pred_proba = model.predict_proba([feats])[0]
        pred_class_idx = np.argmax(pred_proba)
        pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = pred_proba[pred_class_idx] * 100

        st.success(f"✅ Predicted Shape: **{pred_class.capitalize()}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        # Show probability chart
        st.subheader("Class Probabilities")
        st.bar_chart({label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(pred_proba)})

# ----------------------------
# ✍️ Text Classifier Tab (NLP)
# ----------------------------
with tab2:
    st.subheader("Describe the Shape in Text")
    st.write("Example: *A round object with no edges*, *Three corners and three sides*, etc.")

    description = st.text_area("Enter a shape description:")

    if description:
        predicted_text_label = text_clf.predict([description])[0]
        st.success(f"📝 Predicted Shape from Description: **{predicted_text_label.capitalize()}**")
