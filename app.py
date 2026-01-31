import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Surface Defect Detection", layout="centered")

# ----------------------------------
# Load model (cached)
# ----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\Nayan\Desktop\Data Science & Machine Learning\surface_defect_detection\best_defect_model.keras")

model = load_model()

# Class names (same order as training)
CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]



st.title("🔍 Surface Defect Detection (CNN)")
st.write("Upload a steel surface image to classify the defect type.")

# ----------------------------------
# Image upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((128, 128))
    img_array = np.array(image) 
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1,128,128,1)

    # Predict
    if st.button("Predict Defect"):
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]

        st.subheader("Prediction Result")
        # st.write("Raw probabilities:", preds[0])

        st.success(f"**Defect Type:** {CLASS_NAMES[class_idx]}")
        st.write(f"**Confidence:** {confidence:.2f}")
