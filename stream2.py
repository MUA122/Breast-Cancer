import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model

# ----------------------------- Shared Constants -----------------------------
model_columns = [
    'age_group_5_years', 'race_eth', 'first_degree_hx', 'age_menarche',
    'age_first_birth', 'BIRADS_breast_density', 'current_hrt',
    'menopaus', 'bmi_group', 'biophx'
]

encoding_maps = {
    'age_group_5_years': lambda x: min(max(int(float(x) - 19) // 5, 1), 11),
    'race_eth': {'White': 1, 'Black': 2, 'Hispanic': 3, 'Asian': 4, 'Other': 5, 'Unknown': 6},
    'first_degree_hx': {'No': 0, 'Yes': 1},
    'age_menarche': {'<12': 0, '12–14': 1, '>14': 2},
    'age_first_birth': {'<20': 0, '20–24': 1, '25–29': 2, '30–34': 3, '>34': 4},
    'BIRADS_breast_density': {'1': 1, '2': 2, '3': 3, '4': 4},
    'current_hrt': {'No': 0, 'Yes': 1},
    'menopaus': {'Pre': 1, 'Peri': 2, 'Post': 3},
    'bmi_group': {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4},
    'biophx': {'No': 0, 'Yes': 1}
}

# ----------------------------- Caching -----------------------------
@st.cache_resource
def load_categorical_model():
    return joblib.load(r"templates/full_model.pkl")

@st.cache_resource
def load_xai_model():
    return MobileNetV2(weights="imagenet")

# ----------------------------- Sections -----------------------------
def categorical_section():
    st.header("Categorical Risk Estimation")
    model = load_categorical_model()

    with st.form("risk_form"):
        inputs = {}
        for col in model_columns:
            options = list(encoding_maps[col].keys()) if not callable(encoding_maps[col]) else None
            if options:
                inputs[col] = st.selectbox(f"{col.replace('_', ' ').title()}:", options)
            else:
                inputs[col] = st.number_input(f"{col.replace('_', ' ').title()} (20–74):", min_value=20.0, max_value=74.0)

        submitted = st.form_submit_button("Estimate Risk")
        if submitted:
            try:
                encoded = {}
                for col, val in inputs.items():
                    mapping = encoding_maps[col]
                    encoded[col] = mapping(val) if callable(mapping) else mapping[val]

                df = pd.DataFrame([encoded])[model_columns]
                prob = model.predict_proba(df)[0][1]
                st.success(f"Estimated Breast Cancer Risk: **{round(prob * 100, 2)}%**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

def xai_section():
    st.header("Explainable AI (XAI) 🔍")
    model = load_xai_model()

    uploaded_file = st.file_uploader("📤 Upload an image for XAI", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)

        if st.button("Generate XAI"):
            with st.spinner("Generating Grad-CAM..."):
                gradcam_img, prediction = generate_gradcam(img, model)
                st.image(gradcam_img, caption=f"Prediction: {prediction[1]} ({prediction[2]*100:.2f}%)", use_container_width=True)

def generate_gradcam(img, model):
    img_resized = img.resize((224, 224))
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    last_conv_layer = model.get_layer("Conv_1")
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = np.uint8(255 * heatmap)

    heatmap = Image.fromarray(heatmap).resize((224, 224))
    heatmap = np.array(heatmap)
    heatmap = np.uint8(plt.cm.jet(heatmap / 255.0)[:, :, :3] * 255)

    original_img = np.array(img_resized).astype("uint8")
    superimposed_img = 0.6 * original_img + 0.4 * heatmap
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img, decode_predictions(preds, top=1)[0][0]

# ----------------------------- Main -----------------------------
def main():
    st.set_page_config(page_title="Breast Cancer AI Tool", layout="wide")
    st.title("AI-Powered Breast Cancer Care Cycle 🩺")
    choice = st.radio("Choose a module", ["Risk Estimation", "X-AI"])
    if choice == "Risk Estimation":
        categorical_section()
    elif choice == "X-AI":
        xai_section()

if __name__ == "__main__":
    main()
