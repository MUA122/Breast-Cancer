import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hardcoded column names (from index.html/app.py logic)
model_columns = [
    'age_group_5_years',
    'race_eth',
    'first_degree_hx',
    'age_menarche',
    'age_first_birth',
    'BIRADS_breast_density',
    'current_hrt',
    'menopaus',
    'bmi_group',
    'biophx'
]

# Encoding mappings
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

# Load trained model
@st.cache_resource
def load_categorical_model():
    return joblib.load(r"templates/full_model.pkl")

# ---- MODEL LOADING FUNCTIONS (Diagnosis) ----

@st.cache_resource
def load_xray_models():
    TRAIN_DIR = r'C:\Users\mahmo\OneDrive\Desktop\pro1\Breast Cancer Dataset\training'
    gen = ImageDataGenerator(rescale=1/255.0)
    train_data = gen.flow_from_directory(TRAIN_DIR, target_size=(240, 240), batch_size=64, class_mode='categorical')
    class_map = {v: k for k, v in train_data.class_indices.items()}
    return {
        "MobileNetV2": (load_model("MobileNetV2_modelx2.h5"), 0.9781),
        "ResNet50":    (load_model("ResNet50_modelx2.h5"), 0.8925),
        "DenseNet169": (load_model("DenseNet169_modelx2.h5"), 0.9770)
    }, class_map

@st.cache_resource
def load_microscopic_models():
    TRAIN_DIR = r'C:\Users\mahmo\OneDrive\Desktop\pro1\dataset_cancer_v1\classificacao_binaria\trainig'
    gen = ImageDataGenerator(rescale=1/255.0)
    train_data = gen.flow_from_directory(TRAIN_DIR, target_size=(240, 240), batch_size=64, class_mode='categorical')
    class_map = {v: k for k, v in train_data.class_indices.items()}
    return {
        "DenseNet169": (load_model("DenseNet169_model.h5"), 0.9674),
        "MobileNetV2": (load_model("MobileNetV2_model.h5"), 0.8546),
        "ResNet50":    (load_model("ResNet50_model.h5"), 0.7243)
    }, class_map

@st.cache_resource
def load_skin_models():
    TRAIN_DIR = r'C:\Users\mahmo\OneDrive\Desktop\pro1\skin\melanoma_cancer_dataset\train'
    gen = ImageDataGenerator(rescale=1/255.0)
    train_data = gen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=64, class_mode='categorical')
    class_map = {v: k for k, v in train_data.class_indices.items()}
    return {
        "InceptionV3":  (load_model("InceptionV3_model_skin.h5"), 0.902),
        "DenseNet201":  (load_model("DenseNet201_model_skin.h5"), 0.911)
    }, class_map

@st.cache_resource
def load_all_datasets():
    xray, cm_x = load_xray_models()
    micro, cm_m = load_microscopic_models()
    skin, cm_s = load_skin_models()
    return {
        "X-Ray":       (xray, cm_x),
        "Microscopic": (micro, cm_m),
        "Skin Cancer": (skin, cm_s)
    }

def predict_image(model, img: Image.Image, class_map):
    shape = model.input_shape
    if isinstance(shape, list): shape = shape[0]
    _, H, W, _ = shape
    img_resized = img.resize((W, H))
    arr = np.asarray(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    return class_map[np.argmax(preds, axis=1)[0]]

def diagnosis_section():
    st.header("Diagnosis Interface")
    datasets = load_all_datasets()
    dataset_name = st.selectbox("Select Dataset Type", list(datasets.keys()))
    model_dict, class_map = datasets[dataset_name]
    model_name = st.selectbox("Select Model", list(model_dict.keys()))
    model, accuracy = model_dict[model_name]
    st.markdown(f"**Current Model Accuracy:** {accuracy*100:.2f}%")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","tiff"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Input Image", use_container_width=True)
        if st.button("Predict"):
            label = predict_image(model, img, class_map)
            st.success(f"**Prediction:** {label}")

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

def main():
    st.title("AI-Powered Breast cancer care cycle🩺")
    choice = st.radio("Choose a module", ["Diagnosis", "Categorical"])
    if choice == "Diagnosis":
        diagnosis_section()
    elif choice == "Categorical":
        categorical_section()

if __name__ == "__main__":
    main()
