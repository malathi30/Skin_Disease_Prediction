import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# -----------------------------
# Load hybrid model (Random Forest trained on VGG16 features)
# -----------------------------
rf_model = joblib.load("hybrid_rf_vgg16_tuned.pkl")

# Load VGG16 (without top layer for feature extraction)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# -----------------------------
# Class labels
# -----------------------------
CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

# -----------------------------
# Disease details
# -----------------------------
disease_info = {
    "Actinic keratosis": {
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. It can sometimes develop into skin cancer.",
        "treatment": [
            "Apply prescribed topical creams (fluorouracil or imiquimod).",
            "Avoid direct sunlight and use sunscreen daily.",
            "Cryotherapy or laser therapy may be recommended."
        ]
    },
    "Atopic Dermatitis": {
        "description": "A chronic condition causing itchy, inflamed skin (eczema).",
        "treatment": [
            "Use gentle moisturizers and avoid harsh soaps.",
            "Apply corticosteroid creams during flare-ups.",
            "Keep skin hydrated and avoid triggers like stress or dust."
        ]
    },
    "Benign keratosis": {
        "description": "A non-cancerous skin growth appearing as brown, black, or light tan lesions.",
        "treatment": [
            "Usually harmless, but can be removed for cosmetic reasons.",
            "Cryotherapy or laser removal if irritated.",
            "Avoid scratching or picking at the lesion."
        ]
    },
    "Dermatofibroma": {
        "description": "A small, firm, benign bump commonly found on the legs or arms.",
        "treatment": [
            "Generally requires no treatment.",
            "Can be surgically removed if painful or for cosmetic reasons.",
            "Use moisturizing creams to keep skin soft."
        ]
    },
    "Melanocytic nevus": {
        "description": "Commonly known as a mole, a benign cluster of pigmented skin cells.",
        "treatment": [
            "Monitor for any change in shape, color, or border.",
            "Consult a dermatologist if the mole changes or bleeds.",
            "Avoid sunburns to reduce risk of complications."
        ]
    },
    "Melanoma": {
        "description": "A serious skin cancer that develops from melanocytes, often as a new or changing mole.",
        "treatment": [
            "Immediate medical attention is necessary.",
            "Surgical removal is the primary treatment.",
            "May require immunotherapy or chemotherapy for advanced cases."
        ]
    },
    "Squamous cell carcinoma": {
        "description": "A common skin cancer caused by long-term UV exposure, often forming scaly red patches or open sores.",
        "treatment": [
            "Early surgical removal is very effective.",
            "Radiation or chemotherapy may be needed for advanced cases.",
            "Regular check-ups to monitor for recurrence."
        ]
    },
    "Tinea Ringworm Candidiasis": {
        "description": "A fungal infection causing circular, itchy, red patches on the skin.",
        "treatment": [
            "Apply antifungal creams (clotrimazole or terbinafine).",
            "Keep affected areas dry and clean.",
            "Avoid sharing personal items like towels."
        ]
    },
    "Vascular lesion": {
        "description": "An abnormal growth or formation of blood vessels on or under the skin.",
        "treatment": [
            "Laser therapy may help reduce visibility.",
            "Consult a dermatologist if painful or growing.",
            "Avoid trauma or scratching over the lesion."
        ]
    }
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hybrid CNN + RF Skin Disease Predictor", layout="wide")
st.title("🧴 Hybrid AI Skin Disease Predictor (VGG16 + Random Forest)")
st.write("Upload a clear skin image to predict the disease, view details, and treatment suggestions.")

uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])

# -----------------------------
# Prediction section
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for VGG16
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Remove alpha channel if PNG
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract CNN features (VGG16)
    features = vgg_model.predict(img_array)
    flattened_features = features.reshape(features.shape[0], -1)

    # Predict using Random Forest
    prediction = rf_model.predict(flattened_features)
    predicted_class = CLASS_NAMES[prediction[0]]

    # Display prediction
    st.subheader(f"✅ Predicted Disease: {predicted_class}")

    # Show description & treatment
    info = disease_info.get(predicted_class)
    if info:
        st.markdown(f"**🩸 Description:** {info['description']}")
        st.markdown("**💊 Suggested Treatments:**")
        for point in info["treatment"]:
            st.write(f"- {point}")
    else:
        st.warning("No detailed information available for this disease.")
