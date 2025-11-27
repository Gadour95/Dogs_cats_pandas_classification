import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("mon_modele_animaux.h5")
    return model

model = load_my_model()
st.success("Model loaded successfully!")


labels = ["chat", "chien", "panda"]


def predict_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    return predicted_class_idx, confidence, predictions[0]


st.title("🔍 Animal Classifier (Chat / Chien / Panda)")
st.write("Upload an image and the model will predict the class.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Image uploaded", use_container_width=True)


    if st.button("🔮 Predict"):
        with st.spinner("Prediction in progress..."):
            pred_idx, confidence, probs = predict_image(img)

        st.subheader(f"Prediction : **{labels[pred_idx]}** 🐾")
        st.write(f"Confiance : **{confidence:.2f}%**")

        st.subheader("📊 Probabilities")
        for i, prob in enumerate(probs):
            st.write(f"**{labels[i]}** : {prob*100:.2f}%")
