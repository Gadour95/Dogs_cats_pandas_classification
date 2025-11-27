import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Charger le modèle TFLite
interpreter = tflite.Interpreter(model_path="mon_modele_animaux.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["chat", "chien", "panda"]

st.title("🐾 Prédiction Animaux (TFLite)")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image chargée", use_column_width=True)

    # Prétraitement
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Prédiction
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx] * 100

    st.subheader(f"👉 Résultat : **{labels[predicted_idx]}**")
    st.write(f"Confiance : **{confidence:.2f}%**")

    # Probas
    st.write("### 📊 Probabilités")
    for i, p in enumerate(predictions):
        if i == predicted_idx:
            st.markdown(f"**{labels[i]} : {p*100:.2f}%**")
        else:
            st.write(f"{labels[i]} : {p*100:.2f}%")
