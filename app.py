import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import json # Importar la librería json

IMG_SIZE = (224, 224)

# --- Ruta del modelo ---
# Si lo editaste en la celda anterior, Streamlit lo leerá de una variable de entorno opcional
MODEL_PATH = os.environ.get("DERMAI_MODEL_PATH", "dermai_model.h5")

# --- Cargar el modelo con caché ---
@st.cache_resource(show_spinner=True)
def load_model_cached(path):
    model = tf.keras.models.load_model(path)
    return model

try:
    model = load_model_cached(MODEL_PATH)
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"No se pudo cargar el modelo en '{MODEL_PATH}'. Error: {e}")
    st.stop()

# --- Funciones de utilidad ---
def preprocess_image(pil_img, size=IMG_SIZE):
    img = pil_img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0) # (1, H, W, 3)
    return arr

ACTION_MSG = {
    "Benigna": "**Seguimiento en casa**: vuelve a tomar una foto en 30–60 días y cuida la exposición solar.",
    "Sospechosa": "**Monitoreo cercano**: si cambia de forma/color/tamaño, consulta. Considera pedir evaluación profesional.",
    "Maligna": "**Consulta prioritaria**: te recomendamos acudir con un dermatólogo lo antes posible.",
}

# --- Entrada de imagen: cámara o archivo ---
st.subheader("1) Elige cómo subir la imagen")
modo = st.radio("Fuente de imagen:", ["Cámara", "Galería/Archivo"], horizontal=True)

uploaded_image = None

if modo == "Cámara":
    img_data = st.camera_input("Toma una foto nítida del lunar/lesión")
    if img_data is not None:
        uploaded_image = Image.open(img_data)
else:
    file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if file is not None:
        uploaded_image = Image.open(file)

st.subheader("2) Analizar")
if uploaded_image is not None:
    st.image(uploaded_image, caption="Imagen recibida", use_column_width=True)
    if st.button("Analizar imagen"):
        with st.spinner("Procesando…"):
            x = preprocess_image(uploaded_image)
            preds = model.predict(x)[0] # vector de 3 probabilidades
            probs = preds / (preds.sum() + 1e-8)
            best_idx = np.argmax(probs) # Obtener el índice de la clase con mayor probabilidad

            # Cargar las etiquetas desde el archivo JSON
            labels_path = "labels.json" # Asegúrate de que este archivo exista
            try:
                with open(labels_path, "r") as f:
                    LABELS = json.load(f)
            except FileNotFoundError:
                st.error(f"Archivo de etiquetas no encontrado en {labels_path}")
                st.stop()

            predicted_class = LABELS[best_idx]
            confidence = probs[best_idx]

            st.write(f"**Predicción:** {predicted_class}")
            st.write(f"**Confianza:** {confidence:.2f}")
            st.write(f"**Mensaje orientativo:** {ACTION_MSG.get(predicted_class, 'Consulta a un profesional.')}") # Usar .get para evitar errores si la clave no existe

            # Mostrar probabilidades por clase
            st.write("**Probabilidades por clase:**")
            probs_dict = {LABELS[i]: p for i, p in enumerate(probs)}
            st.write(probs_dict)
