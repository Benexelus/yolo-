import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# ===============================================
# KRITISCHE EINSTELLUNGEN (Behebt den Signal-Fehler)
# ===============================================
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)  # PyTorch auf Single-Thread forcieren

# ===============================================
# APP-LOGIK
# ===============================================
st.title("🔍 Fundbüro-Verwaltung")

@st.experimental_singleton  # Besser als cache_resource für YOLO
def load_model():
    """Lädt YOLO-Modell mit Thread-Sicherheit"""
    model = YOLO("yolov8n.pt")
    # Warmup-Run (wichtig für Thread-Synchronisation)
    model.predict("dummy.jpg", imgsz=64, verbose=False, num_threads=1)  
    return model

def analyze_image(image):
    """Bildanalyse mit Thread-Sicherheit"""
    model = load_model()
    results = model.predict(image, num_threads=1, verbose=False)
    return results[0].boxes

# ===============================================
# UI & HAUPTHANDLUNG
# ===============================================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Hochgeladenes Bild")
    
    if st.button("Analyse starten"):
        with st.spinner("Analysiere..."):
            try:
                boxes = analyze_image(img)
                st.success("Ergebnisse:")
                for box in boxes:
                    st.write(f"- Objekt: {box.cls}, Konfidenz: {box.conf:.2f}")
            except Exception as e:
                st.error(f"Fehler: {str(e)}")
