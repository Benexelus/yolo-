import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# App-Konfiguration
st.set_page_config(layout="wide")
st.title("🚀 Ultralytics YOLOv8 Objekterkennung")
st.markdown("""
**Keine Torch-Abhängigkeiten** • Läuft auf CPU/GPU • Einfache Installation
""")

# Modell-Cache
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt')  # Automatische Geräteerkennung
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None

# Bildverarbeitung
def process_image(upload, model):
    img = Image.open(upload)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Inferenz mit automatischer Geräteerkennung
    results = model(img_bgr, verbose=False)  
    
    # Visualisierung
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return img, annotated_rgb, results[0]

# UI
upload = st.file_uploader("Bild hochladen", type=["jpg","png","jpeg"])
if upload:
    model = load_model()
    if model:
        with st.spinner("Analysiere Bild..."):
            original, detected, results = process_image(upload, model)
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_column_width=True)
        with col2:
            st.image(detected, caption="Erkannte Objekte", use_column_width=True)
            
        st.divider()
        st.subheader("Detektionsergebnisse")
        for box in results.boxes:
            st.write(f"- {results.names[int(box.cls)]} (Confidence: {box.conf:.2f})")
