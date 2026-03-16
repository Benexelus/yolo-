import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# App-Titel
st.title("🚀 YOLOv8 Object Detection")
st.markdown("""
Detektiere Objekte in Bildern mit **YOLOv8** (CPU-only)
""")

# Sidebar-Einstellungen
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.01
)

# Modell laden (automatisch CPU wenn keine GPU verfügbar)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Nano-Version (kleinstes Modell)

model = load_model()

# Bild-Upload
uploaded_file = st.file_uploader(
    "Bild hochladen", 
    type=["jpg", "jpeg", "png"],
    help="Lade ein Bild hoch für die Objekterkennung"
)

if uploaded_file is not None:
    # Bild verarbeiten
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # OpenCV Farbkonvertierung (RGB -> BGR)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Detection durchführen
    results = model.predict(
        source=img_cv,
        conf=confidence_threshold,
        device="cpu"  # Erzwinge CPU-Nutzung
    )
    
    # Ergebnisse visualisieren
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Bild anzeigen
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(annotated_img, caption="Detection", use_column_width=True)
    
    # Detection-Ergebnisse anzeigen
    st.subheader("Detektierte Objekte")
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            conf = float(box.conf)
            st.write(f"- {label} (Confidence: {conf:.2f})")

# Download-Link für Modell-Cache
st.sidebar.markdown("""
**Hinweis:** Beim ersten Start wird das YOLOv8n-Modell heruntergeladen (~20MB).
""")
