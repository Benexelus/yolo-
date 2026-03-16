import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Titel der App
st.title("📷 YOLOv8 Objekterkennung")
st.markdown("""
Detektiere Objekte in Bildern mit **YOLOv8** (ohne PyTorch).
""")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Nano-Version (kleinstes Modell)

model = load_model()

# Datei-Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Bild laden und in NumPy-Array konvertieren
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Bild von RGB in BGR umwandeln (OpenCV verwendet BGR statt RGB)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Objekterkennung durchführen
    results = model.predict(source=image_bgr)  # Ultralytics kümmert sich um den Rest

    # Ergebnisse visualisieren
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Originalbild und detektiertes Bild anzeigen
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Originalbild", use_column_width=True)
    with col2:
        st.image(annotated_image, caption="Detektiertes Bild", use_column_width=True)

    # Detektierte Objekte auflisten
    st.subheader("Detektierte Objekte")
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            confidence = float(box.conf)
            st.write(f"- **{label}** (Konfidenz: {confidence:.2f})")
