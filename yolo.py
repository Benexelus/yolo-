import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Funktion zum Laden des YOLO-Modells
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Verwende das Nano-Modell (kleinste Version)

# Funktion zum Zeichnen der Bounding Boxes
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Streamlit-App
st.title("YOLOv8 Objekterkennung")
st.write("Lade ein Bild hoch, um Objekte zu erkennen.")

# Bild hochladen
uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # YOLO-Modell laden
    model = load_model()

    # Objekte erkennen
    results = model(image)

    # Ergebnis anzeigen
    output_image = draw_boxes(image.copy(), results)
    st.image(output_image, caption="Erkannte Objekte", use_column_width=True)
