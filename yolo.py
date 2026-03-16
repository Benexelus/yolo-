import streamlit as st
import cv2
import numpy as np
from PIL import Image
import subprocess
import sys

# Pakete sicher installieren (falls requirements.txt fehlschlägt)
def install_packages():
    required = [
        "streamlit==1.32.0",
        "ultralytics==8.0.0",
        "Pillow==10.1.0",
        "opencv-python-headless==4.8.0.74",
        "numpy==1.24.3",
        "torch==2.0.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu"
    ]
    for package in required:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package.split(" ")[0]])
        except:
            st.error(f"⚠️ Konnte {package} nicht installieren!")

install_packages()

# Haupt-App
try:
    from ultralytics import YOLO
    
    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    def draw_boxes(image, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    st.title("YOLOv8 Objekterkennung")
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        model = load_model()
        results = model(image)
        output_image = draw_boxes(image.copy(), results)
        st.image(output_image, caption="Ergebnis", use_column_width=True)

except Exception as e:
    st.error(f"❌ Kritischer Fehler: {str(e)}")
    st.write("Bitte überprüfe die Logs für Details.")
