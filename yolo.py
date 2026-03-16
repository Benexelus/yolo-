import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Signal-Fehler vermeiden (für Streamlit Cloud)
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"  # Deaktiviere OpenMP-Multithreading
os.environ["MKL_NUM_THREADS"] = "1"  # Deaktiviere MKL-Multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Deaktiviere OpenBLAS-Multithreading

# Titel der App
st.title("🔍 Fundbüro-Verwaltung")
st.write("Lade Bilder hoch, füge Personalien hinzu und verwalte Fundgegenstände.")

# YOLO-Modell cachen (verhindert Thread-Probleme)
@st.experimental_singleton
def load_yolo_model():
    model = YOLO("yolov8n.pt")
    return model

# Bild klassifizieren
def classify_image(image):
    model = load_yolo_model()
    results = model.predict(image, num_threads=1)  # Setze num_threads=1
    return results[0].boxes

# Hauptfunktion
def main():
    uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # Personalien-Formular
        with st.form("personalien_form"):
            name = st.text_input("Name")
            wohnort = st.text_input("Wohnort")
            datum = st.date_input("Datum")
            beschreibung = st.text_area("Beschreibung")
            
            if st.form_submit_button("Speichern"):
                st.success(f"Daten für {name} gespeichert!")
                st.json({  # Beispiel-Ausgabe
                    "Name": name,
                    "Wohnort": wohnort,
                    "Datum": str(datum),
                    "Beschreibung": beschreibung
                })

        # Klassifizierung starten
        if st.button("Bild analysieren"):
            with st.spinner("Analysiere Bild..."):
                boxes = classify_image(image)
                st.write("**Erkannte Objekte:**")
                for box in boxes:
                    st.write(f"- {box.cls}: {box.conf:.2f} Konfidenz")

if __name__ == "__main__":
    main()  # Läuft im Haupt-Thread
