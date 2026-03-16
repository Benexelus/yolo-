import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Titel der App
st.title("🔍 Fundbüro-Verwaltung")
st.write("Lade Bilder hoch, füge Personalien hinzu und verwalte Fundgegenstände.")

# YOLO-Modell laden (integriertes Modell)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Integriertes Modell

# Bild mit YOLO klassifizieren
def classify_image(image):
    model = load_yolo_model()
    results = model.predict(image)
    return results[0].boxes

# Bild hochladen und Personalien hinzufügen
def upload_and_classify():
    uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        
        # Bild speichern
        image_path = os.path.join("images", uploaded_file.name)
        os.makedirs("images", exist_ok=True)
        image.save(image_path)
        
        # Personalien eingeben
        with st.form("personalien_form"):
            st.write("**Personalien hinzufügen**")
            name = st.text_input("Name")
            wohnort = st.text_input("Wohnort")
            datum = st.date_input("Datum")
            beschreibung = st.text_area("Beschreibung")
            submitted = st.form_submit_button("Speichern")
            
            if submitted:
                st.success(f"Daten für {name} erfolgreich gespeichert!")
                st.write(f"**Wohnort:** {wohnort}")
                st.write(f"**Datum:** {datum}")
                st.write(f"**Beschreibung:** {beschreibung}")
        
        # Bild klassifizieren
        if st.button("Bild klassifizieren"):
            boxes = classify_image(image)
            st.write("**Erkannte Objekte:**")
            for box in boxes:
                st.write(f"- {box.cls}: {box.conf:.2f} (Box: {box.xyxy})")

# Galerie anzeigen
def show_gallery():
    st.write("**Fundbüro-Galerie**")
    if os.path.exists("images"):
        for image_file in os.listdir("images"):
            image_path = os.path.join("images", image_file)
            st.image(image_path, caption=image_file, width=300)

# Hauptfunktion
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Seite", ["Bild hochladen", "Galerie"])
    
    if page == "Bild hochladen":
        upload_and_classify()
    elif page == "Galerie":
        show_gallery()

if __name__ == "__main__":
    main()
