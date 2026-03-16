import streamlit as st
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os

# Titel der App
st.title("🔍 Fundbüro-Verwaltung")
st.write("Lade Bilder hoch, füge Personalien hinzu und verwalte Fundgegenstände.")

# YOLO-Modell laden
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

# Datenbank initialisieren
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS fundbuero (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            name TEXT,
            wohnort TEXT,
            datum TEXT,
            beschreibung TEXT
        )"""
    )
    conn.commit()
    conn.close()

# Bild mit YOLO klassifizieren
def classify_image(image):
    model = load_yolo_model()
    results = model.predict(image)
    return results[0].boxes

# Daten in die Datenbank speichern
def save_to_db(image_path, name, wohnort, datum, beschreibung):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO fundbuero (image_path, name, wohnort, datum, beschreibung) VALUES (?, ?, ?, ?, ?)",
        (image_path, name, wohnort, datum, beschreibung),
    )
    conn.commit()
    conn.close()

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
                save_to_db(image_path, name, wohnort, str(datum), beschreibung)
                st.success("Daten erfolgreich gespeichert!")

# Galerie anzeigen
def show_gallery():
    st.write("**Fundbüro-Galerie**")
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fundbuero")
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        st.image(row[1], caption=f"{row[2]} aus {row[3]} (Datum: {row[4]})", width=300)
        st.write(f"**Beschreibung:** {row[5]}")

# Hauptfunktion
def main():
    init_db()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Seite", ["Bild hochladen", "Galerie"])
    
    if page == "Bild hochladen":
        upload_and_classify()
    elif page == "Galerie":
        show_gallery()

if __name__ == "__main__":
    main()
