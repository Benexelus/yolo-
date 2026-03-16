import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Titel der App
st.title("🏞️ Bildverarbeitung mit OpenCV")

# Beschreibung der App
st.markdown("""
Diese App verwendet **OpenCV** und **Pillow**, um Bilder zu verarbeiten. Du kannst ein Bild hochladen und es in Graustufen umwandeln.
""")

# Datei-Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Bild in ein PIL-Image-Objekt laden
    image = Image.open(uploaded_file)

    # Bild als NumPy-Array konvertieren (für OpenCV)
    image_np = np.array(image)

    # Bild von RGB in BGR umwandeln (OpenCV verwendet BGR statt RGB)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Originalbild anzeigen
    st.subheader("Originalbild")
    st.image(image, caption="Originalbild", use_column_width=True)

    # Graustufenbild erstellen
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Graustufenbild anzeigen
    st.subheader("Graustufenbild")
    st.image(gray_image, caption="Graustufenbild", use_column_width=True)

    # Option: Kantenerkennung mit Canny
    st.markdown("""
    **Optional:** Du kannst auch Kanten im Bild erkennen.
    """)
    edges = cv2.Canny(gray_image, 100, 200)  # Canny-Kantenerkennung

    # Kantenerkennungsbild anzeigen
    st.subheader("Kantenerkennung (Canny)")
    st.image(edges, caption="Kantenerkennung", use_column_width=True)

    # Option: Bild speichern
    st.markdown("""
    Du kannst das verarbeitete Bild herunterladen.
    """)
    # Graustufenbild als Datei speichern
    gray_image_pil = Image.fromarray(gray_image)
    gray_image_pil.save("gray_image.png")

    # Download-Link für das Graustufenbild
    with open("gray_image.png", "rb") as file:
        btn = st.download_button(
            label="Graustufenbild herunterladen",
            data=file,
            file_name="gray_image.png",
            mime="image/png"
        )
