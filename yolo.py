import streamlit as st
from transformers import pipeline
from PIL import Image
import os

# ===== Modell laden =====
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_classifier()

# ===== Galerie Ordner =====
GALLERY_DIR = "gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

# ===== Sidebar Navigation =====
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite wählen",
    ["Bilder hochladen", "Galerie", "Bilder suchen"]
)

# ===== Seite: Bilder hochladen =====
if page == "Bilder hochladen":

    st.title("Bild hochladen → KI sagt was drauf ist")
    st.write("Funktioniert mit fast allen Alltagsdingen (ImageNet-Klassen)")

    uploaded_file = st.file_uploader(
        "Wähl ein JPG/PNG/JPEG aus",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Dein hochgeladenes Bild", use_column_width=True)

        with st.spinner("Analysiere... (kann 2–10 Sekunden dauern)"):
            results = classifier(image)

        st.success("Top-Ergebnisse:")

        for i, res in enumerate(results[:5], 1):
            st.write(f"{i}. **{res['label']}** – {res['score']:.1%} sicher")

        best = results[0]

        if best["score"] >= 0.90:
            label = best["label"].replace(" ", "_")
            filename = f"{label}.png"
            filepath = os.path.join(GALLERY_DIR, filename)

            image.save(filepath)

            st.success(f"Bild wurde in der Galerie gespeichert als: {filename}")


# ===== Seite: Galerie =====
elif page == "Galerie":

    st.title("Galerie (≥90% sichere Bilder)")

    images = os.listdir(GALLERY_DIR)

    if len(images) > 0:

        cols = st.columns(4)

        for i, img in enumerate(images):
            path = os.path.join(GALLERY_DIR, img)

            with cols[i % 4]:
                st.image(
                    path,
                    caption=img.replace("_", " ").replace(".png", "")
                )

    else:
        st.write("Noch keine Bilder in der Galerie.")


# ===== Seite: Bilder suchen =====
elif page == "Bilder suchen":

    st.title("Bilder suchen")

    search = st.text_input("Nach Objekt suchen (z.B. dog, cat, car)")

    images = os.listdir(GALLERY_DIR)

    if search:

        results = [
            img for img in images
            if search.lower() in img.lower()
        ]

        if results:

            cols = st.columns(4)

            for i, img in enumerate(results):
                path = os.path.join(GALLERY_DIR, img)

                with cols[i % 4]:
                    st.image(
                        path,
                        caption=img.replace("_", " ").replace(".png", "")
                    )

        else:
            st.write("Keine passenden Bilder gefunden.")
