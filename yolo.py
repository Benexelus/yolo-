import streamlit as st
from transformers import pipeline
from PIL import Image

# Modell einmal laden (wird gecacht)
@st.cache_resource
def load_classifier():
    # Gutes Allround-Modell (ViT base, ~86M Parameter)
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_classifier()

st.title("Bild hochladen → KI sagt was drauf ist")
st.write("Funktioniert mit fast allen Alltagsdingen (ImageNet-Klassen)")

uploaded_file = st.file_uploader("Wähl ein JPG/PNG/JPEG aus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption="Dein hochgeladenes Bild", use_column_width=True)
    
    with st.spinner("Analysiere... (kann 2–10 Sekunden dauern)"):
        # Vorhersage machen
        results = classifier(image)
    
    st.success("Top-Ergebnisse:")
    for i, res in enumerate(results[:5], 1):
        st.write(f"{i}. **{res['label']}** – {res['score']:.1%} sicher")


# ===== Ergänzung: Galerie für ≥90% sichere Bilder =====

import os

GALLERY_DIR = "gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

if uploaded_file is not None:
    best = results[0]

    if best["score"] >= 0.90:
        label = best["label"].replace(" ", "_")
        filename = f"{label}.png"
        filepath = os.path.join(GALLERY_DIR, filename)

        image.save(filepath)

        st.success(f"Bild wurde in der Galerie gespeichert als: {filename}")


# ===== Galerie anzeigen =====

st.subheader("Galerie (Bilder mit ≥90% Sicherheit)")

images = os.listdir(GALLERY_DIR)

if len(images) > 0:
    cols = st.columns(4)

    for i, img in enumerate(images):
        path = os.path.join(GALLERY_DIR, img)

        with cols[i % 4]:
            st.image(path, caption=img.replace("_", " ").replace(".png", ""))
else:
    st.write("Noch keine Bilder in der Galerie.")
