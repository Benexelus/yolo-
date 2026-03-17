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
