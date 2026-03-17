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

    # ===== NEU: Schwellenwert-Slider =====
    threshold = st.slider(
        "Minimale Sicherheit zum Speichern des Bildes (%)",
        min_value=50,
        max_value=99,
        value=90
    ) / 100

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

        # ===== Speicherung abhängig vom Slider =====
        if best["score"] >= threshold:

            label = best["label"].replace(" ", "_")

            # Kategorieordner erstellen
            category_folder = os.path.join(GALLERY_DIR, label)
            os.makedirs(category_folder, exist_ok=True)

            # Dateiname erzeugen
            existing = len(os.listdir(category_folder))
            filename = f"{label}_{existing+1}.png"
            filepath = os.path.join(category_folder, filename)

            image.save(filepath)

            st.success(
                f"Bild wurde in der Kategorie '{label}' gespeichert "
                f"(Schwelle: {int(threshold*100)}%)"
            )
        else:
            st.warning(
                f"Bild nicht gespeichert – KI war nur {best['score']:.1%} sicher "
                f"(Schwelle: {int(threshold*100)}%)"
            )


# ===== Seite: Galerie (alle Bilder anzeigen) =====
elif page == "Galerie":

    st.title("Galerie")

    all_images = []

    for folder in os.listdir(GALLERY_DIR):
        folder_path = os.path.join(GALLERY_DIR, folder)

        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                all_images.append(os.path.join(folder_path, img))

    if len(all_images) > 0:

        cols = st.columns(4)

        for i, path in enumerate(all_images):

            caption = os.path.basename(path).replace("_", " ").replace(".png", "")

            with cols[i % 4]:
                st.image(path, caption=caption)

    else:
        st.write("Noch keine Bilder in der Galerie.")


# ===== Seite: Bilder suchen (Kategorien anzeigen) =====
elif page == "Bilder suchen":

    st.title("Bilder nach Kategorie durchsuchen")

    categories = [
        folder for folder in os.listdir(GALLERY_DIR)
        if os.path.isdir(os.path.join(GALLERY_DIR, folder))
    ]

    if categories:

        selected_category = st.selectbox(
            "Kategorie auswählen",
            categories
        )

        category_path = os.path.join(GALLERY_DIR, selected_category)
        images = os.listdir(category_path)

        st.subheader(f"Bilder in Kategorie: {selected_category}")

        cols = st.columns(4)

        for i, img in enumerate(images):

            path = os.path.join(category_path, img)

            with cols[i % 4]:
                st.image(
                    path,
                    caption=img.replace("_", " ").replace(".png", "")
                )

    else:
        st.write("Noch keine Kategorien vorhanden.")
