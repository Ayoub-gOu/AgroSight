"""
Interface Streamlit : segmente une image UAV avec chaque modèle disponible et les compare.

    uv run streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from inference import COULEURS, NOMS_CLASSES, SegmentateurUAV, calculer_stats, charger_image, colorier, superposer

DOSSIER = Path(__file__).parent
MODELES = {"U-Net": ("unet", DOSSIER / "meilleur_unet.pth"),
           "DeepLabV3": ("deeplabv3", DOSSIER / "meilleur_deeplabv3.pth")}
disponibles = {nom: v for nom, v in MODELES.items() if v[1].is_file()}

st.set_page_config(page_title="Segmentation agricole UAV", layout="wide")
st.title("Segmentation agricole par drone")

if not disponibles:
    st.error("Aucun fichier de poids trouvé. Placez `meilleur_unet.pth` et/ou `meilleur_deeplabv3.pth` à côté de `app.py`.")
    st.stop()


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger(architecture, poids):
    return SegmentateurUAV(poids, architecture)


def octets_png(tableau):
    tampon = io.BytesIO()
    Image.fromarray(tableau).save(tampon, format="PNG")
    return tampon.getvalue()


st.markdown(" ".join(
    f"<span style='background:{c};color:white;padding:2px 8px;border-radius:4px;margin-right:4px'>{n}</span>"
    for n, c in zip(NOMS_CLASSES, COULEURS)), unsafe_allow_html=True)

fichier = st.file_uploader("Image UAV", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if fichier is None:
    st.info("Déposez une image de champ vue du dessus.")
    st.stop()

image = charger_image(Image.open(fichier))
with st.spinner("Segmentation..."):
    masques = {nom: charger(arch, str(poids)).predire(image) for nom, (arch, poids) in disponibles.items()}

# Image d'origine et superposition de chaque modèle
colonnes = st.columns(1 + len(masques))
colonnes[0].image(image, caption="Image UAV")
for col, (nom, masque) in zip(colonnes[1:], masques.items()):
    superposition = superposer(image, masque)
    col.image(superposition, caption=nom)
    col.download_button(f"Télécharger ({nom})", octets_png(superposition), f"{Path(fichier.name).stem}_{nom}.png",
                        "image/png", key=f"dl_{nom}")

if st.checkbox("Afficher les cartes de classes seules"):
    colonnes = st.columns(len(masques))
    for col, (nom, masque) in zip(colonnes, masques.items()):
        col.image(colorier(masque), caption=f"Carte de classes - {nom}")

# Proportions par classe et taux d'infestation
stats = {nom: calculer_stats(masque) for nom, masque in masques.items()}
st.subheader("Proportion de pixels par classe (%)")
st.dataframe(pd.DataFrame({nom: s["proportion_pct"] for nom, s in stats.items()}).round(2))

colonnes = st.columns(len(stats))
for col, (nom, s) in zip(colonnes, stats.items()):
    taux = s["taux_infestation_pct"]
    col.metric(f"Taux d'infestation - {nom}", "n/a" if taux is None else f"{taux:.1f} %")
