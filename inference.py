#!/usr/bin/env python3
"""
Inférence - Segmentation agricole UAV (CropAndWeed, 5 classes)

Charge un modèle entraîné (U-Net ou DeepLabV3) et segmente une image à sa résolution d'origine.

    python inference.py parcelle.jpg --poids meilleur_unet.pth --architecture unet
    streamlit run app.py            # interface web : compare les deux modèles
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

# Doit être identique à l'entraînement dans le notebook (même ordre des classes).
NOMS_CLASSES = ["background", "maize", "autres_cultures", "mauvaises_herbes", "vegetation"]
COULEURS = ["#8B5E3C", "#3E8E41", "#1B9AAA", "#D1495B", "#9E9E9E"]
CULTURES = ("maize", "autres_cultures")
ADVENTICES = ("mauvaises_herbes",)    # "vegetation" (ambiguë) est exclue du taux d'infestation

TAILLE_MODELE = 256
MOYENNE = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ECART_TYPE = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in COULEURS], dtype=np.float32)
ID_FOND = 0


def charger_image(source):
    """Chemin, image PIL ou tableau numpy -> tableau RGB uint8."""
    if isinstance(source, (str, Path)):
        return np.asarray(Image.open(source).convert("RGB"))
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"))
    return np.asarray(source)[..., :3].astype(np.uint8)


def colorier(masque):
    """Masque de classes (H, W) -> image couleur RGB uint8."""
    return (PALETTE[masque] * 255).astype(np.uint8)


def superposer(image, masque, alpha=0.5):
    """Superpose les classes sur l'image ; le sol reste transparent."""
    image_f = image.astype(np.float32) / 255.0
    melange = np.where((masque != ID_FOND)[..., None], (1 - alpha) * image_f + alpha * PALETTE[masque], image_f)
    return (np.clip(melange, 0, 1) * 255).astype(np.uint8)


def calculer_stats(masque):
    """Proportion de pixels par classe (%) et taux d'infestation = adventices / (cultures + adventices)."""
    effectifs = np.bincount(masque.ravel(), minlength=len(NOMS_CLASSES))
    pixels = dict(zip(NOMS_CLASSES, effectifs.tolist()))
    cultures = sum(pixels[c] for c in CULTURES)
    adventices = sum(pixels[c] for c in ADVENTICES)
    return {
        "proportion_pct": {n: 100.0 * v / effectifs.sum() for n, v in pixels.items()},
        "taux_infestation_pct": 100.0 * adventices / (cultures + adventices) if cultures + adventices else None,
    }


class SegmentateurUAV:
    def __init__(self, poids, architecture="unet", encodeur="resnet34", device=None):
        constructeurs = {"unet": smp.Unet, "deeplabv3": smp.DeepLabV3}
        cle = architecture.lower().replace("-", "")
        if cle not in constructeurs:
            raise ValueError(f"Architecture inconnue : {architecture} (unet ou deeplabv3)")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.modele = constructeurs[cle](encoder_name=encodeur, encoder_weights=None,
                                         in_channels=3, classes=len(NOMS_CLASSES))
        self.modele.load_state_dict(torch.load(poids, map_location="cpu", weights_only=True))
        self.modele.to(self.device).eval()

    @torch.no_grad()
    def predire(self, source):
        """Retourne le masque de classes (H, W) uint8, à la résolution de l'image d'origine."""
        image = charger_image(source)
        hauteur, largeur = image.shape[:2]

        # Même prétraitement qu'à l'entraînement : 256x256 bilinéaire + normalisation ImageNet
        petite = Image.fromarray(image).resize((TAILLE_MODELE, TAILLE_MODELE), Image.Resampling.BILINEAR)
        x = (np.asarray(petite, dtype=np.float32) / 255.0 - MOYENNE) / ECART_TYPE
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            logits = self.modele(x)
        # On agrandit les scores (pas le masque) : contours plus lisses
        logits = F.interpolate(logits.float(), size=(hauteur, largeur), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Segmentation agricole UAV")
    ap.add_argument("image")
    ap.add_argument("--poids", required=True)
    ap.add_argument("--architecture", default="unet", choices=["unet", "deeplabv3"])
    ap.add_argument("--sortie", default=None, help="image de sortie (défaut : <image>_superposition.png)")
    args = ap.parse_args()

    image = charger_image(args.image)
    masque = SegmentateurUAV(args.poids, args.architecture).predire(image)

    sortie = args.sortie or f"{Path(args.image).stem}_superposition.png"
    Image.fromarray(superposer(image, masque)).save(sortie)

    stats = calculer_stats(masque)
    for nom, pct in stats["proportion_pct"].items():
        print(f"{nom:18s} {pct:6.2f} %")
    taux = stats["taux_infestation_pct"]
    print("Taux d'infestation :", "n/a" if taux is None else f"{taux:.1f} %")
    print("Image enregistrée :", sortie)


if __name__ == "__main__":
    main()
