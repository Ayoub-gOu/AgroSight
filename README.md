# 🌾 Segmentation agricole par drone (UAV) — U-Net vs DeepLabV3

Segmentation sémantique multi-classes d'images de champs prises par drone, pour distinguer **les cultures, les mauvaises herbes et le sol**, et estimer un **taux d'infestation** par image.

Le projet compare deux architectures de Deep Learning (**U-Net** et **DeepLabV3**) avec le même encodeur **ResNet-34** pré-entraîné sur ImageNet, sur le jeu de données **CropAndWeed**. Il fournit :

- un **notebook** complet (préparation des données, entraînement, évaluation, robustesse) ;
- un module d'**inférence** (`inference.py`) utilisable en ligne de commande ou en bibliothèque ;
- une **application web Streamlit** (`app.py`) qui compare les deux modèles sur une même image.

---

## Sommaire

- [Contexte](#contexte)
- [Jeu de données](#jeu-de-données)
- [Classes](#classes)
- [Méthodologie](#méthodologie)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Limites et points d'attention](#limites-et-points-dattention)
- [Licence et crédits](#licence-et-crédits)

---

## Contexte

L'objectif est de cartographier des parcelles agricoles à partir d'images de drone et de distinguer différentes classes de végétation malgré la variabilité des conditions : luminosité, ombres, zones mélangées, ressemblance entre jeunes cultures et adventices.

Deux architectures sont comparées :

| | **U-Net** | **DeepLabV3** |
|---|---|---|
| Principe | Encodeur-décodeur avec *skip connections* | Convolutions dilatées + module ASPP |
| Contexte multi-échelle | Descente progressive en résolution | Plusieurs taux de dilatation en parallèle |
| Détails fins | Bien conservés à chaque niveau | Sortie basse résolution puis sur-échantillonnage |

Utiliser le **même encodeur** pour les deux permet d'attribuer la différence de performance surtout au décodeur et à l'agrégation du contexte.

## Jeu de données

[**Voxel51/CropAndWeed**](https://huggingface.co/datasets/Voxel51/CropAndWeed) (WACV 2023), chargé avec [FiftyOne](https://docs.voxel51.com/) :

- **8 034 images** annotées de 1920 × 1088 px, avec un masque sémantique par image ;
- prises **du dessus, à environ 1,1 m**, sur des champs réels en Autriche ;
- conditions variées (lumière, humidité, type de sol) ;
- 100 classes fines (espèces et stades de croissance), regroupées ici en 5 classes ;
- téléchargement complet d'environ **11,6 Go** (une seule fois).

> ⚠️ Licence **non commerciale** (recherche et enseignement).

## Classes

Les 100 classes fines sont regroupées à partir de leurs noms, avant le redimensionnement.

| Id | Classe | Contenu |
|----|--------|---------|
| 0 | `background` | Sol |
| 1 | `maize` | Maïs (culture principale) |
| 2 | `autres_cultures` | Betterave, tournesol, haricot, pois, soja, pomme de terre, citrouille… |
| 3 | `mauvaises_herbes` | Toutes les autres plantes (adventices) |
| 4 | `vegetation` | Classe de secours : plantes trop petites ou ambiguës |

Le **taux d'infestation** est calculé ainsi (la classe ambiguë `vegetation` est exclue) :

```
taux = pixels(mauvaises_herbes) / (pixels(maize) + pixels(autres_cultures) + pixels(mauvaises_herbes))
```

## Méthodologie

| Étape | Détail |
|---|---|
| Prétraitement | Regroupement 100 → 5 classes, redimensionnement à **256 × 256** (bilinéaire pour les images, plus proche voisin pour les masques), normalisation ImageNet |
| Séparation | **70 / 15 / 15 %** (entraînement / validation / test), **par session d'enregistrement** pour éviter toute fuite de données entre images quasi identiques |
| Augmentation | Flips horizontal et vertical, rotations de 90°, variation de luminosité (×0,7 à ×1,3) |
| Déséquilibre | Entropie croisée pondérée par classe (poids en racine inverse de la fréquence, calculés sur l'entraînement uniquement) |
| Perte | `0,5 × CE pondérée + 0,5 × Dice` |
| Optimiseur | AdamW (lr = 3e-4, weight decay = 1e-4), Cosine Annealing |
| Entraînement | 20 époques max, arrêt anticipé après 6 époques sans amélioration, batch de 8, précision mixte sur GPU |
| Sélection du modèle | Meilleur **mIoU plantes** sur la validation |
| Métriques | IoU par classe, mIoU, mIoU hors fond, **mIoU plantes**, F1 macro, exactitude pixel |
| Robustesse | Évaluation par condition d'acquisition réelle (lumière, humidité, sol) et par luminosité simulée (50 % à 150 %) |

**Pourquoi le mIoU plantes ?** Le sol représente environ 97 % des pixels : l'exactitude pixel est donc peu informative (un modèle qui prédit toujours « sol » obtient déjà environ 97 %). Le mIoU plantes (moyenne sur `maize`, `autres_cultures`, `mauvaises_herbes`) est la métrique la plus proche de l'objectif : distinguer cultures et mauvaises herbes. Les IoU sont calculées à partir d'une matrice de confusion cumulée sur tout le jeu.

## Structure du dépôt

```
.
├── Segmentation_Agricole_UAV_CropAndWeed_complet.ipynb   # Données, entraînement, évaluation
├── inference.py            # Module d'inférence + CLI
├── app.py                  # Interface Streamlit (compare U-Net et DeepLabV3)
├── meilleur_unet.pth       # Poids U-Net (généré par le notebook)
└── meilleur_deeplabv3.pth  # Poids DeepLabV3 (généré par le notebook)
```

Les fichiers de poids ne sont pas versionnés dans le code source (environ 100 Mo chacun) : ils sont produits par le notebook. Si vous les publiez, utilisez [Git LFS](https://git-lfs.com/) ou les *Releases* GitHub.

## Installation

Python 3.10+ recommandé. Un GPU accélère l'entraînement ; l'inférence fonctionne aussi sur CPU.

```bash
git clone https://github.com/Ayoub-gOu/AgroSight.git
cd AgroSight

pip install torch torchvision segmentation-models-pytorch numpy pillow pandas streamlit
```

Avec [`uv`](https://docs.astral.sh/uv/) :

```bash
uv init inference-uav && cd inference-uav
uv add torch torchvision segmentation-models-pytorch numpy pillow pandas streamlit
```

Pour **ré-entraîner** les modèles dans le notebook, il faut aussi `fiftyone`, `matplotlib`, `tqdm` (le notebook les installe dans sa première cellule).

## Utilisation

### 1. Entraîner les modèles (notebook)

Ouvrez `Segmentation_Agricole_UAV_CropAndWeed_complet.ipynb` (Google Colab avec GPU T4 conseillé : environ 20 à 40 min par modèle) et exécutez les cellules dans l'ordre. Le notebook télécharge CropAndWeed, entraîne U-Net et DeepLabV3, évalue sur le test et écrit `meilleur_unet.pth` et `meilleur_deeplabv3.pth`.

Le paramètre `NOMBRE_IMAGES_MAX` permet de n'utiliser qu'un sous-ensemble aléatoire pour aller plus vite (`None` = les 8 034 images).

### 2. Inférence en ligne de commande

```bash
python inference.py parcelle.jpg --poids meilleur_unet.pth --architecture unet
python inference.py parcelle.jpg --poids meilleur_deeplabv3.pth --architecture deeplabv3 --sortie resultat.png
```

Sortie : une image de superposition (`<image>_superposition.png` par défaut), la proportion de pixels par classe et le taux d'infestation :

```
background          xx.xx %
maize               xx.xx %
autres_cultures     xx.xx %
mauvaises_herbes    xx.xx %
vegetation          xx.xx %
Taux d'infestation : xx.x %
```

### 3. Utilisation comme bibliothèque

```python
from inference import SegmentateurUAV, charger_image, superposer, calculer_stats

image = charger_image("parcelle.jpg")
modele = SegmentateurUAV("meilleur_unet.pth", architecture="unet")   # ou "deeplabv3"
masque = modele.predire(image)             # (H, W) uint8, résolution d'origine

stats = calculer_stats(masque)
print(stats["taux_infestation_pct"])
```

### 4. Interface web Streamlit

Placez les fichiers `.pth` à côté de `app.py`, puis :

```bash
streamlit run app.py
# ou : uv run streamlit run app.py
```

L'application détecte les modèles disponibles et, pour une image déposée :

- affiche l'image d'origine et la **superposition** de chaque modèle côte à côte ;
- propose le téléchargement des superpositions en PNG ;
- permet d'afficher les cartes de classes seules ;
- affiche un tableau des proportions de pixels par classe et le **taux d'infestation** de chaque modèle.

### Fonctionnement de l'inférence

Le prétraitement est **identique à l'entraînement** : redimensionnement à 256 × 256 (bilinéaire) puis normalisation ImageNet. Les scores (logits) sont ensuite agrandis à la résolution d'origine avant l'`argmax`, ce qui donne des contours plus lisses qu'un agrandissement du masque.

## Résultats

Les résultats détaillés (IoU par classe, matrices de confusion, courbes d'apprentissage, robustesse, comparaisons visuelles) sont produits par les sections 10 à 15 du notebook, calculés sur le jeu de test de **sessions jamais vues**.

## Limites et points d'attention

- **Échelle** : les modèles ont appris sur des images prises à environ 1,1 m du sol. À une autre altitude, la qualité peut chuter nettement.
- **Cohérence avec l'entraînement** : l'encodeur (`resnet34`), la liste des classes **et leur ordre** doivent être identiques à ceux du notebook, sinon les poids ne se chargent pas.
- **Classe `vegetation`** : fourre-tout très rare et ambigu, donc exclu du calcul du taux d'infestation.
- **Proportions relatives** : sans la résolution au sol, les proportions de pixels ne se convertissent pas en m² ou en hectares.
- **Résolution** : l'entrée est réduite à 256 × 256 ; les très petites plantes peuvent être manquées. Augmenter `TAILLE_IMAGE` (384, 512) dans le notebook peut aider si la mémoire GPU le permet.

## Licence et crédits

- Jeu de données : **CropAndWeed** (Steininger et al., WACV 2023), distribué par [Voxel51](https://huggingface.co/datasets/Voxel51/CropAndWeed), licence non commerciale.
- Modèles : [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- Chargement des données : [FiftyOne](https://docs.voxel51.com/).
