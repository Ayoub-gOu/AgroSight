import json

src = "Segmentation_Agricole_UAV_CropAndWeed_complet.ipynb"
dst = "notebook_leger.ipynb"

nb = json.load(open(src, encoding="utf-8"))
nb["metadata"].pop("widgets", None)          # état des widgets FiftyOne

for cellule in nb["cells"]:
    if cellule["cell_type"] != "code":
        continue
    sorties = []
    for sortie in cellule.get("outputs", []):
        if "data" in sortie:                  # on ne garde que le texte, pas les images
            texte = {k: v for k, v in sortie["data"].items() if k == "text/plain"}
            if not texte:
                continue
            sortie["data"] = texte
        sorties.append(sortie)
    cellule["outputs"] = sorties

json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
print("OK")