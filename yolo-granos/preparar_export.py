"""Convierte un export YOLO de Label Studio en un dataset listo para entrenar.

Label Studio exporta en una sola carpeta:
    export/images/*.jpg   export/labels/*.txt   export/classes.txt   (notes.json)
sin separar train/val/test. Este script:

  1. Toma los nombres de clase de classes.txt (el MISMO orden que usan los índices de los .txt),
     para que el data.yaml no se pueda desalinear de las etiquetas.
  2. Valida: índices fuera de rango, imágenes sin .txt, .txt sin imagen, clases sin etiquetas.
  3. Reparte las imágenes en train/val/test de forma ESTRATIFICADA: asigna primero las imágenes con
     las clases más raras, para que cada clase tenga su proporción en val y test.
  4. Copia todo a una carpeta NUEVA (nunca sobre un dataset anterior: así no sobreviven .txt viejos
     ni caches) y escribe data.yaml.

Uso:
  python preparar_export.py --export ruta/export_labelstudio --out dataset_v2 --val 0.15 --test 0.15
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from comun import IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="carpeta del export YOLO de Label Studio")
    ap.add_argument("--out", required=True, help="carpeta nueva del dataset (no debe existir)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp, out = Path(args.export), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} ya existe: usa una carpeta nueva para no mezclar con un dataset anterior.")
    clases_txt = exp / "classes.txt"
    if not clases_txt.is_file():
        raise SystemExit(f"No encuentro {clases_txt}. ¿Exportaste en formato YOLO?")
    names = [n.strip() for n in clases_txt.read_text(encoding="utf-8").splitlines() if n.strip()]
    print("Clases según classes.txt:")
    for i, n in enumerate(names):
        print(f"  {i:>2}  {n}")

    imagenes = {p.stem: p for p in (exp / "images").rglob("*") if p.suffix.lower() in IMG_EXTS}
    etiquetas = {p.stem: p for p in (exp / "labels").rglob("*.txt")}
    sin_txt = sorted(set(imagenes) - set(etiquetas))
    sin_img = sorted(set(etiquetas) - set(imagenes))
    if sin_txt:
        print(f"\n!! {len(sin_txt)} imágenes sin .txt (quedarán como imágenes sin granos): {sin_txt[:5]}")
    if sin_img:
        print(f"!! {len(sin_img)} .txt sin imagen (se ignoran): {sin_img[:5]}")

    clases_de = {}
    total = Counter()
    errores = 0
    for stem in imagenes:
        cs = []
        if stem in etiquetas:
            for n_linea, linea in enumerate(etiquetas[stem].read_text(encoding="utf-8").splitlines(), 1):
                v = linea.split()
                if not v:
                    continue
                c = int(float(v[0]))
                if not 0 <= c < len(names) or len(v) < 5:
                    print(f"!! {etiquetas[stem].name}:{n_linea} línea inválida: {linea!r}")
                    errores += 1
                    continue
                cs.append(c)
        clases_de[stem] = Counter(cs)
        total.update(cs)
    if errores:
        raise SystemExit(f"\n{errores} líneas inválidas: corrige el export antes de continuar.")
    vacias = [names[i] for i in range(len(names)) if total[i] == 0]
    if vacias:
        print(f"\n!! Clases SIN etiquetas: {vacias}")

    # --- reparto estratificado ---
    rng = random.Random(args.seed)
    fracs = {"train": 1 - args.val - args.test, "val": args.val, "test": args.test}
    objetivo = {s: {c: total[c] * f for c in total} for s, f in fracs.items()}
    asignado = {s: Counter() for s in fracs}
    n_img = {s: 0 for s in fracs}
    stems = list(imagenes)
    rng.shuffle(stems)
    # primero las imágenes cuya clase más rara es más rara
    stems.sort(key=lambda s: min((total[c] for c in clases_de[s]), default=10**9))
    split_de = {}
    for stem in stems:
        cs = clases_de[stem]
        if cs:
            rara = min(cs, key=lambda c: total[c])
            # el split al que más le falta (en proporción) de la clase rara de esta imagen
            split = max(fracs, key=lambda s: (objetivo[s][rara] - asignado[s][rara]) / max(objetivo[s][rara], 1e-9)
                        if fracs[s] > 0 else -1e9)
        else:
            split = max(fracs, key=lambda s: fracs[s] * len(stems) - n_img[s])
        split_de[stem] = split
        asignado[split].update(cs)
        n_img[split] += 1

    for stem, split in split_de.items():
        for sub, src in (("images", imagenes[stem]), ("labels", etiquetas.get(stem))):
            if src is None:
                continue
            dst = out / sub / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    data = {"path": str(out.resolve()), "names": dict(enumerate(names))}
    for s in fracs:
        if n_img[s]:
            data[s] = f"images/{s}"
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    splits = [s for s in fracs if n_img[s]]
    print(f"\nImágenes: " + ", ".join(f"{s}={n_img[s]}" for s in splits))
    print(f"\n  {'clase':<14}" + "".join(f"{s:>10}" for s in splits))
    for i, n in enumerate(names):
        print(f"  {n:<14}" + "".join(f"{asignado[s][i]:>10}" for s in splits))
    print(f"\nDataset en {out}/  ->  data.yaml listo. Siguiente: python 0_auditar_dataset.py --data {out}/data.yaml")


if __name__ == "__main__":
    main()
