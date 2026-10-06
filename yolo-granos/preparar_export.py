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


def desviacion(clases_de: dict, split_de: dict, fracs: dict) -> float:
    """Peor desviación relativa entre lo que recibe cada split (por clase y en imágenes) y su proporción."""
    total, n_img = Counter(), Counter(split_de.values())
    asignado = {s: Counter() for s in fracs}
    for st, cs in clases_de.items():
        total.update(cs)
        asignado[split_de[st]].update(cs)
    peor = max(abs(n_img[s] / len(clases_de) - f) / f for s, f in fracs.items() if f > 0)
    for s, f in fracs.items():
        if f > 0:
            peor = max([peor] + [abs(asignado[s][c] / total[c] - f) / f for c in total])
    return peor


def repartir(clases_de: dict, fracs: dict, seed: int = 0, intentos: int = 300) -> dict:
    """Prueba `intentos` órdenes aleatorios del reparto voraz y devuelve el más parejo."""
    mejor, mejor_d = None, float("inf")
    for k in range(intentos):
        split_de = _repartir_una_vez(clases_de, fracs, seed * 100003 + k)
        d = desviacion(clases_de, split_de, fracs)
        if d < mejor_d:
            mejor, mejor_d = split_de, d
    return mejor


def _repartir_una_vez(clases_de: dict, fracs: dict, seed: int) -> dict:
    """Reparto estratificado multi-etiqueta (cada imagen tiene muchas clases).

    Recorre las imágenes empezando por las que contienen las clases más raras y asigna cada una al split
    al que más le falta, sumando el déficit relativo de TODAS sus clases (las raras pesan más) y el
    déficit de número de imágenes, para que cada split reciba su proporción de clases y de imágenes.
    """
    rng = random.Random(seed)
    total = Counter()
    for cs in clases_de.values():
        total.update(cs)
    splits = [s for s in fracs if fracs[s] > 0]
    objetivo = {s: {c: total[c] * fracs[s] for c in total} for s in splits}
    obj_img = {s: len(clases_de) * fracs[s] for s in splits}
    asignado = {s: Counter() for s in splits}
    n_img = Counter()

    stems = list(clases_de)
    rng.shuffle(stems)
    stems.sort(key=lambda st: min((total[c] for c in clases_de[st]), default=10**9))

    def necesidad(s, cs):
        if n_img[s] >= obj_img[s] + 1:  # no pasarse del número de imágenes
            return -1e9
        peso_total = sum(n / total[c] for c, n in cs.items()) or 1.0
        clases = sum((n / total[c]) * (objetivo[s][c] - asignado[s][c]) / max(objetivo[s][c], 1e-9)
                     for c, n in cs.items()) / peso_total
        imagenes = (obj_img[s] - n_img[s]) / obj_img[s]
        return clases + imagenes

    split_de = {}
    for st in stems:
        cs = clases_de[st]
        s = max(splits, key=lambda s: necesidad(s, cs))
        split_de[st] = s
        asignado[s].update(cs)
        n_img[s] += 1
    return split_de


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
    fracs = {"train": 1 - args.val - args.test, "val": args.val, "test": args.test}
    split_de = repartir(clases_de, fracs, args.seed)
    print(f"\nPeor desviación de una clase respecto a su proporción: {desviacion(clases_de, split_de, fracs):.0%}")
    asignado = {s: Counter() for s in fracs}
    n_img = Counter(split_de.values())
    for stem, split in split_de.items():
        asignado[split].update(clases_de[stem])

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
