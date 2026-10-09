"""Dibuja sobre cada imagen completa los granos a revisar, numerados, para encontrarlos rápido en Label Studio.

Lee el errores.csv que genera `4_dos_etapas.py evaluar --guardar-errores` y, para cada imagen, guarda una
copia con un recuadro y un número por grano. Los números siguen el mismo orden que la hoja de revisión
(de arriba hacia abajo y de izquierda a derecha), así que coinciden con la columna "Grano en la imagen".

Junto a cada recuadro va la etiqueta actual y lo que dice el modelo, por ejemplo "3 sanog>vinagre".
Se omiten las confusiones sanog<->sanop porque desaparecen al fusionar esas clases (--incluir-sanos para no omitirlas).

Uso:
  python marcar_errores.py --errores errores_train/errores.csv --data dataset_v2/data.yaml --out marcadas_train
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from comun import cargar_data_yaml, listar_imagenes


def leer_errores(ruta, incluir_sanos=False):
    """Lee el CSV (tolera los ';' que añade Excel al guardar) y devuelve {imagen: [granos ordenados]}."""
    texto = Path(ruta).read_text(encoding="utf-8-sig")
    lineas = [l.rstrip().rstrip(";") for l in texto.splitlines() if l.strip()]
    por_img = defaultdict(list)
    for f in csv.DictReader(lineas):
        if not incluir_sanos and {f["etiqueta"], f["modelo_dice"]} == {"sanog", "sanop"}:
            continue
        pos = {k: float(v) for k, v in re.findall(r"([xywh])=([\d.]+)%", f["posicion_label_studio"])}
        por_img[f["imagen"]].append({**pos, "etiqueta": f["etiqueta"], "modelo": f["modelo_dice"],
                                     "conf": float(f["confianza"])})
    for granos in por_img.values():
        granos.sort(key=lambda g: (round(g["y"] / 5), g["x"]))  # mismo orden que la hoja de revisión
    return por_img


def zona(x, y, w, h):
    cx, cy = x + w / 2, y + h / 2
    v = "arriba" if cy < 33.3 else "centro" if cy < 66.6 else "abajo"
    hz = "izquierda" if cx < 33.3 else "centro" if cx < 66.6 else "derecha"
    return v if v == hz == "centro" else f"{v} · {hz}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errores", required=True, help="errores.csv de 4_dos_etapas.py --guardar-errores")
    ap.add_argument("--data", required=True, help="data.yaml del dataset (para encontrar las imágenes)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="marcadas")
    ap.add_argument("--incluir-sanos", action="store_true", help="no omitir las confusiones sanog<->sanop")
    args = ap.parse_args()

    por_img = leer_errores(args.errores, args.incluir_sanos)
    data = cargar_data_yaml(args.data, args.root)
    rutas = {p.name: p for s in ("train", "val", "test") for p in listar_imagenes(data.get(s) or [])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    faltan = 0
    for nombre, granos in sorted(por_img.items()):
        if nombre not in rutas:
            faltan += 1
            continue
        with Image.open(rutas[nombre]) as im:
            im = im.convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        grosor = max(3, W // 400)
        try:
            fuente = ImageFont.load_default(size=max(16, W // 60))
        except TypeError:  # Pillow antiguo sin tamaño de fuente
            fuente = ImageFont.load_default()
        for k, g in enumerate(granos, 1):
            x1, y1 = g["x"] / 100 * W, g["y"] / 100 * H
            x2, y2 = x1 + g["w"] / 100 * W, y1 + g["h"] / 100 * H
            d.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=grosor)
            texto = f"{k} {g['etiqueta']}>{g['modelo']}"
            tx, ty = x1, max(0, y1 - fuente.size - 6) if hasattr(fuente, "size") else max(0, y1 - 20)
            caja = d.textbbox((tx, ty), texto, font=fuente)
            if caja[2] > W - 3:  # que el texto no se salga por la derecha
                tx -= caja[2] - (W - 3)
                caja = d.textbbox((tx, ty), texto, font=fuente)
            d.rectangle((caja[0] - 3, caja[1] - 3, caja[2] + 3, caja[3] + 3), fill=(0, 0, 0))
            d.text((tx, ty), texto, fill=(255, 255, 0), font=fuente)
        im.save(out / f"{Path(nombre).stem}__{len(granos)}granos.jpg", quality=90)

    with open(out / "indice.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["imagen", "grano", "zona", "etiqueta", "modelo_dice", "confianza", "x %", "y %"])
        for nombre, granos in sorted(por_img.items()):
            for k, g in enumerate(granos, 1):
                w.writerow([nombre, k, zona(g["x"], g["y"], g["w"], g["h"]), g["etiqueta"], g["modelo"],
                            f"{g['conf']:.3f}".replace(".", ","), str(g["x"]).replace(".", ","),
                            str(g["y"]).replace(".", ",")])
    print(f"{len(por_img) - faltan} imágenes marcadas en {out}/ ({sum(map(len, por_img.values()))} granos)")
    if faltan:
        print(f"!! {faltan} imágenes del CSV no están en el dataset de --data (¿otro dataset o split?)")


if __name__ == "__main__":
    main()
