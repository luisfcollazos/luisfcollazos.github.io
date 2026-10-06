"""Paso 2 — Construye el dataset de clasificación recortando cada grano etiquetado.

Usa las etiquetas reales (no las predicciones) para que el clasificador aprenda de datos limpios.
Genera la estructura que espera `yolo classify`:

  salida/
    train/<clase>/*.jpg
    val/<clase>/*.jpg
    test/<clase>/*.jpg   (si el data.yaml tiene test)

Además puede crear un data.yaml de UNA sola clase ("grano") que reutiliza las mismas imágenes,
para entrenar el detector de la etapa 1 sin duplicar el dataset.

Con --fusionar se pueden unir clases que solo se distinguen por tamaño (ej. sanog/sanop): al
reescalar cada recorte a 224 px el clasificador pierde el tamaño real, así que es mejor clasificar
"sano" y decidir grande/pequeño midiendo el grano (ver 0_auditar_dataset.py y 4_dos_etapas.py).

Uso:
  python 2_recortar_granos.py --data data.yaml --out dataset_cls --balancear 600 --max-por-clase 3000 \
      --yaml-detector data_1clase.yaml \
      --fusionar sano=sanog,sanop
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from comun import cargar_data_yaml, leer_etiquetas, listar_imagenes, recortar, xywhn_a_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml del dataset de detección")
    ap.add_argument("--root", default=None, help="sobrescribe el campo `path` del data.yaml")
    ap.add_argument("--out", default="dataset_cls")
    ap.add_argument("--margen", type=float, default=0.15, help="margen alrededor de la caja (fracción)")
    ap.add_argument("--min-px", type=int, default=8, help="descarta cajas más pequeñas que esto")
    ap.add_argument("--balancear", type=int, default=0,
                    help="en train, duplica recortes de clases con menos de N ejemplos hasta llegar a N")
    ap.add_argument("--yaml-detector", default=None, help="si se da, escribe un data.yaml de una sola clase")
    ap.add_argument("--max-por-clase", type=int, default=0,
                    help="en train, deja como máximo N recortes por clase (submuestrea la clase dominante)")
    ap.add_argument("--fusionar", action="append", default=[], metavar="NUEVA=A,B",
                    help="une clases en una sola carpeta; se puede repetir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    destino_de = {n: n for n in names}
    for regla in args.fusionar:
        nueva, _, origen = regla.partition("=")
        for n in origen.split(","):
            if n not in destino_de:
                raise SystemExit(f"--fusionar: la clase '{n}' no existe en el data.yaml")
            destino_de[n] = nueva
    clases_salida = list(dict.fromkeys(destino_de.values()))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        conteo = Counter()
        sin_etiqueta = 0
        archivos_por_clase = defaultdict(list)
        imagenes = listar_imagenes(data[split])
        for img_path in imagenes:
            etiquetas = leer_etiquetas(img_path)
            if not etiquetas:
                sin_etiqueta += 1
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for i, (cls, xc, yc, w, h) in enumerate(etiquetas):
                    if cls >= len(names):
                        print(f"  ! clase {cls} fuera de rango en {img_path}")
                        continue
                    x1, y1, x2, y2 = xywhn_a_xyxy(xc, yc, w, h, W, H)
                    if min(x2 - x1, y2 - y1) < args.min_px:
                        continue
                    nombre = destino_de[names[cls]]
                    destino = out / split / nombre / f"{img_path.stem}_{i:03d}.jpg"
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    recortar(im, (x1, y1, x2, y2), args.margen).save(destino, quality=95)
                    conteo[nombre] += 1
                    archivos_por_clase[nombre].append(destino)

        # Ultralytics classify necesita la carpeta de cada clase en todos los splits
        for n in clases_salida:
            (out / split / n).mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)
        if split == "train" and args.max_por_clase:
            for n, archivos in archivos_por_clase.items():
                if len(archivos) > args.max_por_clase:
                    rng.shuffle(archivos)
                    for f in archivos[args.max_por_clase:]:
                        f.unlink()
                    del archivos[args.max_por_clase:]
        if split == "train" and args.balancear:
            for n, archivos in archivos_por_clase.items():
                faltan = args.balancear - len(archivos)
                for k in range(max(0, faltan)):
                    src = rng.choice(archivos)
                    shutil.copy(src, src.with_name(f"{src.stem}_dup{k:04d}.jpg"))

        print(f"\n[{split}] {len(imagenes)} imágenes, {sin_etiqueta} sin etiquetas, {sum(conteo.values())} recortes")
        for n in clases_salida:
            final = len(list((out / split / n).glob("*.jpg")))
            print(f"  {n:<14} {conteo[n]:>6}" + (f"  -> {final} en disco" if final != conteo[n] else ""))

    if args.yaml_detector:
        det = {"names": {0: "grano"}}
        for split in ("train", "val", "test"):
            if data.get(split):
                det[split] = data[split] if len(data[split]) > 1 else data[split][0]
        with open(args.yaml_detector, "w", encoding="utf-8") as f:
            yaml.safe_dump(det, f, allow_unicode=True, sort_keys=False)
        print(f"\ndata.yaml de una clase escrito en {args.yaml_detector} (entrénalo con single_cls=True)")


if __name__ == "__main__":
    main()
