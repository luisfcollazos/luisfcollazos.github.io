"""Paso 2 — Construye el dataset de clasificación recortando cada grano etiquetado.

Usa las etiquetas reales (no las predicciones) para que el clasificador aprenda de datos limpios.
Genera la estructura que espera `yolo classify`:

  salida/
    train/<clase>/*.jpg
    val/<clase>/*.jpg
    test/<clase>/*.jpg   (si el data.yaml tiene test)

Además puede crear un data.yaml de UNA sola clase ("grano") que reutiliza las mismas imágenes,
para entrenar el detector de la etapa 1 sin duplicar el dataset.

Con --ventana 128 cada recorte es un cuadrado fijo de 128 px centrado en el grano, sin reescalar al
grano: un grano grande ocupa más del recorte que uno pequeño, así el clasificador puede usar el tamaño
(sanog/sanop, malla). Usa el mismo valor en 4_dos_etapas.py.

Con --normalizar-fondo cada foto se corrige con su fondo como referencia de blanco antes de recortar
(ver 0d_iluminacion.py). Úsalo también en 4_dos_etapas.py y preetiquetar_ls.py.

Fotos de distintas cámaras (celular): con --escala auto cada foto se reescala para que su grano mediano
mida lo mismo que el grano mediano del dataset, igual que hace 4_dos_etapas.py --escala auto al predecir.
Con --degradar N se guardan además N copias degradadas de cada recorte de train (desenfoque, menos
resolución, compresión JPEG), para que el clasificador tolere fotos de celular.

En la carpeta de salida se guarda preproceso.json (ventana, margen, normalización): 4_dos_etapas.py lo
compara con sus opciones y se detiene si no coinciden.

Con --fusionar se pueden unir clases que solo se distinguen por tamaño (ej. sanog/sanop): al
reescalar cada recorte a 224 px el clasificador pierde el tamaño real, así que es mejor clasificar
"sano" y decidir grande/pequeño midiendo el grano (ver 0_auditar_dataset.py y 4_dos_etapas.py).

Uso:
  python 2_recortar_granos.py --data data.yaml --out dataset_cls --balancear 600 --max-por-clase 3000 \
      --yaml-detector data_1clase.yaml \
      --fusionar sano=sanog,sanop
"""

import argparse
import io
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from comun import cargar_data_yaml, leer_etiquetas, listar_imagenes, normalizar_fondo, recortar, xywhn_a_xyxy


def degradar(img, rng):
    """Simula una foto de celular: menos resolución, desenfoque leve y compresión JPEG. No toca el color
    (el tono distingue Oscuro/Oreado/vinagre y se corrige aparte con --normalizar-fondo)."""
    from PIL import ImageFilter

    w, h = img.size
    f = rng.uniform(0.35, 0.8)
    out = img.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(35, 75))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml del dataset de detección")
    ap.add_argument("--root", default=None, help="sobrescribe el campo `path` del data.yaml")
    ap.add_argument("--out", default="dataset_cls")
    ap.add_argument("--margen", type=float, default=0.15, help="margen alrededor de la caja (fracción)")
    ap.add_argument("--ventana", type=int, default=0,
                    help="recorte cuadrado fijo de N px (conserva el tamaño real; recomendado ~128). 0 = ajustado a la caja")
    ap.add_argument("--min-px", type=int, default=8, help="descarta cajas más pequeñas que esto")
    ap.add_argument("--balancear", type=int, default=0,
                    help="en train, duplica recortes de clases con menos de N ejemplos hasta llegar a N")
    ap.add_argument("--yaml-detector", default=None, help="si se da, escribe un data.yaml de una sola clase")
    ap.add_argument("--max-por-clase", type=int, default=0,
                    help="en train, deja como máximo N recortes por clase (submuestrea la clase dominante)")
    ap.add_argument("--fusionar", action="append", default=[], metavar="NUEVA=A,B",
                    help="une clases en una sola carpeta; se puede repetir")
    ap.add_argument("--escala", default="0", metavar="auto|PX|0",
                    help="reescala cada foto para que su grano mediano mida como el del dataset (auto) o PX")
    ap.add_argument("--degradar", type=int, default=0,
                    help="en train, N copias degradadas (desenfoque/resolución/JPEG) de cada recorte")
    ap.add_argument("--normalizar-fondo", action="store_true",
                    help="corrige la iluminación de cada foto con el fondo como referencia (ver 0d_iluminacion.py)")
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

    lados = []  # lado mayor de cada caja de train: tamaño de referencia de un grano (para --escala auto)
    mediana_img = {}  # lado mayor mediano de los granos de cada imagen
    for split in ("train", "val", "test"):
        for img_path in listar_imagenes(data.get(split) or []):
            et = leer_etiquetas(img_path)
            if et:
                with Image.open(img_path) as im:
                    W, H = im.size
                ls = sorted(max(w * W, h * H) for _, _, _, w, h in et)
                mediana_img[img_path] = ls[len(ls) // 2]
    ref = None
    if args.escala == "auto":
        tr = sorted(mediana_img[p] for p in listar_imagenes(data.get("train") or []) if p in mediana_img)
        ref = tr[len(tr) // 2]
        print(f"Escala: cada foto se lleva a un grano mediano de {ref:.1f} px")
    elif args.escala not in ("0", ""):
        ref = float(args.escala)
    rng_deg = random.Random(args.seed + 1)
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
                im = normalizar_fondo(im) if args.normalizar_fondo else im.convert("RGB")
                if ref:
                    f = ref / mediana_img[img_path]
                    if abs(f - 1) >= 0.03:
                        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
                W, H = im.size
                for i, (cls, xc, yc, w, h) in enumerate(etiquetas):
                    if cls >= len(names):
                        print(f"  ! clase {cls} fuera de rango en {img_path}")
                        continue
                    x1, y1, x2, y2 = xywhn_a_xyxy(xc, yc, w, h, W, H)
                    if min(x2 - x1, y2 - y1) < args.min_px:
                        continue
                    if split == "train":
                        lados.append(max(x2 - x1, y2 - y1))
                    nombre = destino_de[names[cls]]
                    destino = out / split / nombre / f"{img_path.stem}_{i:03d}.jpg"
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    rec = recortar(im, (x1, y1, x2, y2), args.margen, ventana=args.ventana)
                    rec.save(destino, quality=95)
                    conteo[nombre] += 1
                    archivos_por_clase[nombre].append(destino)
                    if split == "train":
                        for k in range(args.degradar):
                            copia = destino.with_name(f"{destino.stem}_deg{k}.jpg")
                            degradar(rec, rng_deg).save(copia, quality=95)
                            archivos_por_clase[nombre].append(copia)

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

    with open(out / "preproceso.json", "w", encoding="utf-8") as f:
        json.dump({"ventana": args.ventana, "margen": args.margen, "normalizar_fondo": args.normalizar_fondo,
                   "lado_mediano_px": round(ref or (sorted(lados)[len(lados) // 2] if lados else 0), 1) or None,
                   "escala": ref is not None, "degradar": args.degradar}, f)

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
