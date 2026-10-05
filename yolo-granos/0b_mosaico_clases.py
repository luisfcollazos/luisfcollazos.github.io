"""Paso 0b — Verificación visual: ¿cada índice de clase corresponde al nombre del data.yaml?

Para cada índice que aparece en los .txt de etiquetas genera un mosaico de recortes al azar, con la
caja etiquetada dibujada en rojo. Así se ve de un vistazo:
  * si el índice 3 realmente son granos "blanqueado" (o si los nombres están corridos/desordenados)
  * si la caja cubre el grano entero o solo el defecto (ej. solo el orificio de broca)
  * el tamaño relativo: todos los recortes usan la misma ventana en píxeles (2,5 veces el tamaño mediano
    de caja), así que un grano grande se ve grande y uno pequeño se ve pequeño (útil para sanog/sanop)

También imprime el histograma de índices crudos, incluidos los que no tienen nombre en el data.yaml.
Útil para comparar varias versiones del data.yaml (por ejemplo, la de entrenamiento y la exportada).

Uso:
  python 0b_mosaico_clases.py --data data.yaml --n 48
  python 0b_mosaico_clases.py --data data.yaml --modelo best.pt   # compara con los nombres del modelo
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from comun import cargar_data_yaml, leer_etiquetas, listar_imagenes, xywhn_a_xyxy

LADO = 128  # tamaño de cada celda del mosaico


def celda(img_path, caja_n, ventana):
    """Recorte de `ventana` px centrado en la caja (más grande si la caja no cabe), escalado a LADO x LADO."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        x1, y1, x2, y2 = xywhn_a_xyxy(*caja_n, W, H)
        lado = max(ventana, (x2 - x1) * 1.2, (y2 - y1) * 1.2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = cx - lado / 2, cy - lado / 2
        rec = im.crop((int(ox), int(oy), int(ox + lado), int(oy + lado)))
    esc = LADO / rec.width
    rec = rec.resize((LADO, LADO))
    ImageDraw.Draw(rec).rectangle(((x1 - ox) * esc, (y1 - oy) * esc, (x2 - ox) * esc, (y2 - oy) * esc),
                                  outline=(255, 0, 0), width=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--modelo", default=None, help="best.pt: muestra también los nombres guardados en el modelo")
    ap.add_argument("--n", type=int, default=48, help="recortes por clase")
    ap.add_argument("--out", default="auditoria/mosaicos")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    nombres_modelo = None
    if args.modelo:
        from ultralytics import YOLO

        nombres_modelo = YOLO(args.modelo).names

    por_indice = defaultdict(list)
    conteo = Counter()
    for split in ("train", "val", "test"):
        for img_path in listar_imagenes(data.get(split) or []):
            for c, *caja in leer_etiquetas(img_path):
                conteo[c] += 1
                por_indice[c].append((img_path, caja))

    print(f"{'índice':>6}  {'nombre en data.yaml':<22}" + (f"{'nombre en el modelo':<22}" if nombres_modelo else "")
          + f"{'instancias':>11}")
    for c in sorted(set(conteo) | set(range(len(names)))):
        n_yaml = names[c] if c < len(names) else "!! SIN NOMBRE"
        n_mod = f"{nombres_modelo.get(c, '!! no existe'):<22}" if nombres_modelo else ""
        aviso = ""
        if nombres_modelo and nombres_modelo.get(c) != n_yaml:
            aviso = "  <- difiere"
        if conteo[c] == 0:
            aviso += "  <- sin etiquetas"
        print(f"{c:>6}  {n_yaml:<22}{n_mod}{conteo[c]:>11}{aviso}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # ventana común: 2,5 x la mediana del lado mayor de caja, medida en una muestra de imágenes
    lados = []
    for c, muestras in por_indice.items():
        for img_path, caja in rng.sample(muestras, min(20, len(muestras))):
            with Image.open(img_path) as im:
                lados.append(max(caja[2] * im.width, caja[3] * im.height))
    ventana = 2.5 * sorted(lados)[len(lados) // 2]
    for c, muestras in sorted(por_indice.items()):
        sel = rng.sample(muestras, min(args.n, len(muestras)))
        cols = 8
        filas = (len(sel) + cols - 1) // cols
        lienzo = Image.new("RGB", (cols * LADO, filas * LADO + 24), (255, 255, 255))
        nombre = names[c] if c < len(names) else "SIN_NOMBRE"
        ImageDraw.Draw(lienzo).text((6, 6), f"indice {c} = '{nombre}'  ({conteo[c]} etiquetas)", fill=(0, 0, 0))
        for k, (img_path, caja) in enumerate(sel):
            lienzo.paste(celda(img_path, caja, ventana), ((k % cols) * LADO, 24 + (k // cols) * LADO))
        lienzo.save(out / f"{c:02d}_{nombre}.jpg", quality=90)
    print(f"\nMosaicos en {out}/ — revisa que cada uno contenga solo granos de la clase de su nombre.")


if __name__ == "__main__":
    main()
