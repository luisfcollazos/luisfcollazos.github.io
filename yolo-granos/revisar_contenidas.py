"""Ayuda a corregir en Label Studio las cajas que contienen a otra ("caja_dentro_de_otra" en la auditoría)
y los duplicados (dos cajas casi iguales sobre el mismo grano, "duplicado_*").

Para cada par genera una imagen ampliada con:
  * en AZUL la caja grande (exterior)
  * en ROJO la caja contenida (interior)
y una tabla con la imagen, las clases y la posición de cada caja en PORCENTAJE (las mismas unidades que
muestra Label Studio en el panel de la región), para encontrarlas rápido.

Casos típicos y qué hacer:
  1. La caja azul abarca DOS granos pegados y uno de ellos tiene además su propia caja roja
     -> borrar la azul y dibujar una caja por cada grano.
  2. La caja roja marca solo el DEFECTO (orificio de broca, mancha) dentro del grano de la azul
     -> borrar la roja; si el defecto es la clase correcta, cambiar la clase de la azul.
  3. Un fragmento pequeño (partido) que realmente está ENCIMA de otro grano
     -> está bien; no hay que cambiar nada.
  4. DUPLICADO: dos cajas casi iguales sobre el mismo grano (suele pasar al corregir: se dibuja una caja
     nueva sin borrar la anterior) -> borrar la que tiene la clase equivocada.

Uso:
  python revisar_contenidas.py --data dataset_v2/data.yaml --out auditoria/contenidas
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from comun import cargar_data_yaml, iou, leer_etiquetas, listar_imagenes, xywhn_a_xyxy


def contencion(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / menor if menor > 0 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def pct(b, W, H):
    """Posición estilo Label Studio: x, y de la esquina superior izquierda, ancho y alto, en %."""
    return f"x={100 * b[0] / W:.1f}% y={100 * b[1] / H:.1f}% w={100 * (b[2] - b[0]) / W:.1f}% h={100 * (b[3] - b[1]) / H:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="auditoria/contenidas")
    ap.add_argument("--iou-dup", type=float, default=0.6, help="mismo valor que en 0_auditar_dataset.py")
    ap.add_argument("--contencion", type=float, default=0.9, help="mismo valor que en 0_auditar_dataset.py")
    args = ap.parse_args()

    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filas = []
    for split in ("train", "val", "test"):
        for img_path in listar_imagenes(data.get(split) or []):
            etiquetas = leer_etiquetas(img_path)
            if len(etiquetas) < 2:
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
            W, H = im.size
            cajas = [(c, xywhn_a_xyxy(xc, yc, w, h, W, H)) for c, xc, yc, w, h in etiquetas]
            for i in range(len(cajas)):
                for j in range(i + 1, len(cajas)):
                    (ci, bi), (cj, bj) = cajas[i], cajas[j]
                    # mismo criterio que la auditoría
                    if iou(bi, bj) >= args.iou_dup:
                        tipo = "duplicado"
                    elif contencion(bi, bj) >= args.contencion:
                        tipo = "contenida"
                    else:
                        continue
                    (c_ext, b_ext), (c_int, b_int) = sorted([(ci, bi), (cj, bj)], key=lambda x: -area(x[1]))
                    n = len(filas) + 1
                    # recorte ampliado alrededor de la caja exterior
                    m = max(b_ext[2] - b_ext[0], b_ext[3] - b_ext[1])
                    x0, y0 = max(0, b_ext[0] - m), max(0, b_ext[1] - m)
                    x1, y1 = min(W, b_ext[2] + m), min(H, b_ext[3] + m)
                    rec = im.crop((int(x0), int(y0), int(x1), int(y1)))
                    esc = 400 / max(rec.size)
                    rec = rec.resize((int(rec.width * esc), int(rec.height * esc)))
                    d = ImageDraw.Draw(rec)
                    for b, color in ((b_ext, (0, 90, 255)), (b_int, (255, 0, 0))):
                        d.rectangle([(b[0] - x0) * esc, (b[1] - y0) * esc, (b[2] - x0) * esc, (b[3] - y0) * esc],
                                    outline=color, width=3)
                    d.text((4, 4), f"#{n} {tipo}: azul={names[c_ext]}  rojo={names[c_int]}", fill=(0, 0, 0))
                    rec.save(out / f"{n:02d}_{img_path.stem}.jpg", quality=92)
                    filas.append([n, tipo, split, img_path.name, names[c_ext], pct(b_ext, W, H),
                                  names[c_int], pct(b_int, W, H), round(contencion(bi, bj), 2)])

    with open(out / "contenidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "tipo", "split", "imagen", "clase_exterior(azul)", "pos_exterior",
                    "clase_interior(rojo)", "pos_interior", "contencion"])
        w.writerows(filas)
    for n, tipo, split, img, ce, pe, ci, pi, k in filas:
        print(f"#{n:<3} {tipo:<10} {img}\n     azul {ce:<11} {pe}\n     rojo {ci:<11} {pi}")
    print(f"\n{len(filas)} pares. Imágenes ampliadas y contenidas.csv en {out}/")


if __name__ == "__main__":
    main()
