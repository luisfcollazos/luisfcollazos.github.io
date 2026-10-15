"""Revisión completa de una o varias clases: TODOS sus granos, numerados, para unificar el criterio.

Para cada clase genera:
  * mosaicos <clase>_pagNN.jpg con todos los granos, a la MISMA escala (un grano grande se ve grande),
    ordenados de menor a mayor área: los que no encajan con el resto quedan juntos al principio o al final.
    Cada celda lleva su número; "N" en naranja = viene de una imagen nueva (que no estaba en --anterior).
  * <clase>.csv con número, imagen, split, tamaño medido y posición en % (como en Label Studio).
    Tiene las mismas columnas que errores.csv, así que marcar_errores.py puede dibujar los granos
    numerados sobre las imágenes completas para encontrarlos rápido.
  * un resumen del tamaño mediano en imágenes anteriores vs nuevas, para ver si cambió el criterio.

Uso:
  python revisar_clase.py --data YoloDataset6/data.yaml --clases Oreado,malla --anterior YoloDataset5
  python marcar_errores.py --errores revision_clases/Oreado.csv --data YoloDataset6/data.yaml --out revision_clases/Oreado_imagenes
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw

from comun import IMG_EXTS, cargar_data_yaml, leer_etiquetas, listar_imagenes, medir_grano, recortar, xywhn_a_xyxy

LADO = 128  # píxeles de cada celda del mosaico
COLS, FILAS = 10, 8
PREFIJO_LS = re.compile(r"^[0-9a-f]{8}-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clases", required=True, help="separadas por coma, ej. Oreado,malla")
    ap.add_argument("--anterior", default=None, help="dataset anterior: marca los granos de imágenes nuevas")
    ap.add_argument("--ventana", type=int, default=0,
                    help="lado en px de cada recorte (por defecto 2,5 x la mediana de las cajas de estas clases)")
    ap.add_argument("--out", default="revision_clases")
    args = ap.parse_args()

    data = cargar_data_yaml(args.data)
    names = data["names"]
    clases = [c.strip() for c in args.clases.split(",")]
    faltan = [c for c in clases if c not in names]
    if faltan:
        raise SystemExit(f"No están en el data.yaml: {faltan}. Clases: {names}")
    idx = {names.index(c): c for c in clases}

    anteriores = set()
    if args.anterior:
        anteriores = {PREFIJO_LS.sub("", p.stem) for p in (Path(args.anterior) / "images").rglob("*")
                      if p.suffix.lower() in IMG_EXTS}
        if not anteriores:
            raise SystemExit(f"No hay imágenes en {args.anterior}/images")

    granos = defaultdict(list)
    for split in ("train", "val", "test"):
        for img_path in listar_imagenes(data.get(split) or []):
            etiquetas = [e for e in leer_etiquetas(img_path) if e[0] in idx]
            if not etiquetas:
                continue
            nueva = bool(anteriores) and PREFIJO_LS.sub("", img_path.stem) not in anteriores
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for c, *caja_n in etiquetas:
                    xyxy = xywhn_a_xyxy(*caja_n, W, H)
                    m = medir_grano(recortar(im, xyxy, 0.15))
                    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    if m is None or m["pegado"]:  # granos pegados: se usa la caja
                        m = {"area": w * h * 0.785, "largo": max(w, h), "ancho": min(w, h)}
                    granos[idx[c]].append({"img": img_path, "split": split, "nueva": nueva, "xyxy": xyxy,
                                           "W": W, "H": H, **{k: m[k] for k in ("area", "largo", "ancho")}})

    lados = [max(g["xyxy"][2] - g["xyxy"][0], g["xyxy"][3] - g["xyxy"][1]) for gs in granos.values() for g in gs]
    ventana = args.ventana or int(2.5 * median(lados))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for clase in clases:
        gs = sorted(granos[clase], key=lambda g: g["area"])
        if not gs:
            print(f"{clase}: sin etiquetas")
            continue
        for n, g in enumerate(gs, 1):
            g["n"] = n
        por_pagina = COLS * FILAS
        for p in range(0, len(gs), por_pagina):
            pagina = gs[p:p + por_pagina]
            filas = (len(pagina) + COLS - 1) // COLS
            lienzo = Image.new("RGB", (COLS * LADO, filas * LADO + 24), (255, 255, 255))
            d = ImageDraw.Draw(lienzo)
            d.text((6, 6), f"{clase}: granos {pagina[0]['n']}-{pagina[-1]['n']} de {len(gs)}, de menor a mayor area "
                           f"(ventana {ventana}px; N = imagen nueva)", fill=(0, 0, 0))
            abiertas = {}
            for k, g in enumerate(pagina):
                if g["img"] not in abiertas:
                    abiertas = {g["img"]: Image.open(g["img"]).convert("RGB")}  # solo la última, por memoria
                rec = recortar(abiertas[g["img"]], g["xyxy"], 0, ventana=ventana)
                lado = rec.width  # la ventana, o más si el grano no cabe
                ox = (g["xyxy"][0] + g["xyxy"][2]) / 2 - lado / 2
                oy = (g["xyxy"][1] + g["xyxy"][3]) / 2 - lado / 2
                esc = LADO / lado
                rec = rec.resize((LADO, LADO))
                dr = ImageDraw.Draw(rec)
                dr.rectangle(((g["xyxy"][0] - ox) * esc, (g["xyxy"][1] - oy) * esc,
                              (g["xyxy"][2] - ox) * esc, (g["xyxy"][3] - oy) * esc), outline=(255, 0, 0), width=1)
                dr.rectangle((0, 0, 34, 13), fill=(0, 0, 0))
                dr.text((2, 1), str(g["n"]), fill=(255, 255, 255))
                if g["nueva"]:
                    dr.rectangle((35, 0, 49, 13), fill=(255, 140, 0))
                    dr.text((39, 1), "N", fill=(0, 0, 0))
                lienzo.paste(rec, ((k % COLS) * LADO, 24 + (k // COLS) * LADO))
            lienzo.save(out / f"{clase}_pag{p // por_pagina + 1:02d}.jpg", quality=90)

        with open(out / f"{clase}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["carpeta", "n", "imagen", "etiqueta", "modelo_dice", "confianza", "posicion_label_studio",
                        "split", "imagen_nueva", "area_px", "ancho_px", "largo_px"])
            for g in sorted(gs, key=lambda g: (g["img"].name, g["n"])):
                x1, y1, x2, y2 = g["xyxy"]
                pos = (f"x={x1 / g['W'] * 100:.1f}% y={y1 / g['H'] * 100:.1f}% "
                       f"w={(x2 - x1) / g['W'] * 100:.1f}% h={(y2 - y1) / g['H'] * 100:.1f}%")
                w.writerow([clase, g["n"], g["img"].name, clase, f"#{g['n']}", 0, pos, g["split"],
                            "si" if g["nueva"] else "no", round(g["area"]), round(g["ancho"]), round(g["largo"])])

        def resumen(sub):
            if not sub:
                return "—"
            return (f"{len(sub):>4} granos  área {median(g['area'] for g in sub):>6.0f}  "
                    f"ancho {median(g['ancho'] for g in sub):>4.0f}  largo {median(g['largo'] for g in sub):>4.0f}")
        print(f"\n{clase} ({len(gs)} granos, {(len(gs) + por_pagina - 1) // por_pagina} páginas) — medianas en px:")
        if anteriores:
            print(f"  imágenes anteriores: {resumen([g for g in gs if not g['nueva']])}")
            print(f"  imágenes nuevas:     {resumen([g for g in gs if g['nueva']])}")
        else:
            print(f"  todas: {resumen(gs)}")
    print(f"\nMosaicos y CSV en {out}/")


if __name__ == "__main__":
    main()
