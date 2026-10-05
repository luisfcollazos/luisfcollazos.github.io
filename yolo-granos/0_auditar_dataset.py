"""Paso 0 — Auditoría del dataset de detección.

Revisa tres problemas que afectan directamente la matriz de confusión:

1. Cajas solapadas (granos pegados o etiquetados dos veces)
   * IoU >= --iou-dup                        -> probable DUPLICADO del mismo grano
     (si las clases difieren, el modelo recibe dos respuestas para el mismo grano)
   * una caja contiene casi entera a otra    -> probable caja que ABARCA VARIOS GRANOS
   * IoU entre 0 y --iou-dup                 -> granos pegados: sirve para elegir el `iou` del NMS
     (si dos granos pegados tienen IoU mayor que el umbral del NMS, uno de los dos se pierde)

2. Tamaño por clase (en píxeles), medido sobre el grano real (no la caja) para que no dependa de la
   orientación. Para el par grande/pequeño (sanog/sanop) busca el mejor umbral y dice qué tan
   separables son: si ni el mejor umbral acierta >90 %, las etiquetas de tamaño no son consistentes.

3. Clases raras: instancias e imágenes por clase y split, con aviso si hay muy pocas en val/test.

4. Cajas anormalmente pequeñas (< --frac-pequena x la mediana de todas): suele indicar que en algunas
   imágenes se etiquetó solo el defecto (ej. el orificio de broca) y en otras el grano entero.

Salidas en --out: problemas_solapamiento.csv, tamanos.csv y, con --dibujar N, imágenes de muestra
con las cajas problemáticas marcadas.

Uso:
  python 0_auditar_dataset.py --data data.yaml --grande sanog --pequeno sanop --dibujar 30
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from comun import cargar_data_yaml, iou, leer_etiquetas, listar_imagenes, medir_grano, recortar, xywhn_a_xyxy


def contencion(a, b) -> float:
    """Fracción de la caja más pequeña que queda dentro de la otra."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / menor if menor > 0 else 0.0


def mejor_umbral(grandes: np.ndarray, pequenos: np.ndarray):
    """Umbral que maximiza la exactitud balanceada al separar grande (>= u) de pequeño (< u)."""
    valores = np.unique(np.concatenate([grandes, pequenos]))
    mejor = (0.0, float("nan"))
    for u in valores:
        exactitud = ((grandes >= u).mean() + (pequenos < u).mean()) / 2
        if exactitud > mejor[0]:
            mejor = (float(exactitud), float(u))
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="auditoria")
    ap.add_argument("--iou-dup", type=float, default=0.6)
    ap.add_argument("--contencion", type=float, default=0.9)
    ap.add_argument("--grande", default="sanog")
    ap.add_argument("--pequeno", default="sanop")
    ap.add_argument("--frac-pequena", type=float, default=0.5,
                    help="caja pequeña si su lado medio es menor que esta fracción de la mediana global")
    ap.add_argument("--min-val", type=int, default=30, help="aviso si una clase tiene menos instancias en val/test")
    ap.add_argument("--no-medir", action="store_true", help="omite la medición del grano (más rápido)")
    ap.add_argument("--dibujar", type=int, default=0, help="guarda N imágenes con solapamientos marcados")
    args = ap.parse_args()

    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    instancias = defaultdict(Counter)
    imagenes_con = defaultdict(Counter)
    resoluciones = Counter()
    problemas, tamanos, iou_vecinos, diametros = [], [], [], []
    dibujadas = 0

    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        for img_path in listar_imagenes(data[split]):
            etiquetas = leer_etiquetas(img_path)
            with Image.open(img_path) as im:
                im = im.convert("RGB")
            W, H = im.size
            resoluciones[(W, H)] += 1
            cajas = [(c, xywhn_a_xyxy(xc, yc, w, h, W, H)) for c, xc, yc, w, h in etiquetas]
            for c, b in cajas:
                instancias[split][names[c]] += 1
                diametros.append((split, str(img_path), names[c], ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5))
            for n in {names[c] for c, _ in cajas}:
                imagenes_con[split][n] += 1

            marcadas = []
            for i in range(len(cajas)):
                for j in range(i + 1, len(cajas)):
                    (ci, bi), (cj, bj) = cajas[i], cajas[j]
                    v, k = iou(bi, bj), contencion(bi, bj)
                    if v >= args.iou_dup:
                        tipo = "duplicado_misma_clase" if ci == cj else "duplicado_distinta_clase"
                    elif k >= args.contencion:
                        tipo = "caja_dentro_de_otra"
                    elif v > 0:
                        iou_vecinos.append(v)
                        continue
                    else:
                        continue
                    problemas.append([split, str(img_path), tipo, names[ci], names[cj], round(v, 3), round(k, 3)])
                    marcadas += [i, j]

            if args.dibujar and marcadas and dibujadas < args.dibujar:
                vis = im.copy()
                d = ImageDraw.Draw(vis)
                for i, (c, b) in enumerate(cajas):
                    d.rectangle(b, outline=(255, 0, 0) if i in marcadas else (0, 200, 0), width=2)
                    d.text((b[0] + 2, b[1] + 2), names[c], fill=(255, 255, 0))
                (out / "muestras").mkdir(exist_ok=True)
                vis.save(out / "muestras" / f"{split}_{img_path.stem}.jpg")
                dibujadas += 1

            if not args.no_medir:
                for c, b in cajas:
                    caja_d = (b[2] - b[0]) ** 0.5 * (b[3] - b[1]) ** 0.5
                    m = medir_grano(recortar(im, b, 0.15)) or {}
                    tamanos.append([split, img_path.name, names[c], round(caja_d, 1),
                                    m.get("area"), m.get("largo"), m.get("ancho"), m.get("pegado")])

    # ---------- 4 (se calcula antes para incluirlo en el CSV). Cajas pequeñas ----------
    mediana = float(np.median([d for *_, d in diametros])) if diametros else 0.0
    pequenas = [d for d in diametros if d[3] < args.frac_pequena * mediana]
    for split, ruta, clase, d in pequenas:
        problemas.append([split, ruta, "caja_pequena", clase, "", "", round(d / mediana, 2)])

    # ---------- 1. Solapamientos ----------
    with open(out / "problemas_solapamiento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "imagen", "tipo", "clase_a", "clase_b", "iou", "contencion"])
        w.writerows(problemas)
    print("\n=== 1. Cajas solapadas ===")
    tipos = Counter(p[2] for p in problemas)
    for t in ("duplicado_misma_clase", "duplicado_distinta_clase", "caja_dentro_de_otra"):
        print(f"  {t:<26} {tipos[t]:>6}")
    pares = Counter(tuple(sorted((p[3], p[4]))) for p in problemas if p[2] == "duplicado_distinta_clase")
    if pares:
        print("  Pares de clases más frecuentes en duplicados de distinta clase:")
        for (a, b), n in pares.most_common(8):
            print(f"    {a} / {b}: {n}")
    if iou_vecinos:
        q = np.percentile(iou_vecinos, [50, 90, 99])
        print(f"  Cajas que se tocan (IoU 0–{args.iou_dup}): {len(iou_vecinos)} pares; "
              f"IoU p50={q[0]:.2f} p90={q[1]:.2f} p99={q[2]:.2f}")
        if q[2] < 0.5:
            print("  -> OK: con el iou de NMS por defecto (0.5–0.7) no se eliminan granos pegados reales")
        else:
            print(f"  -> usa un NMS con iou > {q[2]:.2f}; si no, se pierde uno de cada par de granos pegados")

    print(f"\n=== Cajas pequeñas (lado medio < {args.frac_pequena:.0%} de la mediana = "
          f"{args.frac_pequena * mediana:.0f} px) ===")
    if pequenas:
        total = Counter(c for _, _, c, _ in diametros)
        for clase, n in Counter(c for _, _, c, _ in pequenas).most_common():
            print(f"  {clase:<14} {n:>6}  ({n / total[clase]:.0%} de la clase)")
        print("  -> revisa si en esas se etiquetó solo el defecto y no el grano entero (ver 0b_mosaico_clases.py)")
    else:
        print("  ninguna")

    # ---------- 2. Tamaños ----------
    if len(resoluciones) > 1:
        print(f"\n!! Hay {len(resoluciones)} resoluciones distintas: {resoluciones.most_common(5)}")
        print("   Medir en píxeles solo es válido si la cámara está a la misma distancia en todas las imágenes.")
    if tamanos:
        with open(out / "tamanos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "imagen", "clase", "caja_diam_px", "area_px", "largo_px", "ancho_px", "pegado"])
            w.writerows(tamanos)
        print("\n=== 2. Tamaño por clase (px; mediana [p10–p90]) — solo granos no pegados ===")
        print(f"  {'clase':<14}{'n':>6}{'ancho':>22}{'largo':>22}{'área':>26}")
        por_clase = defaultdict(lambda: defaultdict(list))
        for _, _, c, _, area, largo, ancho, pegado in tamanos:
            if area is not None and not pegado:
                por_clase[c]["area"].append(area)
                por_clase[c]["largo"].append(largo)
                por_clase[c]["ancho"].append(ancho)

        def fmt(v):
            p = np.percentile(v, [50, 10, 90])
            return f"{p[0]:.0f} [{p[1]:.0f}–{p[2]:.0f}]"

        for n in names:
            v = por_clase.get(n)
            if v and v["area"]:
                print(f"  {n:<14}{len(v['area']):>6}{fmt(v['ancho']):>22}{fmt(v['largo']):>22}{fmt(v['area']):>26}")

        g, p = por_clase.get(args.grande), por_clase.get(args.pequeno)
        if g and p and g["area"] and p["area"]:
            print(f"\n  Separación {args.grande} (grande) vs {args.pequeno} (pequeño) con un solo umbral:")
            for metrica in ("ancho", "largo", "area"):
                exactitud, u = mejor_umbral(np.array(g[metrica]), np.array(p[metrica]))
                print(f"    {metrica:<6} umbral={u:8.1f}  exactitud balanceada={exactitud:.1%}")
            print("    > 95 %: el tamaño se puede MEDIR en vez de aprender (fusionar clases + umbral).")
            print("    < 90 %: las etiquetas de tamaño no son consistentes; conviene re-etiquetar con una regla fija.")

    # ---------- 3. Clases raras ----------
    print("\n=== 3. Instancias (imágenes) por clase y split ===")
    splits = [s for s in ("train", "val", "test") if instancias[s]]
    print(f"  {'clase':<14}" + "".join(f"{s:>18}" for s in splits))
    for n in names:
        celdas = "".join(f"{f'{instancias[s][n]} ({imagenes_con[s][n]})':>18}" for s in splits)
        aviso = ""
        if any(s != "train" and instancias[s][n] < args.min_val for s in splits):
            aviso = "  <- pocas en evaluación: métricas poco fiables"
        print(f"  {n:<14}{celdas}{aviso}")
    print(f"\nDetalle en {out}/")


if __name__ == "__main__":
    main()
