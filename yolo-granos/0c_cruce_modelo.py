"""Paso 0c — Cruce etiqueta vs. predicción del modelo: detecta clases renombradas o índices corridos.

Para cada caja etiquetada busca la predicción del modelo con mayor IoU (>= --iou) y tabula
"índice de la etiqueta" contra "clase que predice el modelo". Aunque el modelo acierte solo un ~50 %,
un desajuste sistemático se ve claro:
  * si las cajas etiquetadas 'sanop' el modelo las llama mayoritariamente 'vinagre', y las 'sanog'
    las llama 'sanop', las etiquetas de este dataset están corridas respecto a las del entrenamiento
  * si el máximo de cada fila está en la diagonal, etiquetas y modelo hablan el mismo idioma

Uso:
  python 0c_cruce_modelo.py --model best.pt --data data.yaml --split val --imgsz 640
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

from comun import cargar_data_yaml, iou, leer_etiquetas, listar_imagenes, xywhn_a_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val", help="val, test, train o 'todos'")
    ap.add_argument("--imgsz", type=int, default=640, help="el mismo con el que se entrenó el modelo")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="auditoria/cruce_modelo.csv")
    args = ap.parse_args()

    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    model = YOLO(args.model)
    mnames = model.names
    splits = ("train", "val", "test") if args.split == "todos" else (args.split,)
    imagenes = [p for s in splits for p in listar_imagenes(data.get(s) or [])]

    cruce = defaultdict(Counter)  # etiqueta -> Counter(predicción)
    for k, img_path in enumerate(imagenes, 1):
        r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, agnostic_nms=True,
                          device=args.device, verbose=False)[0]
        preds = list(zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()))
        H, W = r.orig_shape
        for c, xc, yc, w, h in leer_etiquetas(img_path):
            caja = xywhn_a_xyxy(xc, yc, w, h, W, H)
            mejor = max(((iou(caja, p), pc) for p, pc in preds), default=(0.0, None))
            cruce[c][mnames[mejor[1]] if mejor[0] >= args.iou else "(no detectado)"] += 1
        if k % 20 == 0:
            print(f"  {k}/{len(imagenes)} imágenes")

    print(f"\n{'idx':>3}  {'etiqueta':<12}{'n':>6}   lo que predice el modelo (top 3)")
    filas = []
    for c in sorted(cruce):
        total = sum(cruce[c].values())
        nombre = names[c] if c < len(names) else "?"
        top = cruce[c].most_common(3)
        texto = ",  ".join(f"{p} {n / total:.0%}" for p, n in top)
        aviso = "" if top[0][0] in (nombre, "(no detectado)") else "   <- el modelo la ve como otra clase"
        print(f"{c:>3}  {nombre:<12}{total:>6}   {texto}{aviso}")
        filas += [[c, nombre, p, n, round(n / total, 3)] for p, n in cruce[c].most_common()]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["indice", "etiqueta", "prediccion_modelo", "n", "fraccion"])
        w.writerows(filas)
    print(f"\nTabla completa en {args.out}")


if __name__ == "__main__":
    main()
