"""Paso 3 — Entrena las dos etapas con aumentos de color suaves.

  detector     : localiza granos (una sola clase)
  clasificador : decide el defecto a partir del recorte de cada grano

Los aumentos HSV por defecto de Ultralytics (hsv_s=0.7, hsv_v=0.4) cambian justo el color que
distingue negro / oscuro / vinagre / oreado / cardenillo / blanqueado, por eso aquí se reducen.

También `completo`: reentrena el detector de 14 clases de una sola etapa con la misma configuración
(color suave, imgsz alto), como línea base justa para comparar con las dos etapas.

Uso:
  python 3_entrenar.py completo     --data data.yaml        --model yolo11s.pt     --imgsz 1280
  python 3_entrenar.py detector     --data data_1clase.yaml --model yolo11s.pt     --imgsz 1280
  python 3_entrenar.py clasificador --data dataset_cls      --model yolo11s-cls.pt --imgsz 224
"""

import argparse

from ultralytics import YOLO

COLOR_SUAVE = {"hsv_h": 0.0, "hsv_s": 0.1, "hsv_v": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["completo", "detector", "clasificador"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16, help="-1 = automático según la memoria de la GPU")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    comunes = {"data": args.data, "batch": args.batch, "patience": args.patience, **COLOR_SUAVE}
    if args.device is not None:
        comunes["device"] = args.device

    if args.etapa == "completo":
        model = YOLO(args.model or "yolo11s.pt")
        model.train(
            **comunes,
            imgsz=args.imgsz or 1280,
            epochs=args.epochs or 150,
            flipud=0.5,
            degrees=180,
            scale=0.25,  # menos reescalado: conserva mejor el tamaño, que distingue sanog/sanop y malla
            name="completo_granos",
        )
    elif args.etapa == "detector":
        model = YOLO(args.model or "yolo11s.pt")
        model.train(
            **comunes,
            single_cls=True,  # reutiliza las etiquetas originales tratándolas como "grano"
            imgsz=args.imgsz or 1280,
            epochs=args.epochs or 100,
            flipud=0.5,  # un grano no tiene "arriba": se puede voltear y rotar libremente
            degrees=180,
            name="det_granos",
        )
    else:
        model = YOLO(args.model or "yolo11s-cls.pt")
        model.train(
            **comunes,
            imgsz=args.imgsz or 224,
            epochs=args.epochs or 100,
            flipud=0.5,
            degrees=180,
            dropout=0.2,
            name="cls_granos",
        )


if __name__ == "__main__":
    main()
