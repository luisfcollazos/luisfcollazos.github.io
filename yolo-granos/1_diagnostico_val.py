"""Paso 1 — Diagnóstico del modelo actual.

Valida el mismo modelo de tres formas y compara:
  * normal          : NMS por clase (lo que produjo la matriz original)
  * agnostic_nms    : un solo NMS para todas las clases -> elimina cajas duplicadas de distinta clase
  * single_cls      : ignora la clase -> mide SOLO la calidad de localización

Lectura:
  - Si con agnostic_nms caen mucho los falsos positivos (columna background), el problema eran
    cajas duplicadas sobre el mismo grano.
  - Si siguen altos, probablemente hay granos SIN ETIQUETAR en las imágenes de validación.
  - Si single_cls da mAP50 alto (>0.9) el detector localiza bien y conviene el enfoque en dos etapas.

OJO con el umbral: en Ultralytics 8.4+ la matriz de confusión de `val` usa conf=0.001 (el mismo umbral
que el cálculo del mAP). Con ese umbral casi no hay "no detectados" pero aparecen miles de "sobrantes" y
cajas de muy baja confianza con la clase equivocada. Por eso este script calcula las métricas con el
umbral por defecto y las matrices aparte con --conf-matriz (0.25 por defecto), e incluye la fila
`normal@0.001` para comparar con la matriz que genera el entrenamiento.

Uso:
  python 1_diagnostico_val.py --model runs/detect/train/weights/best.pt --data data.yaml
"""

import argparse

from ultralytics import YOLO


def resumen_matriz(cm, names):
    """Filas = predicción, columnas = real; la última fila/columna es background."""
    m = cm.matrix
    nc = len(names)
    reales = m[:, :nc].sum(0)
    aciertos = m.diagonal()[:nc]
    fp = m[:nc, nc].sum()  # predicho clase X, no había grano
    fn = m[nc, :nc].sum()  # grano real no detectado
    mal_clasificados = reales.sum() - aciertos.sum() - fn
    return {
        "granos_reales": int(reales.sum()),
        "clase_correcta": int(aciertos.sum()),
        "mal_clasificados": int(mal_clasificados),
        "no_detectados(FN)": int(fn),
        "sobrantes(FP)": int(fp),
    }, {n: (aciertos[i] / reales[i] if reales[i] else float("nan")) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf-matriz", type=float, default=0.25, help="umbral de confianza para la matriz de confusión")
    args = ap.parse_args()

    model = YOLO(args.model)
    names = list(model.names.values())
    configs = {
        "normal@0.001": ({}, 0.001),
        "normal": ({}, args.conf_matriz),
        "agnostic_nms": ({"agnostic_nms": True}, args.conf_matriz),
        "single_cls": ({"single_cls": True}, None),
    }
    comunes = {"data": args.data, "imgsz": args.imgsz, "split": args.split, "batch": args.batch}
    filas = {}
    for nombre, (extra, conf_matriz) in configs.items():
        print(f"\n===== Validando: {nombre} =====")
        fila = {}
        if nombre != "normal@0.001":  # mismas métricas que "normal"
            m = model.val(**comunes, plots=False, name=f"diag_{nombre}", **extra)
            fila = {"P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map}
        if conf_matriz is not None:
            # segunda pasada solo para la matriz: el umbral de val también filtra la matriz de confusión
            m = model.val(**comunes, plots=True, conf=conf_matriz, name=f"diag_{nombre}_matriz", **extra)
            conteos, recall_cls = resumen_matriz(m.confusion_matrix, names)
            fila.update(conteos)
            if nombre == "normal":
                print(f"\nRecall por clase (diagonal / reales) con conf>={conf_matriz}:")
                for n, r in sorted(recall_cls.items(), key=lambda x: x[1]):
                    print(f"  {n:<14} {r:6.1%}")
                validos = [r for r in recall_cls.values() if r == r]  # sin NaN
                print(f"  {'PROMEDIO':<14} {sum(validos) / len(validos):6.1%}   (cada clase pesa igual)")
        filas[nombre] = fila

    print("\n===== Comparación =====")
    columnas = list(filas["normal"])
    print(f"{'config':<14}" + "".join(f"{c:>18}" for c in columnas))
    for nombre, f in filas.items():
        celdas = []
        for c in columnas:
            v = f.get(c, "")
            celdas.append(f"{v:>18.3f}" if isinstance(v, float) else f"{v!s:>18}")
        print(f"{nombre:<14}" + "".join(celdas))
    print("\nLas matrices de confusión quedaron en runs/detect/diag_*/")


if __name__ == "__main__":
    main()
