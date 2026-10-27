"""Paso 0e — ¿En qué se diferencian las fotos nuevas (por ejemplo, de celular) de las del dataset?

El modelo aprendió con fotos de una cámara, a una distancia y con una luz concretas. Si las fotos nuevas
cambian en algo de eso, la precisión baja. Este script mide lo mismo en una muestra del dataset y en las
fotos nuevas, usando el detector, y dice qué cambió:

  * tamaño del grano en píxeles: otra resolución/distancia. El clasificador usa el tamaño (Oreado, malla):
    se corrige con `4_dos_etapas.py --escala auto`.
  * nitidez: fotos movidas, desenfocadas o muy comprimidas. No se corrige después: hay que mejorar la toma
    o entrenar con fotos así.
  * color del fondo y de los granos (tras normalizar con el fondo): otra luz o el procesado del celular
    (balance de blancos, HDR, saturación). Oscuro/Oreado/vinagre dependen del tono.

Uso:
  python 0e_comparar_fotos.py --det det.pt --data YoloDataset8/data.yaml --fotos fotos_celular
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from comun import IMG_EXTS, cargar_data_yaml, listar_imagenes, normalizar_fondo


def medir(det, img_path, imgsz, ref=None):
    import cv2

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    r = det.predict(im, imgsz=imgsz, conf=0.25, agnostic_nms=True, verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    fila = {"imagen": img_path.name, "resolucion": f"{W}x{H}", "granos": len(b)}
    if len(b) < 3:
        return fila
    lado = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
    a = np.asarray(im, dtype=np.float32)
    fondo = np.median(a[::8, ::8].reshape(-1, 3), axis=0)
    norm = np.asarray(normalizar_fondo(im), dtype=np.float32)
    # a la escala del dataset, para que nitidez y color se midan igual en las dos fuentes
    f = (ref / lado) if ref else 1.0
    gris = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    nitidez, color = [], []
    for x1, y1, x2, y2 in b[:200]:
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        rec = gris[max(0, y1):y2, max(0, x1):x2]
        if rec.size < 25:
            continue
        rec = cv2.resize(rec, (max(8, round(rec.shape[1] * f)), max(8, round(rec.shape[0] * f))))
        nitidez.append(cv2.Laplacian(rec, cv2.CV_64F).var())
        dx, dy = (x2 - x1) // 4, (y2 - y1) // 4  # centro del grano: sin el borde, que es fondo
        color.append(np.median(norm[y1 + dy:y2 - dy, x1 + dx:x2 - dx].reshape(-1, 3), axis=0))
    fila.update({"lado_px": round(lado, 1), "brillo_fondo": round(float(fondo.mean()), 1),
                 "fondo_rgb": "/".join(str(int(v)) for v in fondo),
                 "nitidez": round(float(np.median(nitidez)), 1) if nitidez else None,
                 "grano_rgb": "/".join(str(int(v)) for v in np.median(color, axis=0)) if color else ""})
    return fila


def resumen(filas, clave):
    v = np.array([f[clave] for f in filas if f.get(clave) is not None], dtype=float)
    if not len(v):
        return None, "—"
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return p50, f"{p50:.0f} [{p10:.0f}–{p90:.0f}]"


def rgb_mediano(filas):
    v = np.array([[int(x) for x in f["grano_rgb"].split("/")] for f in filas if f.get("grano_rgb")])
    return np.median(v, axis=0) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="detector (best.pt)")
    ap.add_argument("--data", required=True, help="data.yaml del dataset de entrenamiento")
    ap.add_argument("--fotos", required=True, help="carpeta con las fotos nuevas")
    ap.add_argument("--muestra", type=int, default=20, help="imágenes del dataset a medir")
    ap.add_argument("--imgsz-det", type=int, default=1280)
    ap.add_argument("--out", default="comparacion_fotos.csv")
    args = ap.parse_args()

    det = YOLO(args.det)
    data = cargar_data_yaml(args.data)
    ds = listar_imagenes(data.get("val") or data["train"])
    ds = ds[:: max(1, len(ds) // args.muestra)][:args.muestra]
    nuevas = sorted(p for p in Path(args.fotos).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not nuevas:
        raise SystemExit(f"No hay imágenes en {args.fotos}")

    print(f"Midiendo {len(ds)} imágenes del dataset y {len(nuevas)} fotos nuevas...")
    f_ds = [medir(det, p, args.imgsz_det) for p in ds]
    ref, _ = resumen(f_ds, "lado_px")
    f_ds = [medir(det, p, args.imgsz_det, ref) for p in ds]
    f_nv = [medir(det, p, args.imgsz_det, ref) for p in nuevas]

    print(f"\n{'':<26}{'dataset':>22}{'fotos nuevas':>22}   (mediana [p10–p90])")
    vals = {}
    for clave, titulo in (("granos", "granos detectados"), ("lado_px", "tamaño del grano (px)"),
                          ("brillo_fondo", "brillo del fondo"), ("nitidez", "nitidez del grano")):
        a, ta = resumen(f_ds, clave)
        b, tb = resumen(f_nv, clave)
        vals[clave] = (a, b)
        print(f"{titulo:<26}{ta:>22}{tb:>22}")
    ca, cb = rgb_mediano(f_ds), rgb_mediano(f_nv)
    if ca is not None and cb is not None:
        print(f"{'color del grano (R/G/B)':<26}{'/'.join(map(str, ca.astype(int))):>22}"
              f"{'/'.join(map(str, cb.astype(int))):>22}   (tras normalizar con el fondo)")
    print(f"{'resolución':<26}{f_ds[0]['resolucion']:>22}{f_nv[0]['resolucion']:>22}")

    print("\nLectura:")
    a, b = vals["lado_px"]
    if a and b:
        r = b / a
        if abs(r - 1) > 0.1:
            print(f"  * Los granos miden {r:.2f} veces lo del dataset: usa --escala auto en 4_dos_etapas.py.")
        else:
            print("  * Tamaño del grano similar: la escala no es el problema.")
    a, b = vals["nitidez"]
    if a and b:
        if b < 0.5 * a:
            print(f"  * Fotos mucho menos nítidas ({b / a:.0%} de la nitidez del dataset): desenfoque, movimiento o "
                  "compresión. Mejora la toma (apoyo fijo, enfocar tocando la pantalla, buena luz) o entrena con "
                  "fotos de celular etiquetadas.")
        else:
            print(f"  * Nitidez aceptable ({b / a:.0%} de la del dataset).")
    a, b = vals["brillo_fondo"]
    if a and b and abs(b - a) > 20:
        print(f"  * El fondo es distinto (brillo {b:.0f} vs {a:.0f}): usa el mismo fondo blanco mate y "
              "--normalizar-fondo.")
    if ca is not None and cb is not None:
        dif = np.abs(cb - ca).max()
        if dif > 15:
            print(f"  * El color de los granos difiere hasta {dif:.0f} niveles incluso tras normalizar: el celular "
                  "procesa el color distinto (HDR, saturación, balance de blancos). Desactiva HDR/filtros o "
                  "entrena con fotos de celular.")
        else:
            print("  * Color de los granos similar tras normalizar.")
    a, b = vals["granos"]
    if a and b and b < 0.7 * a:
        print("  * Se detectan muchos menos granos por foto: puede que el detector no los encuentre (revisa las "
              "imágenes de predecir --guardar-imagenes) o que las fotos tengan menos granos.")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, delimiter=";", fieldnames=["fuente", "imagen", "resolucion", "granos", "lado_px",
                                                          "brillo_fondo", "fondo_rgb", "nitidez", "grano_rgb"])
        w.writeheader()
        for fuente, filas in (("dataset", f_ds), ("nuevas", f_nv)):
            for f in filas:
                w.writerow({"fuente": fuente, **f})
    print(f"\nDetalle por imagen en {args.out}")


if __name__ == "__main__":
    main()
