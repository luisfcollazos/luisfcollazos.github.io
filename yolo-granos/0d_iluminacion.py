"""Paso 0d — ¿La iluminación y el color de fondo son iguales en todas las fotos?

Oscuro y Oreado se distinguen por el tono (grado de humedad). Si unas fotos salen más claras o más
amarillas que otras, el clasificador confunde esa diferencia de luz con la del grano. Este script mide,
en cada imagen, el color del FONDO (los píxeles que no caen dentro de ninguna caja) y el de los granos
sanos, que deberían verse igual en todas las fotos.

Lectura:
  * variación (p90/p10) menor de ~5 %: iluminación estable. Se puede bajar el aumento de brillo del
    clasificador (3_entrenar.py clasificador --hsv-v 0.05) para que respete mejor el tono.
  * variación mayor: hay fotos más claras/oscuras. Conviene fijar cámara, luz y balance de blancos al
    fotografiar; mientras tanto deja --hsv-v 0.15 para que el modelo tolere esa variación.
  Las imágenes más alejadas de la mediana se listan para revisarlas.

Uso:
  python 0d_iluminacion.py --data YoloDataset7/data.yaml
"""

import argparse

import numpy as np
from PIL import Image

from comun import cargar_data_yaml, leer_etiquetas, listar_imagenes, xywhn_a_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sano", default="sano", help="clase de referencia (granos que deberían verse iguales)")
    ap.add_argument("--peores", type=int, default=10)
    args = ap.parse_args()

    data = cargar_data_yaml(args.data)
    names = data["names"]
    i_sano = names.index(args.sano) if args.sano in names else None
    filas = []
    for split in ("train", "val", "test"):
        for img_path in listar_imagenes(data.get(split) or []):
            with Image.open(img_path) as im:
                a = np.asarray(im.convert("RGB").reduce(4), dtype=np.float32)  # 1/4 de resolución basta
            H, W = a.shape[:2]
            fondo = np.ones((H, W), bool)
            sano = np.zeros((H, W), bool)
            for c, *caja in leer_etiquetas(img_path):
                x1, y1, x2, y2 = (int(round(v)) for v in xywhn_a_xyxy(*caja, W, H))
                fondo[max(0, y1):y2, max(0, x1):x2] = False
                if c == i_sano:  # centro de la caja: evita el borde, que es fondo
                    dx, dy = (x2 - x1) // 4, (y2 - y1) // 4
                    sano[y1 + dy:y2 - dy, x1 + dx:x2 - dx] = True
            f = np.median(a[fondo], axis=0) if fondo.any() else np.full(3, np.nan)
            s = np.median(a[sano], axis=0) if sano.any() else np.full(3, np.nan)
            filas.append((split, img_path.name, f, s))

    def informe(titulo, valores):
        v = np.array([x for x in valores if not np.isnan(x).any()])
        if not len(v):
            return
        brillo = v.mean(1)
        p10, p50, p90 = np.percentile(brillo, [10, 50, 90])
        print(f"\n{titulo} ({len(v)} imágenes)")
        print(f"  brillo medio  p10={p10:.0f}  mediana={p50:.0f}  p90={p90:.0f}   variación p90/p10 = {p90 / p10 - 1:.1%}")
        med = np.median(v, axis=0)
        print(f"  color mediano R={med[0]:.0f} G={med[1]:.0f} B={med[2]:.0f}   "
              f"dispersión (p90-p10) R={np.ptp(np.percentile(v[:, 0], [10, 90])):.0f} "
              f"G={np.ptp(np.percentile(v[:, 1], [10, 90])):.0f} B={np.ptp(np.percentile(v[:, 2], [10, 90])):.0f}")

    informe("FONDO", [f for _, _, f, _ in filas])
    informe(f"GRANOS '{args.sano}' (centro del grano)", [s for _, _, _, s in filas])

    print("\nPor split (brillo mediano del fondo):")
    for split in ("train", "val", "test"):
        b = [f.mean() for sp, _, f, _ in filas if sp == split and not np.isnan(f).any()]
        if b:
            print(f"  {split:<6} {np.median(b):6.0f}   ({len(b)} imágenes)")

    ref = np.nanmedian(np.array([f for _, _, f, _ in filas]), axis=0)
    lejos = sorted(filas, key=lambda r: -np.abs(r[2] - ref).sum() if not np.isnan(r[2]).any() else 0)
    print(f"\nImágenes con el fondo más distinto al habitual (R,G,B; habitual {ref.round().astype(int).tolist()}):")
    for split, nombre, f, _ in lejos[:args.peores]:
        print(f"  {split:<6} {nombre:<50} {f.round().astype(int).tolist()}")


if __name__ == "__main__":
    main()
