"""Borra caches y salidas generadas para empezar limpio tras re-etiquetar.

Por defecto solo LISTA lo que borraría; añade --borrar para hacerlo.

  * *.cache        caches de etiquetas de Ultralytics (labels/train.cache, labels/val.cache, ...)
                   y de clasificación (dataset_cls/train.cache)
  * *.npy          imágenes cacheadas con cache='disk' (junto a las imágenes)
  * salidas de estos scripts: auditoria/, auditoria_entreno/, dataset_cls/, resultados_dos_etapas/,
    data_1clase.yaml   (con --salidas)
  * runs/          entrenamientos y validaciones anteriores   (solo con --runs; guarda antes el best.pt
                   del modelo actual si lo quieres comparar con el nuevo)

Uso:
  python limpiar_caches.py --dataset ruta/al/dataset --salidas          # ver qué se borraría
  python limpiar_caches.py --dataset ruta/al/dataset --salidas --borrar
"""

import argparse
import shutil
from pathlib import Path

SALIDAS = ["auditoria", "auditoria_entreno", "dataset_cls", "resultados_dos_etapas", "data_1clase.yaml"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[], help="carpeta del dataset (se puede repetir)")
    ap.add_argument("--salidas", action="store_true", help="incluye las salidas de los scripts de esta carpeta")
    ap.add_argument("--runs", action="store_true", help="incluye la carpeta runs/")
    ap.add_argument("--base", default=".", help="carpeta donde están las salidas y runs/")
    ap.add_argument("--borrar", action="store_true")
    args = ap.parse_args()

    objetivos = []
    for d in map(Path, args.dataset):
        if not d.is_dir():
            raise SystemExit(f"No existe: {d}")
        objetivos += sorted(d.rglob("*.cache")) + sorted(d.rglob("*.npy"))
    base = Path(args.base)
    if args.salidas:
        objetivos += [base / s for s in SALIDAS if (base / s).exists()]
    if args.runs and (base / "runs").exists():
        objetivos.append(base / "runs")

    if not objetivos:
        print("Nada que borrar.")
        return
    for p in objetivos:
        print(("borrando  " if args.borrar else "borraría  ") + str(p))
        if args.borrar:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not args.borrar:
        print(f"\n{len(objetivos)} elementos. Repite con --borrar para eliminarlos.")


if __name__ == "__main__":
    main()
