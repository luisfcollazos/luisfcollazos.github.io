"""Utilidades compartidas: lectura del data.yaml de YOLO, rutas de etiquetas y recortes."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def cargar_data_yaml(data_yaml: str | Path, root: str | Path | None = None) -> dict:
    """Lee el data.yaml y resuelve las rutas de cada split a rutas absolutas."""
    data_yaml = Path(data_yaml).resolve()
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = Path(root) if root else Path(data.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    data["names"] = list(names)

    for split in ("train", "val", "test"):
        entrada = data.get(split)
        if not entrada:
            continue
        entradas = entrada if isinstance(entrada, list) else [entrada]
        data[split] = [str(p if Path(p).is_absolute() else base / p) for p in entradas]
    return data


def listar_imagenes(entradas: list[str]) -> list[Path]:
    """Expande carpetas o archivos .txt (como hace Ultralytics) a una lista de imágenes."""
    imagenes: list[Path] = []
    for entrada in entradas:
        p = Path(entrada)
        if p.is_dir():
            imagenes += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        elif p.suffix == ".txt" and p.is_file():
            for linea in p.read_text(encoding="utf-8").splitlines():
                linea = linea.strip()
                if linea:
                    f = Path(linea)
                    imagenes.append(f if f.is_absolute() else (p.parent / f).resolve())
        else:
            raise FileNotFoundError(f"No se encontró el split: {entrada}")
    return imagenes


def ruta_etiqueta(imagen: Path) -> Path:
    """Misma convención que Ultralytics: .../images/x.jpg -> .../labels/x.txt"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(imagen)
    if sa in s:
        s = sb.join(s.rsplit(sa, 1))
    return Path(s).with_suffix(".txt")


def leer_etiquetas(imagen: Path) -> list[tuple[int, float, float, float, float]]:
    """Devuelve [(clase, xc, yc, w, h)] normalizados. Ignora segmentos (toma su caja envolvente)."""
    f = ruta_etiqueta(imagen)
    if not f.is_file():
        return []
    filas = []
    for linea in f.read_text(encoding="utf-8").splitlines():
        v = linea.split()
        if len(v) < 5:
            continue
        cls, coords = int(float(v[0])), list(map(float, v[1:]))
        if len(coords) > 4:  # polígono de segmentación -> caja envolvente
            xs, ys = coords[0::2], coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            coords = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        filas.append((cls, *coords[:4]))
    return filas


def xywhn_a_xyxy(xc: float, yc: float, w: float, h: float, ancho: int, alto: int) -> tuple[float, float, float, float]:
    return ((xc - w / 2) * ancho, (yc - h / 2) * alto, (xc + w / 2) * ancho, (yc + h / 2) * alto)


def recortar(img: Image.Image, caja_xyxy, margen: float = 0.15, cuadrado: bool = True) -> Image.Image:
    """Recorta una caja con margen relativo; si `cuadrado`, usa el lado mayor para no deformar el grano."""
    x1, y1, x2, y2 = caja_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * (1 + 2 * margen), (y2 - y1) * (1 + 2 * margen)
    if cuadrado:
        w = h = max(w, h)
    caja = (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))
    # PIL rellena con negro lo que queda fuera de la imagen
    return img.crop(caja)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def medir_grano(recorte: Image.Image) -> dict | None:
    """Mide el grano del centro del recorte separándolo del fondo (Otsu + componente conectada).

    Devuelve área (px), largo y ancho del rectángulo mínimo rotado, independientes de la orientación,
    a diferencia de la caja de YOLO, que cambia si el grano está girado. Supone un fondo uniforme.
    Si hay granos pegados que se fusionan con el del centro, la medida sale inflada: `pegado=True`.
    """
    import cv2
    import numpy as np

    gris = cv2.cvtColor(np.asarray(recorte.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # el fondo es lo que domina en el borde del recorte; el grano es lo contrario
    borde = np.concatenate([binaria[0], binaria[-1], binaria[:, 0], binaria[:, -1]])
    if borde.mean() > 127:
        binaria = 255 - binaria
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria)
    if n < 2:
        return None
    h, w = binaria.shape
    comp = etiquetas[h // 2, w // 2]
    if comp == 0:  # el centro cayó en fondo: toma la componente más grande
        comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mascara = (etiquetas == comp).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (a, b), _ = cv2.minAreaRect(max(contornos, key=cv2.contourArea))
    x, y, cw, ch = stats[comp, :4]
    return {
        "area": float(stats[comp, cv2.CC_STAT_AREA]),
        "largo": float(max(a, b)),
        "ancho": float(min(a, b)),
        "pegado": bool(x == 0 or y == 0 or x + cw >= w or y + ch >= h),
    }
