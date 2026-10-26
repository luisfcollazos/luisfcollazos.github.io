"""Pre-etiqueta en Label Studio las imágenes nuevas con el modelo de dos etapas.

En vez de dibujar ~100 cajas por imagen, el modelo propone una caja y una clase por grano y en Label Studio
solo corriges lo que esté mal y envías (Submit). Las propuestas se suben como "predicciones": no cuentan
como etiquetas hasta que las revisas y envías, así que nunca entran al dataset sin pasar por ti.

Pasos:
  1. Sube las imágenes nuevas al MISMO proyecto de Label Studio (Import).
  2. Copia tu token: Label Studio -> Account & Settings -> Access Token (o Personal Access Token).
  3. Ejecuta este script (en tu PC, con Label Studio abierto; no necesita GPU):

     python preetiquetar_ls.py --url http://localhost:8080 --token TU_TOKEN --proyecto 1 \\
         --det det_best.pt --cls cls_best.pt --ventana 160 --normalizar-fondo --tamano "sano:ancho:UMBRAL:sanog:sanop"

     El número de proyecto es el de la dirección del navegador: .../projects/1/data
  4. En Label Studio abre cada tarea, corrige las cajas y clases, y pulsa Submit.

Solo se procesan las tareas SIN etiquetas ni predicciones (las 300 ya etiquetadas no se tocan); con
--reemplazar también las que ya tienen predicciones (por ejemplo, tras entrenar un modelo mejor).

Clases: el clasificador usa `sano` (sanog+sanop fusionadas), pero el proyecto conserva sanog y sanop. Con
--tamano cada sano se asigna a sanog o sanop por su tamaño medido (el umbral sale de 0_auditar_dataset.py
sobre un dataset SIN fusionar). La separación por tamaño acierta ~80 %: revisa esos granos. Si ya no te
interesa distinguirlos, usa --renombrar sano=sanog en lugar de --tamano.

Escribe además preetiquetas.csv con los granos propuestos por clase en cada imagen: ordénalo por las clases
raras (vinagre, malla, broca...) para etiquetar primero las imágenes que más aportan.
"""

import argparse
import csv
import importlib.util
import io
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

from comun import IMG_EXTS, cargar_data_yaml

_spec = importlib.util.spec_from_file_location("dos_etapas", Path(__file__).with_name("4_dos_etapas.py"))
dos_etapas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dos_etapas)

PREFIJO_LS = re.compile(r"^[0-9a-f]{8}-")  # Label Studio antepone 8 caracteres al subir un archivo


class LabelStudio:
    """Cliente mínimo de la API. Acepta el token clásico y el Personal Access Token (JWT)."""

    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token
        self.s = requests.Session()
        self.jwt = token.count(".") == 2
        self._autenticar()

    def _autenticar(self):
        if self.jwt:  # el Personal Access Token es un "refresh token": se cambia por uno de acceso
            r = requests.post(f"{self.url}/api/token/refresh/", json={"refresh": self.token})
            if r.status_code != 200:
                raise SystemExit(f"Token rechazado ({r.status_code}): {r.text[:200]}")
            self.s.headers["Authorization"] = f"Bearer {r.json()['access']}"
        else:
            self.s.headers["Authorization"] = f"Token {self.token}"

    def pedir(self, metodo, ruta, **kw):
        url = ruta if ruta.startswith("http") else f"{self.url}{ruta}"
        r = self.s.request(metodo, url, **kw)
        if r.status_code == 401 and self.jwt:  # el token de acceso caduca a los pocos minutos
            self._autenticar()
            r = self.s.request(metodo, url, **kw)
        if r.status_code >= 400:
            raise SystemExit(f"{metodo} {ruta} -> {r.status_code}: {r.text[:300]}")
        return r

    def tareas(self, proyecto):
        pagina, tam = 1, 100
        while True:
            r = self.pedir("GET", "/api/tasks/", params={"project": proyecto, "page": pagina, "page_size": tam})
            d = r.json()
            lote = d["tasks"] if isinstance(d, dict) else d
            yield from lote
            total = d.get("total") if isinstance(d, dict) else None
            if len(lote) < tam or (total is not None and pagina * tam >= total):
                return
            pagina += 1


def leer_config(xml):
    """Nombres de la herramienta de etiquetado (rectángulos o polígonos) y de la imagen, y las etiquetas."""
    raiz = ET.fromstring(xml)
    imagen = raiz.find(".//Image")
    herramienta = raiz.find(".//RectangleLabels")
    tipo = "rectanglelabels"
    if herramienta is None:
        herramienta, tipo = raiz.find(".//PolygonLabels"), "polygonlabels"
    if herramienta is None or imagen is None:
        raise SystemExit("La configuración del proyecto no tiene <Image> con <RectangleLabels> o <PolygonLabels>")
    etiquetas = [e.get("value") for e in herramienta.findall("Label")]
    return {"from_name": herramienta.get("name"), "to_name": herramienta.get("toName"), "tipo": tipo,
            "clave_imagen": imagen.get("value").lstrip("$"), "etiquetas": etiquetas}


def resultado(caja, nombre, conf, W, H, cfg):
    x1, y1, x2, y2 = caja
    x, y, w, h = x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100
    if cfg["tipo"] == "rectanglelabels":
        valor = {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "rectanglelabels": [nombre]}
    else:
        valor = {"points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], "closed": True,
                 "polygonlabels": [nombre]}
    return {"from_name": cfg["from_name"], "to_name": cfg["to_name"], "type": cfg["tipo"],
            "original_width": W, "original_height": H, "image_rotation": 0, "value": valor,
            "score": round(conf, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--proyecto", type=int, required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--imgsz-det", type=int, default=1280)
    ap.add_argument("--ventana", type=int, default=160, help="la misma que en 2_recortar_granos.py")
    ap.add_argument("--normalizar-fondo", action="store_true", help="si se usó en 2_recortar_granos.py")
    ap.add_argument("--umbrales", default=None, help="umbrales.json de 4_dos_etapas.py --umbrales auto")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tamano", default=None, metavar="CLASE:METRICA:UMBRAL:GRANDE:PEQUENO")
    ap.add_argument("--renombrar", action="append", default=[], metavar="MODELO=LABEL_STUDIO",
                    help="usa otro nombre de clase en Label Studio, ej. sano=sanog")
    ap.add_argument("--ajuste-prior", type=float, default=0.0, metavar="TAU",
                    help="como en 4_dos_etapas.py; necesita --data y --cls-data")
    ap.add_argument("--data", default=None)
    ap.add_argument("--cls-data", default=None)
    ap.add_argument("--imagenes", default=None,
                    help="carpeta con las imágenes originales (si no, se descargan de Label Studio)")
    ap.add_argument("--reemplazar", action="store_true", help="procesa también tareas que ya tienen predicciones")
    ap.add_argument("--max", type=int, default=0, help="procesa solo N tareas (para probar)")
    ap.add_argument("--out", default="preetiquetas.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ls = LabelStudio(args.url, args.token)
    cfg = leer_config(ls.pedir("GET", f"/api/projects/{args.proyecto}/").json()["label_config"])

    pipe = dos_etapas.DosEtapas(args.det, args.cls, args.imgsz_det, conf=args.conf, device=args.device,
                                ventana=args.ventana, normalizar=args.normalizar_fondo,
                                tamano=dos_etapas.parsear_tamano(args.tamano) if args.tamano else None)
    if args.ajuste_prior:
        if not (args.data and args.cls_data):
            ap.error("--ajuste-prior necesita --data y --cls-data")
        pipe.fijar_prior(dos_etapas.contar_clases(cargar_data_yaml(args.data), "train"),
                         dos_etapas.contar_recortes(args.cls_data))
    if args.umbrales:
        pipe.umbrales = dos_etapas.leer_umbrales(args.umbrales, args.ajuste_prior)
    renombrar = dict(r.split("=", 1) for r in args.renombrar)
    salida = {c: renombrar.get(c, c) for c in pipe.clases_salida}
    faltan = sorted(n for n in salida.values() if n not in cfg["etiquetas"])
    if faltan:
        raise SystemExit(f"Estas clases del modelo no existen en el proyecto: {faltan}\n"
                         f"Etiquetas del proyecto: {cfg['etiquetas']}\n"
                         "Usa --tamano (sano -> sanog/sanop) o --renombrar MODELO=LABEL_STUDIO.")

    locales = {}
    if args.imagenes:
        for p in Path(args.imagenes).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                locales[p.name] = locales[PREFIJO_LS.sub("", p.name)] = p

    pendientes = [t for t in ls.tareas(args.proyecto)
                  if not t.get("total_annotations") and (args.reemplazar or not t.get("total_predictions"))]
    if args.max:
        pendientes = pendientes[:args.max]
    print(f"{len(pendientes)} tareas sin etiquetar para pre-etiquetar")

    filas, clases = [], sorted(set(salida.values()))
    for k, t in enumerate(pendientes, 1):
        ruta = t["data"][cfg["clave_imagen"]]
        nombre = Path(ruta.split("?d=")[-1]).name
        local = locales.get(nombre) or locales.get(PREFIJO_LS.sub("", nombre))
        if local:
            im = Image.open(local)
        else:
            im = Image.open(io.BytesIO(ls.pedir("GET", ruta).content))
        im = im.convert("RGB")
        _, preds = pipe(im, tau=args.ajuste_prior)
        W, H = im.size
        res = [resultado(c, salida[n], cc, W, H, cfg) for c, n, cc, _ in preds]
        conf_media = sum(r["score"] for r in res) / len(res) if res else 0.0
        if args.reemplazar:
            for p in ls.pedir("GET", "/api/predictions/", params={"task": t["id"]}).json():
                ls.pedir("DELETE", f"/api/predictions/{p['id']}/")
        ls.pedir("POST", "/api/predictions/", json={"task": t["id"], "result": res, "score": round(conf_media, 4),
                                                    "model_version": Path(args.cls).parent.parent.name or "dos_etapas"})
        cuenta = Counter(salida[n] for _, n, _, _ in preds)
        filas.append([t["id"], PREFIJO_LS.sub("", nombre), len(preds)] + [cuenta[c] for c in clases])
        print(f"[{k}/{len(pendientes)}] tarea {t['id']} {nombre}: {len(preds)} granos")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tarea", "imagen", "granos"] + clases)
        w.writerows(filas)
    print(f"\nListo. Resumen por imagen en {args.out}. Revisa y envía (Submit) cada tarea en Label Studio.")


if __name__ == "__main__":
    main()
