"""Paso 4 — Inferencia y evaluación del pipeline detector + clasificador.

Modo `predecir`: procesa imágenes y escribe un CSV (imagen, caja, clase, confianzas) y, opcionalmente,
imágenes anotadas.

Modo `evaluar`: corre el pipeline sobre un split del data.yaml ORIGINAL (14 clases), empareja cada
detección con la etiqueta real (IoU >= 0.5) y construye una matriz de confusión con el mismo formato
que la de Ultralytics (filas = predicción, columnas = real, última fila/columna = background), para
compararla directamente con la del modelo de una sola etapa.

Separación por tamaño: si el clasificador se entrenó con una clase fusionada (ej. `sano` = sanog + sanop),
--tamano "sano:ancho:23.5:sanog:sanop" mide cada grano clasificado como `sano` y lo asigna a `sanog` si
su ancho >= 23.5 px, o a `sanop` si no. El umbral y la métrica salen de 0_auditar_dataset.py.

Ajuste por frecuencia de clases (--ajuste-prior): el clasificador se entrena con clases balanceadas
(--balancear / --max-por-clase), así que "cree" que todas las clases son igual de frecuentes y le quita
granos a la clase dominante (sanog). Con TAU > 0 se multiplican sus probabilidades por
(frecuencia real / frecuencia en el entrenamiento)^TAU: TAU=0 no corrige, TAU=1 corrige del todo.
Se pueden probar varios valores en una pasada (--ajuste-prior 0,0.5,1). Elige TAU en val, no en test.

Uso:
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --split val
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --split val \
      --cls-data dataset_cls --ajuste-prior 0,0.25,0.5,0.75,1
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --tamano "sano:ancho:23.5:sanog:sanop"
  python 4_dos_etapas.py predecir --det det.pt --cls cls.pt --source carpeta_imagenes/ --guardar-imagenes
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from comun import (IMG_EXTS, cargar_data_yaml, iou, leer_etiquetas, listar_imagenes, medir_grano, recortar,
                   xywhn_a_xyxy)

METRICAS = ("ancho", "largo", "area")


def parsear_tamano(spec):
    """'sano:ancho:23.5:sanog:sanop' -> dict"""
    try:
        clase, metrica, umbral, grande, pequeno = spec.split(":")
        assert metrica in METRICAS
        return {"clase": clase, "metrica": metrica, "umbral": float(umbral), "grande": grande, "pequeno": pequeno}
    except (ValueError, AssertionError):
        raise SystemExit(f"--tamano inválido: {spec!r}. Formato: CLASE:{'|'.join(METRICAS)}:UMBRAL:GRANDE:PEQUENO")


def medida_caja(caja, metrica):
    """Respaldo cuando no se puede aislar el grano: medida aproximada con la caja (depende de la orientación)."""
    w, h = caja[2] - caja[0], caja[3] - caja[1]
    return {"ancho": min(w, h), "largo": max(w, h), "area": w * h}[metrica]


class DosEtapas:
    def __init__(self, det, cls, imgsz_det=1024, imgsz_cls=224, conf=0.25, iou_nms=0.5, margen=0.15, device=None,
                 tamano=None, ventana=0):
        self.det, self.cls = YOLO(det), YOLO(cls)
        self.imgsz_det, self.imgsz_cls = imgsz_det, imgsz_cls
        self.conf, self.iou_nms, self.margen, self.device = conf, iou_nms, margen, device
        self.ventana = ventana
        self.names = self.cls.names  # {i: nombre}
        self.tamano = tamano
        self.clases_salida = set(self.names.values())
        if tamano:
            if tamano["clase"] not in self.clases_salida:
                raise SystemExit(f"--tamano: el clasificador no tiene la clase '{tamano['clase']}'")
            self.clases_salida = (self.clases_salida - {tamano["clase"]}) | {tamano["grande"], tamano["pequeno"]}
        self.sin_medida = 0  # granos en los que no se pudo aislar el grano y se usó la caja
        self.log_ratio = np.zeros(len(self.names))

    def fijar_prior(self, frec_real: dict, frec_entreno: dict):
        """Guarda log(frec_real / frec_entreno) por índice del clasificador (1 = sin corrección)."""
        self.log_ratio = np.zeros(len(self.names))
        for i, n in self.names.items():
            if frec_real.get(n) and frec_entreno.get(n):
                self.log_ratio[i] = np.log(frec_real[n] / frec_entreno[n])

    def nombrar(self, probs, medida, tau=0.0):
        """Clase final a partir de las probabilidades del clasificador (y el tamaño si hay --tamano)."""
        p = probs * np.exp(tau * self.log_ratio) if tau else probs
        p = p / p.sum()
        top = int(p.argmax())
        nombre = self.names[top]
        if self.tamano and nombre == self.tamano["clase"]:
            t = self.tamano
            nombre = t["grande"] if medida >= t["umbral"] else t["pequeno"]
        return nombre, float(p[top])

    def _medida(self, recorte, caja):
        m = medir_grano(recorte)
        if m is None or m["pegado"]:
            self.sin_medida += 1
            return medida_caja(caja, self.tamano["metrica"])
        return m[self.tamano["metrica"]]

    def crudo(self, img_path):
        """Detecta y clasifica: devuelve (imagen, [(xyxy, probs, medida, conf_det)])."""
        im = Image.open(img_path).convert("RGB")
        r = self.det.predict(im, imgsz=self.imgsz_det, conf=self.conf, iou=self.iou_nms,
                             agnostic_nms=True, device=self.device, verbose=False)[0]
        cajas = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        conf_det = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        if not len(cajas):
            return im, []
        recortes = [recortar(im, c, self.margen, ventana=self.ventana) for c in cajas]
        salida = []
        # en lotes para no saturar memoria con imágenes de muchos granos
        for i in range(0, len(recortes), 64):
            res = self.cls.predict(recortes[i:i + 64], imgsz=self.imgsz_cls, device=self.device, verbose=False)
            for j, rc in enumerate(res):
                k = i + j
                medida = self._medida(recortes[k], cajas[k]) if self.tamano else None
                salida.append((cajas[k].tolist(), rc.probs.data.cpu().numpy().astype(float), medida,
                               float(conf_det[k])))
        return im, salida

    def __call__(self, img_path, tau=0.0):
        """Devuelve (imagen, [(xyxy, nombre_clase, conf_clase, conf_det)])."""
        im, crudos = self.crudo(img_path)
        salida = []
        for caja, probs, medida, cd in crudos:
            nombre, conf = self.nombrar(probs, medida, tau)
            salida.append((caja, nombre, conf, cd))
        return im, salida


def contar_clases(data, split="train"):
    """Etiquetas por nombre de clase en un split del data.yaml (frecuencia real)."""
    from collections import Counter
    c = Counter()
    for img in listar_imagenes(data.get(split) or []):
        c.update(data["names"][e[0]] for e in leer_etiquetas(img))
    return c


def contar_recortes(cls_data):
    """Recortes por clase en dataset_cls/train (frecuencia con la que se entrenó el clasificador)."""
    raiz = Path(cls_data) / "train"
    return {d.name: sum(1 for _ in d.glob("*.jpg")) for d in raiz.iterdir() if d.is_dir()}


def evaluar(pipe, args):
    data = cargar_data_yaml(args.data, args.root)
    names = data["names"]
    nc = len(names)
    faltan = set(names) - pipe.clases_salida
    if faltan:
        raise SystemExit(f"El clasificador no tiene las clases: {sorted(faltan)}")
    idx = {n: i for i, n in enumerate(names)}
    taus = [float(t) for t in str(args.ajuste_prior).split(",")]
    if any(taus):
        if not args.cls_data:
            raise SystemExit("--ajuste-prior necesita --cls-data (la carpeta de recortes con la que se entrenó)")
        pipe.fijar_prior(contar_clases(data, "train"), contar_recortes(args.cls_data))

    # detección y clasificación una sola vez; luego se evalúa cada tau
    imagenes = listar_imagenes(data[args.split])
    cache = []
    for n_img, img_path in enumerate(imagenes, 1):
        im, crudos = pipe.crudo(img_path)
        W, H = im.size
        gts = [(c, xywhn_a_xyxy(xc, yc, w, h, W, H)) for c, xc, yc, w, h in leer_etiquetas(img_path)]
        cache.append((img_path, (W, H), crudos, gts))
        if n_img % 50 == 0:
            print(f"  {n_img}/{len(imagenes)} imágenes")

    resumen, errores = [], []
    for tau in taus:
        m = np.zeros((nc + 1, nc + 1), dtype=int)
        for img_path, tam, crudos, gts in cache:
            preds = [(caja, *pipe.nombrar(probs, medida, tau)) for caja, probs, medida, _ in crudos]
            # emparejamiento voraz por IoU descendente (como Ultralytics: solo la caja, no la clase)
            pares = sorted(((iou(p[0], g[1]), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
                           reverse=True)
            usados_p, usados_g = set(), set()
            for v, pi, gi in pares:
                if v < 0.5:
                    break
                if pi in usados_p or gi in usados_g:
                    continue
                usados_p.add(pi)
                usados_g.add(gi)
                m[idx[preds[pi][1]], gts[gi][0]] += 1
                if args.guardar_errores and tau == taus[0] and preds[pi][1] != names[gts[gi][0]]:
                    errores.append((img_path, tam, gts[gi][1], names[gts[gi][0]], preds[pi][1], preds[pi][2]))
            for pi, p in enumerate(preds):
                if pi not in usados_p:
                    m[idx[p[1]], nc] += 1  # falso positivo
            for gi, g in enumerate(gts):
                if gi not in usados_g:
                    m[nc, g[0]] += 1  # no detectado
        sufijo = f"_tau{tau:g}" if len(taus) > 1 else ""
        print(f"\n========== ajuste-prior TAU = {tau:g} ==========" if len(taus) > 1 else "")
        resumen.append((tau, *informe(m, names, Path(args.out), sufijo)))

    if len(taus) > 1:
        print("\n===== Resumen por TAU =====")
        print(f"{'TAU':>6}{'aciertos':>11}{'recall medio por clase':>25}")
        for tau, global_, macro in resumen:
            print(f"{tau:>6g}{global_:>11.1%}{macro:>25.1%}")
        print("Elige TAU en val (no en test) y luego evalúa test una sola vez con ese valor.")
    if args.guardar_errores:
        guardar_errores(errores, Path(args.guardar_errores), args.errores_max, taus[0])
    if pipe.tamano:
        print(f"\nGranos medidos con la caja por no poder aislarlos (pegados/fondo irregular): {pipe.sin_medida}")


def guardar_errores(errores, out, maximo, tau):
    """Recortes de los granos mal clasificados, agrupados por confusión, para revisarlos en Label Studio.

    out/<real>__como__<predicho>/NNN_<imagen>.jpg   recorte con la caja (ordenados por confianza, mayor primero)
    out/<real>__como__<predicho>/_mosaico.jpg       todos los recortes del par en una sola imagen
    out/errores.csv                                  imagen y posición en % (como en Label Studio) de cada uno
    """
    from collections import defaultdict

    grupos = defaultdict(list)
    for e in errores:
        grupos[(e[3], e[4])].append(e)
    out.mkdir(parents=True, exist_ok=True)
    filas, lado = [], 160
    print(f"\n===== Errores guardados en {out} (TAU {tau:g}) =====")
    for (real, pred), lista in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        lista.sort(key=lambda e: -e[5])  # más seguro primero: más probable que la etiqueta esté mal
        carpeta = out / f"{real}__como__{pred}"
        carpeta.mkdir(exist_ok=True)
        celdas = []
        for k, (img_path, (W, H), b, _, _, conf) in enumerate(lista[:maximo], 1):
            v = 3 * max(b[2] - b[0], b[3] - b[1])
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0, y0 = cx - v / 2, cy - v / 2
            with Image.open(img_path) as im:
                rec = im.convert("RGB").crop((int(x0), int(y0), int(x0 + v), int(y0 + v)))
            esc = lado / rec.width
            rec = rec.resize((lado, lado))
            d = ImageDraw.Draw(rec)
            d.rectangle(((b[0] - x0) * esc, (b[1] - y0) * esc, (b[2] - x0) * esc, (b[3] - y0) * esc),
                        outline=(255, 0, 0), width=2)
            d.rectangle((0, 0, 62, 13), fill=(0, 0, 0))
            d.text((3, 1), f"{k} ({conf:.0%})", fill=(255, 255, 0))
            rec.save(carpeta / f"{k:03d}_{img_path.stem}.jpg", quality=92)
            celdas.append(rec)
            pos = (f"x={100 * b[0] / W:.1f}% y={100 * b[1] / H:.1f}% "
                   f"w={100 * (b[2] - b[0]) / W:.1f}% h={100 * (b[3] - b[1]) / H:.1f}%")
            filas.append([carpeta.name, k, img_path.name, real, pred, round(conf, 3), pos])
        cols = 8
        mosaico = Image.new("RGB", (cols * lado, ((len(celdas) + cols - 1) // cols) * lado), (255, 255, 255))
        for k, c in enumerate(celdas):
            mosaico.paste(c, ((k % cols) * lado, (k // cols) * lado))
        mosaico.save(carpeta / "_mosaico.jpg", quality=90)
        print(f"  {real:>11} como {pred:<11} {len(lista):>4}  (guardados {min(len(lista), maximo)})")
    with open(out / "errores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["carpeta", "n", "imagen", "etiqueta", "modelo_dice", "confianza", "posicion_label_studio"])
        w.writerows(filas)


def informe(m, names, out, sufijo=""):
    nc = len(names)
    reales = m[:, :nc].sum(0)
    pred = m[:nc, :].sum(1)
    diag = m.diagonal()[:nc]
    recalls = [diag[i] / reales[i] for i in range(nc) if reales[i]]
    global_ = diag.sum() / max(reales.sum(), 1)
    macro = float(np.mean(recalls)) if recalls else float("nan")
    print(f"\nGranos reales: {reales.sum()}  |  clase correcta: {diag.sum()} ({global_:.1%})"
          f"  |  no detectados: {m[nc, :nc].sum()}  |  sobrantes (FP): {m[:nc, nc].sum()}")
    print(f"Recall promedio por clase (cada clase pesa igual): {macro:.1%}")
    print(f"\n{'clase':<14}{'reales':>8}{'recall':>9}{'precisión':>11}")
    for i, n in enumerate(names):
        rec = diag[i] / reales[i] if reales[i] else float("nan")
        prec = diag[i] / pred[i] if pred[i] else float("nan")
        print(f"{n:<14}{reales[i]:>8}{rec:>9.1%}{prec:>11.1%}")

    out.mkdir(parents=True, exist_ok=True)
    etiquetas = names + ["background"]
    with open(out / f"matriz_confusion{sufijo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred \\ real"] + etiquetas)
        for i, fila in enumerate(m):
            w.writerow([etiquetas[i]] + fila.tolist())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="white" if v > m.max() * 0.6 else "black")
        ax.set_xticks(range(nc + 1), etiquetas, rotation=90)
        ax.set_yticks(range(nc + 1), etiquetas)
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicho")
        ax.set_title(f"Matriz de confusión — dos etapas{sufijo.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(out / f"matriz_confusion{sufijo}.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(f"\nMatriz guardada en {out}/matriz_confusion{sufijo}.csv (y .png)")
    return global_, macro


def predecir(pipe, args):
    src = Path(args.source)
    imagenes = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predicciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "x1", "y1", "x2", "y2", "clase", "conf_clase", "conf_det"])
        for img_path in imagenes:
            im, preds = pipe(img_path, tau=float(str(args.ajuste_prior).split(",")[0]))
            for caja, nombre, cc, cd in preds:
                w.writerow([img_path.name, *(round(v, 1) for v in caja), nombre, round(cc, 4), round(cd, 4)])
            if args.guardar_imagenes:
                d = ImageDraw.Draw(im)
                for caja, nombre, cc, _ in preds:
                    d.rectangle(caja, outline=(255, 0, 0), width=2)
                    d.text((caja[0] + 2, caja[1] + 2), f"{nombre} {cc:.2f}", fill=(255, 255, 0))
                im.save(out / img_path.name)
            print(f"{img_path.name}: {len(preds)} granos")
    print(f"\nResultados en {out}/predicciones.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["evaluar", "predecir"])
    ap.add_argument("--det", required=True, help="pesos del detector de una clase")
    ap.add_argument("--cls", required=True, help="pesos del clasificador")
    ap.add_argument("--imgsz-det", type=int, default=1280)
    ap.add_argument("--imgsz-cls", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="NMS; debe ser mayor que el IoU típico entre granos pegados (ver 0_auditar_dataset.py)")
    ap.add_argument("--tamano", default=None, metavar="CLASE:METRICA:UMBRAL:GRANDE:PEQUENO",
                    help="separa una clase fusionada por tamaño medido, ej. sano:ancho:23.5:sanog:sanop")
    ap.add_argument("--margen", type=float, default=0.15, help="debe coincidir con el usado en 2_recortar_granos.py")
    ap.add_argument("--ventana", type=int, default=0, help="debe coincidir con el usado en 2_recortar_granos.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", help="(evaluar) data.yaml original con las 14 clases")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ajuste-prior", default="0", metavar="TAU[,TAU...]",
                    help="corrige el balanceo del clasificador hacia la frecuencia real de las clases (0 = no)")
    ap.add_argument("--cls-data", default=None, help="carpeta de recortes del clasificador (para --ajuste-prior)")
    ap.add_argument("--guardar-errores", default=None, metavar="CARPETA",
                    help="(evaluar) guarda recortes de los granos mal clasificados, por confusión (usa el primer TAU)")
    ap.add_argument("--errores-max", type=int, default=80, help="máximo de recortes por confusión")
    ap.add_argument("--source", help="(predecir) imagen o carpeta")
    ap.add_argument("--guardar-imagenes", action="store_true")
    ap.add_argument("--out", default="resultados_dos_etapas")
    args = ap.parse_args()

    pipe = DosEtapas(args.det, args.cls, args.imgsz_det, args.imgsz_cls, args.conf, iou_nms=args.iou,
                     margen=args.margen, device=args.device, ventana=args.ventana,
                     tamano=parsear_tamano(args.tamano) if args.tamano else None)
    if args.modo == "evaluar":
        if not args.data:
            ap.error("evaluar requiere --data")
        evaluar(pipe, args)
    else:
        if not args.source:
            ap.error("predecir requiere --source")
        if float(str(args.ajuste_prior).split(",")[0]):
            if not (args.data and args.cls_data):
                ap.error("--ajuste-prior en predecir necesita --data (frecuencia real) y --cls-data")
            pipe.fijar_prior(contar_clases(cargar_data_yaml(args.data, args.root), "train"),
                             contar_recortes(args.cls_data))
        predecir(pipe, args)


if __name__ == "__main__":
    main()
