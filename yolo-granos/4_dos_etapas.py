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

Umbrales por clase (--umbrales): un grano solo se asigna a un defecto si el clasificador está
suficientemente seguro; si no, queda como `sano` (--clase-tolerante). Así las imperfecciones leves, que se
toleran en un grano sano, no se cuentan como defecto. Con `--umbrales auto` se busca en el split evaluado
(val) el umbral de cada defecto que maximiza su F1 (o, con --precision-min, el menor que alcanza esa
precisión) y se guardan en <out>/umbrales.json; después se aplican en test con --umbrales <out>/umbrales.json.

Uso:
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --split val
  python 4_dos_etapas.py evaluar  ... --split val  --ajuste-prior 0.5 --umbrales auto --out res_val
  python 4_dos_etapas.py evaluar  ... --split test --ajuste-prior 0.5 --umbrales res_val/umbrales.json
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --split val \
      --cls-data dataset_cls --ajuste-prior 0,0.25,0.5,0.75,1
  python 4_dos_etapas.py evaluar  --det det.pt --cls cls.pt --data data.yaml --tamano "sano:ancho:23.5:sanog:sanop"
  python 4_dos_etapas.py predecir --det det.pt --cls cls.pt --source carpeta_imagenes/ --guardar-imagenes
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from comun import (IMG_EXTS, cargar_data_yaml, iou, leer_etiquetas, listar_imagenes, medir_grano,
                   normalizar_fondo, recortar, xywhn_a_xyxy)

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
                 tamano=None, ventana=0, normalizar=False, escala_ref=None):
        self.det, self.cls = YOLO(det), YOLO(cls)
        self.imgsz_det, self.imgsz_cls = imgsz_det, imgsz_cls
        self.conf, self.iou_nms, self.margen, self.device = conf, iou_nms, margen, device
        self.ventana = ventana
        self.normalizar = normalizar
        self.escala_ref = escala_ref  # lado mediano de un grano (px) en las fotos de entrenamiento
        self.ultima_escala = 1.0
        self.names = self.cls.names  # {i: nombre}
        self.tamano = tamano
        self.clases_salida = set(self.names.values())
        if tamano:
            if tamano["clase"] not in self.clases_salida:
                raise SystemExit(f"--tamano: el clasificador no tiene la clase '{tamano['clase']}'")
            self.clases_salida = (self.clases_salida - {tamano["clase"]}) | {tamano["grande"], tamano["pequeno"]}
        self.indice = {n: i for i, n in self.names.items()}
        self.umbrales = {}  # {defecto: probabilidad mínima}; por debajo el grano queda como self.tolerante
        self.tolerante = "sano"
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
        u = self.umbrales.get(nombre)
        if u and p[top] < u and self.tolerante in self.indice:  # imperfección leve: se tolera
            top = self.indice[self.tolerante]
            nombre = self.tolerante
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
        im = (img_path if isinstance(img_path, Image.Image) else Image.open(img_path)).convert("RGB")
        f = self.ultima_escala = self.factor_escala(im) if self.escala_ref else 1.0
        im_esc = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS) if f != 1.0 else im
        r = self.det.predict(im_esc, imgsz=self.imgsz_det, conf=self.conf, iou=self.iou_nms,
                             agnostic_nms=True, device=self.device, verbose=False)[0]
        cajas = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        conf_det = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        if not len(cajas):
            return im, []
        im_cls = normalizar_fondo(im_esc) if self.normalizar else im_esc  # el detector usa la foto sin normalizar
        recortes = [recortar(im_cls, c, self.margen, ventana=self.ventana) for c in cajas]
        salida = []
        # en lotes para no saturar memoria con imágenes de muchos granos
        for i in range(0, len(recortes), 64):
            res = self.cls.predict(recortes[i:i + 64], imgsz=self.imgsz_cls, device=self.device, verbose=False)
            for j, rc in enumerate(res):
                k = i + j
                medida = self._medida(recortes[k], cajas[k]) if self.tamano else None
                salida.append(((cajas[k] / f).tolist(), rc.probs.data.cpu().numpy().astype(float), medida,
                               float(conf_det[k])))
        return im, salida

    def factor_escala(self, im):
        """Factor para que el grano mediano de la foto mida lo mismo (px) que en el entrenamiento.

        Fotos de celular: otra resolución y otra distancia -> otro tamaño de grano en píxeles, y el modelo
        usa el tamaño (Oreado, malla). Se detecta una vez, se mide el lado mayor mediano de los granos y se
        reescala la foto. Se usa la mediana: la mayoría son sanos, así que no depende de los defectos."""
        r = self.det.predict(im, imgsz=self.imgsz_det, conf=self.conf, iou=self.iou_nms,
                             agnostic_nms=True, device=self.device, verbose=False)[0]
        if len(r.boxes) < 5:
            return 1.0
        b = r.boxes.xyxy.cpu().numpy()
        lado = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
        f = float(np.clip(self.escala_ref / lado, 1 / 3, 3))
        return 1.0 if abs(f - 1) < 0.03 else f

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

    # emparejamiento voraz por IoU descendente (como Ultralytics: solo la caja, no la clase): no depende
    # de TAU ni de los umbrales, así que se calcula una vez
    for k, (img_path, tam, crudos, gts) in enumerate(cache):
        cache[k] = (img_path, tam, crudos, gts, emparejar([c[0] for c in crudos], [g[1] for g in gts]))

    if args.umbrales == "auto" and len(taus) > 1:
        raise SystemExit("--umbrales auto se busca para un solo TAU: elige primero TAU en val (--ajuste-prior 0.5)")
    if args.umbrales and args.umbrales != "auto":
        pipe.umbrales = leer_umbrales(args.umbrales, taus[0])
        print("Umbrales por clase: " + ", ".join(f"{k}={v:g}" for k, v in sorted(pipe.umbrales.items())))

    def matriz(tau, errores=None):
        m = np.zeros((nc + 1, nc + 1), dtype=int)
        for img_path, tam, crudos, gts, pares in cache:
            preds = [(caja, *pipe.nombrar(probs, medida, tau)) for caja, probs, medida, _ in crudos]
            for pi, gi in pares.items():
                m[idx[preds[pi][1]], gts[gi][0]] += 1
                if errores is not None and preds[pi][1] != names[gts[gi][0]]:
                    errores.append((img_path, tam, gts[gi][1], names[gts[gi][0]], preds[pi][1], preds[pi][2]))
            for pi, p in enumerate(preds):
                if pi not in pares:
                    m[idx[p[1]], nc] += 1  # falso positivo
            for gi in set(range(len(gts))) - set(pares.values()):
                m[nc, gts[gi][0]] += 1  # no detectado
        return m

    resumen, errores = [], []
    for tau in taus:
        m = matriz(tau, errores if args.guardar_errores and tau == taus[0] and args.umbrales != "auto" else None)
        sufijo = f"_tau{tau:g}" if len(taus) > 1 else ""
        print(f"\n========== ajuste-prior TAU = {tau:g} ==========" if len(taus) > 1 else "")
        resumen.append((tau, *informe(m, names, Path(args.out), sufijo)))

    if args.umbrales == "auto":
        tau = taus[0]
        pipe.umbrales = buscar_umbrales(pipe, cache, names, tau, args.precision_min)
        m = matriz(tau, errores if args.guardar_errores else None)
        print("\n========== con umbrales por clase ==========")
        global_, macro = informe(m, names, Path(args.out), "_umbrales")
        print(f"\nAntes: aciertos {resumen[0][1]:.1%}, recall medio {resumen[0][2]:.1%}"
              f"   ->   con umbrales: aciertos {global_:.1%}, recall medio {macro:.1%}")
        destino = Path(args.out) / "umbrales.json"
        destino.write_text(json.dumps({"tau": tau, "tolerante": pipe.tolerante, "umbrales": pipe.umbrales},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Umbrales guardados en {destino}: aplícalos en test con --umbrales {destino} --ajuste-prior {tau:g}")

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


def emparejar(cajas_pred, cajas_gt, umbral=0.5):
    """{índice de predicción: índice de etiqueta} por IoU descendente, cada una usada una sola vez."""
    pares = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(cajas_pred) for gi, g in enumerate(cajas_gt)),
                   reverse=True)
    res, usados_g = {}, set()
    for v, pi, gi in pares:
        if v < umbral:
            break
        if pi not in res and gi not in usados_g:
            res[pi] = gi
            usados_g.add(gi)
    return res


def buscar_umbrales(pipe, cache, names, tau, precision_min=None):
    """Umbral de probabilidad por defecto: el que maximiza su F1 en este split (o el menor que alcanza
    `precision_min`). Los granos de ese defecto por debajo del umbral pasan a `pipe.tolerante`."""
    from collections import defaultdict

    pipe.umbrales = {}
    reales = defaultdict(int)
    por_clase = defaultdict(list)  # clase predicha -> [(probabilidad, ¿acierto?)]
    for _, _, crudos, gts, pares in cache:
        for c, *_ in gts:
            reales[names[c]] += 1
        for pi, (_, probs, medida, _) in enumerate(crudos):
            nombre, conf = pipe.nombrar(probs, medida, tau)
            gi = pares.get(pi)
            por_clase[nombre].append((conf, gi is not None and names[gts[gi][0]] == nombre))

    umbrales = {}
    print(f"\n===== Umbrales por clase (TAU {tau:g}; objetivo: "
          f"{f'precisión >= {precision_min:.0%}' if precision_min else 'máximo F1'}) =====")
    print(f"{'clase':<14}{'umbral':>8}{'precisión':>20}{'recall':>20}{'a ' + pipe.tolerante:>10}")
    for nombre in sorted(por_clase):
        if nombre == pipe.tolerante or not reales[nombre]:
            continue
        lista = por_clase[nombre]

        def medir(t):
            tp = sum(1 for p, ok in lista if p >= t and ok)
            n = sum(1 for p, _ in lista if p >= t)
            prec = tp / n if n else 0.0
            rec = tp / reales[nombre]
            return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0), len(lista) - n

        candidatos = [0.0] + [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
        evaluados = [(t, *medir(t)) for t in candidatos]
        if precision_min:
            ok = [e for e in evaluados if e[1] >= precision_min]
            elegido = ok[0] if ok else max(evaluados, key=lambda e: e[3])
        else:
            elegido = max(evaluados, key=lambda e: (e[3], -e[0]))
        base = evaluados[0]
        if elegido[0] > 0:
            umbrales[nombre] = elegido[0]
        print(f"{nombre:<14}{elegido[0]:>8.2f}{base[1]:>10.1%} -> {elegido[1]:<6.1%}"
              f"{base[2]:>10.1%} -> {elegido[2]:<6.1%}{elegido[4]:>8}")
    return umbrales


def leer_umbrales(spec, tau):
    """Archivo umbrales.json de --umbrales auto, o lista 'vinagre:0.6,malla:0.5'."""
    if spec.endswith(".json"):
        d = json.loads(Path(spec).read_text(encoding="utf-8"))
        if abs(d.get("tau", tau) - tau) > 1e-9:
            print(f"!! Los umbrales se buscaron con TAU {d['tau']:g} y ahora usas {tau:g}: usa el mismo TAU.")
        return {k: float(v) for k, v in d["umbrales"].items()}
    return {k: float(v) for k, v in (x.split(":") for x in spec.split(",") if x)}


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


# colores bien distinguibles; el primero (gris claro) es para la clase tolerante (sano)
PALETA = [(200, 200, 200), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
          (170, 110, 40), (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128)]


def fuente(tam):
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:  # Pillow antiguo
        return ImageFont.load_default()


def dibujar(im, preds, clases, tolerante, mostrar_confianza=False):
    """Foto con un color por clase, nombre sobre cada defecto y una leyenda con los conteos abajo.
    Los granos de la clase tolerante (sano) van con un recuadro fino gris, sin texto, para no tapar."""
    from collections import Counter

    color = {c: PALETA[k % len(PALETA)] for k, c in enumerate(clases)}
    W, H = im.size
    tam = max(14, W // 90)
    f = fuente(tam)
    d = ImageDraw.Draw(im)
    for caja, nombre, cc, _ in sorted(preds, key=lambda p: p[1] != tolerante):  # defectos encima
        if nombre == tolerante:
            d.rectangle(caja, outline=color[nombre], width=1)
            continue
        d.rectangle(caja, outline=color[nombre], width=max(2, W // 500))
        texto = f"{nombre} {cc:.0%}" if mostrar_confianza else nombre
        x, y = caja[0], max(0, caja[1] - tam - 4)
        b = d.textbbox((x, y), texto, font=f)
        if b[2] > W:
            x -= b[2] - W
            b = d.textbbox((x, y), texto, font=f)
        d.rectangle((b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2), fill=color[nombre])
        claro = sum(color[nombre]) > 450
        d.text((x, y), texto, font=f, fill=(0, 0, 0) if claro else (255, 255, 255))

    # leyenda
    cuenta = Counter(n for _, n, _, _ in preds)
    total = sum(cuenta.values())
    items = [c for c in clases if cuenta[c]]
    fl = fuente(int(tam * 1.1))
    alto_fila = int(tam * 1.8)
    por_fila = max(1, W // (tam * 14))
    filas = (len(items) + por_fila - 1) // por_fila + 1
    lienzo = Image.new("RGB", (W, H + filas * alto_fila + tam), (255, 255, 255))
    lienzo.paste(im, (0, 0))
    dl = ImageDraw.Draw(lienzo)
    defectos = total - cuenta[tolerante]
    dl.text((tam, H + tam // 2), f"{total} granos  |  defectos: {defectos} ({defectos / total:.1%})" if total
            else "sin granos", font=fl, fill=(0, 0, 0))
    for k, c in enumerate(items):
        x = tam + (k % por_fila) * (W // por_fila)
        y = H + tam // 2 + (k // por_fila + 1) * alto_fila
        dl.rectangle((x, y, x + tam, y + tam), fill=color[c], outline=(0, 0, 0))
        dl.text((x + tam * 1.5, y - 2), f"{c}: {cuenta[c]}", font=fl, fill=(0, 0, 0))
    return lienzo


def predecir(pipe, args):
    src = Path(args.source)
    imagenes = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from collections import Counter

    clases = [pipe.tolerante] + sorted(pipe.clases_salida - {pipe.tolerante}, key=str.lower)
    resumen = []
    with open(out / "predicciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "x1", "y1", "x2", "y2", "clase", "conf_clase", "conf_det"])
        for img_path in imagenes:
            im, preds = pipe(img_path, tau=float(str(args.ajuste_prior).split(",")[0]))
            resumen.append((img_path.name, Counter(nombre for _, nombre, _, _ in preds)))
            for caja, nombre, cc, cd in preds:
                w.writerow([img_path.name, *(round(v, 1) for v in caja), nombre, round(cc, 4), round(cd, 4)])
            if args.guardar_imagenes:
                dibujar(im, preds, clases, pipe.tolerante, args.mostrar_confianza).save(
                    out / f"{img_path.stem}.jpg", quality=92)
            print(f"{img_path.name}: {len(preds)} granos"
                  + (f"  (escala x{pipe.ultima_escala:.2f})" if pipe.escala_ref else ""))

    # resumen por foto (formato para Excel en español: ';' y coma decimal)
    def fila(nombre, cuenta):
        total = sum(cuenta.values())
        defectos = total - cuenta[pipe.tolerante]
        pct = f"{100 * defectos / total:.1f}".replace(".", ",") if total else ""
        return [nombre, total] + [cuenta[c] for c in clases] + [defectos, pct]

    total = sum((c for _, c in resumen), Counter())
    with open(out / "resumen_por_imagen.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["imagen", "granos"] + clases + ["defectos", "% defectos"])
        for nombre, cuenta in resumen:
            w.writerow(fila(nombre, cuenta))
        w.writerow(fila("TOTAL", total))

    n = sum(total.values())
    print(f"\nTotal: {len(resumen)} imágenes, {n} granos")
    for c in clases:
        if total[c]:
            print(f"  {c:<14}{total[c]:>7}  {total[c] / n:6.1%}")
    if n:
        print(f"  {'defectos':<14}{n - total[pipe.tolerante]:>7}  {(n - total[pipe.tolerante]) / n:6.1%}")
    print(f"\nResultados en {out}/predicciones.csv y {out}/resumen_por_imagen.csv")


def leer_escala(args):
    if args.escala in ("0", "", None):
        return None
    if args.escala != "auto":
        return float(args.escala)
    f = Path(args.cls_data or "") / "preproceso.json"
    ref = json.loads(f.read_text(encoding="utf-8")).get("lado_mediano_px") if f.is_file() else None
    if not ref:
        raise SystemExit("--escala auto necesita --cls-data con un preproceso.json reciente (vuelve a generar los "
                         "recortes con 2_recortar_granos.py) o pasa el tamaño a mano: --escala 66")
    print(f"Escala: los granos se llevan a un lado mediano de {ref:g} px, como en el entrenamiento")
    return float(ref)


def comprobar_preproceso(args):
    """Los recortes de inferencia deben hacerse igual que los de entrenamiento (preproceso.json de --cls-data)."""
    f = Path(args.cls_data or "") / "preproceso.json"
    if not args.cls_data or not f.is_file():
        return
    entreno = json.loads(f.read_text(encoding="utf-8"))
    ahora = {"ventana": args.ventana, "margen": args.margen, "normalizar_fondo": args.normalizar_fondo,
             "escala": args.escala not in ("0", "", None)}
    distintos = {k: (v, ahora[k]) for k, v in entreno.items() if k in ahora and ahora[k] != v}
    if distintos:
        raise SystemExit("Los recortes no se hacen igual que al entrenar el clasificador:\n" + "\n".join(
            f"  {k}: entrenamiento={a}  ahora={b}" for k, (a, b) in distintos.items()) +
            "\nUsa las mismas opciones que en 2_recortar_granos.py.")


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
    ap.add_argument("--normalizar-fondo", action="store_true",
                    help="debe coincidir con el usado en 2_recortar_granos.py")
    ap.add_argument("--escala", default="0", metavar="auto|PX|0",
                    help="reescala cada foto para que sus granos midan como en el entrenamiento (fotos de celular). "
                         "'auto' toma el tamaño de referencia de preproceso.json de --cls-data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", help="(evaluar) data.yaml original con las 14 clases")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ajuste-prior", default="0", metavar="TAU[,TAU...]",
                    help="corrige el balanceo del clasificador hacia la frecuencia real de las clases (0 = no)")
    ap.add_argument("--cls-data", default=None, help="carpeta de recortes del clasificador (para --ajuste-prior)")
    ap.add_argument("--umbrales", default=None, metavar="auto|umbrales.json|CLASE:P,...",
                    help="probabilidad mínima por defecto; por debajo el grano queda como --clase-tolerante. "
                         "'auto' (evaluar, en val) los busca y guarda en <out>/umbrales.json")
    ap.add_argument("--precision-min", type=float, default=None,
                    help="con --umbrales auto: menor umbral que alcanza esta precisión (si no, máximo F1)")
    ap.add_argument("--clase-tolerante", default="sano", help="clase a la que pasan los granos bajo el umbral")
    ap.add_argument("--guardar-errores", default=None, metavar="CARPETA",
                    help="(evaluar) guarda recortes de los granos mal clasificados, por confusión (usa el primer TAU)")
    ap.add_argument("--errores-max", type=int, default=80, help="máximo de recortes por confusión")
    ap.add_argument("--source", help="(predecir) imagen o carpeta")
    ap.add_argument("--guardar-imagenes", action="store_true",
                    help="(predecir) guarda cada foto con un color por clase y una leyenda con los conteos")
    ap.add_argument("--mostrar-confianza", action="store_true", help="añade la confianza al nombre de cada defecto")
    ap.add_argument("--out", default="resultados_dos_etapas")
    args = ap.parse_args()

    pipe = DosEtapas(args.det, args.cls, args.imgsz_det, args.imgsz_cls, args.conf, iou_nms=args.iou,
                     margen=args.margen, device=args.device, ventana=args.ventana,
                     normalizar=args.normalizar_fondo, escala_ref=leer_escala(args),
                     tamano=parsear_tamano(args.tamano) if args.tamano else None)
    comprobar_preproceso(args)
    pipe.tolerante = args.clase_tolerante
    if args.umbrales and pipe.tolerante not in pipe.indice:
        raise SystemExit(f"--clase-tolerante '{pipe.tolerante}' no es una clase del clasificador")
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
        if args.umbrales:
            if args.umbrales == "auto":
                ap.error("--umbrales auto solo en evaluar (en val); en predecir usa el umbrales.json resultante")
            pipe.umbrales = leer_umbrales(args.umbrales, float(str(args.ajuste_prior).split(",")[0]))
        predecir(pipe, args)


if __name__ == "__main__":
    main()
