# Granos de café: diagnóstico y pipeline en dos etapas (YOLO)

Scripts para diagnosticar el modelo YOLO de 14 clases de defectos y probar la alternativa
**detector (1 clase) + clasificador (14 clases)**.

```bash
pip install -r requirements.txt   # ultralytics >= 8.3
```

## 0. Antes de nada: revisa el umbral de la matriz de confusión

En Ultralytics 8.4+, la `confusion_matrix.png` que genera el entrenamiento se calcula con **conf = 0.001**,
el mismo umbral de validación que usa el mAP. Con ese umbral:

- la fila *background* (no detectados) queda casi en 0;
- la columna *background* (sobrantes) se llena de cajas de confianza muy baja;
- una caja de baja confianza con la clase equivocada puede quedar emparejada con el grano y contar como confusión.

Si la matriz original tiene ~1 no detectado y ~9.000 sobrantes, es muy probable que esto la esté distorsionando.
El paso 1 la recalcula con un umbral realista.

## 1. Diagnóstico del modelo actual

```bash
python 1_diagnostico_val.py --model ruta/best.pt --data data.yaml --imgsz 640
```

Compara cuatro configuraciones e imprime una tabla con P, R, mAP, aciertos, mal clasificados,
no detectados y sobrantes:

| fila | qué mide |
|---|---|
| `normal@0.001` | la matriz tal como la genera el entrenamiento |
| `normal` | la matriz con `--conf-matriz` (0.25 por defecto) |
| `agnostic_nms` | igual, pero con NMS sin tener en cuenta la clase (elimina cajas duplicadas de distinta clase sobre el mismo grano) |
| `single_cls` | solo localización: si el mAP50 es alto, el problema es clasificar y conviene el enfoque en dos etapas |

Las matrices quedan en `runs/detect/diag_*_matriz/`.

## 2. Dataset de clasificación (recortes de cada grano)

```bash
python 2_recortar_granos.py --data data.yaml --out dataset_cls --balancear 300 --yaml-detector data_1clase.yaml
```

- Recorta cada caja **etiquetada** con un 15 % de margen y en formato cuadrado (`--margen`).
- `--balancear N` duplica recortes en *train* para las clases con menos de N ejemplos.
- `--yaml-detector` crea un `data.yaml` de una sola clase que reutiliza las mismas imágenes y etiquetas.
- Imprime cuántos recortes hay por clase y split, lo que sirve para ver el desbalance.

## 3. Entrenamiento

```bash
python 3_entrenar.py detector     --data data_1clase.yaml --model yolo11s.pt     --imgsz 1024
python 3_entrenar.py clasificador --data dataset_cls      --model yolo11s-cls.pt --imgsz 224
```

Ambos usan **aumentos de color suaves** (`hsv_h=0, hsv_s=0.1, hsv_v=0.15`) para no destruir el color
que distingue negro, oscuro, vinagre, oreado, cardenillo y blanqueado. También usan rotación libre y volteo
vertical, porque un grano no tiene orientación.

## 4. Evaluación y uso del pipeline

```bash
# matriz de confusión comparable a la de Ultralytics (mismo formato, IoU >= 0.5, conf >= 0.25)
python 4_dos_etapas.py evaluar  --det runs/detect/det_granos/weights/best.pt \
                                --cls runs/classify/cls_granos/weights/best.pt --data data.yaml

# predicción sobre imágenes nuevas -> predicciones.csv (+ imágenes anotadas)
python 4_dos_etapas.py predecir --det ... --cls ... --source carpeta/ --guardar-imagenes
```

Para comparar de forma justa, contrasta la salida de `evaluar` con la fila `normal` del paso 1
(mismo umbral), no con la matriz original a conf 0.001.
