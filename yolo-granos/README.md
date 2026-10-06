# Granos de café: diagnóstico y pipeline en dos etapas (YOLO)

Scripts para diagnosticar el modelo YOLO de 14 clases de defectos y probar la alternativa
**detector (1 clase) + clasificador (14 clases)**.

```bash
pip install -r requirements.txt   # ultralytics >= 8.3
```

## Re-exportar desde Label Studio y empezar limpio

```bash
# 1. ver y luego borrar caches de Ultralytics (*.cache, *.npy) y salidas de estos scripts
python limpiar_caches.py --dataset ruta/dataset_viejo --salidas
python limpiar_caches.py --dataset ruta/dataset_viejo --salidas --borrar

# 2. export YOLO de Label Studio -> dataset nuevo con splits estratificados y data.yaml
python preparar_export.py --export ruta/export_labelstudio --out dataset_v2 --val 0.15 --test 0.15

# 3. verificar antes de entrenar
python 0_auditar_dataset.py   --data dataset_v2/data.yaml
python 0b_mosaico_clases.py   --data dataset_v2/data.yaml
```

`preparar_export.py` toma los nombres del `classes.txt` del export (el mismo orden que los índices),
valida las etiquetas y nunca escribe sobre un dataset existente. Así no sobreviven `.txt` viejos de un export anterior.

## 0. Antes de nada: revisa el umbral de la matriz de confusión

En Ultralytics 8.4+, la `confusion_matrix.png` que genera el entrenamiento se calcula con **conf = 0.001**,
el mismo umbral de validación que usa el mAP. Con ese umbral:

- la fila *background* (no detectados) queda casi en 0;
- la columna *background* (sobrantes) se llena de cajas de confianza muy baja;
- una caja de baja confianza con la clase equivocada puede quedar emparejada con el grano y contar como confusión.

Si la matriz original tiene ~1 no detectado y ~9.000 sobrantes, es muy probable que esto la esté distorsionando.
El paso 1 la recalcula con un umbral realista.

## Auditoría del dataset (hacer primero)

```bash
python 0_auditar_dataset.py --data data.yaml --grande sanog --pequeno sanop --dibujar 30
```

| Revisión | Qué reporta | Qué hacer |
|---|---|---|
| **Cajas solapadas** | duplicados (IoU ≥ 0.6) con la misma clase o con clases distintas, cajas que contienen a otra (abarcan varios granos) e IoU entre granos que se tocan | Corregir los duplicados (sobre todo los de distinta clase: el modelo recibe dos respuestas para el mismo grano). Revisar las muestras en `auditoria/muestras/`. |
| **Tamaño por clase** | ancho, largo y área del grano real (segmentado contra el fondo, sin depender de la orientación). Para sanog/sanop busca el mejor umbral único | Si el umbral acierta más de 95 %, fusionar las clases y **medir** el tamaño (ver abajo). Si acierta menos de 90 %, las etiquetas de tamaño no siguen una regla fija y conviene re-etiquetar. |
| **Clases raras** | instancias e imágenes por clase y split, con aviso si hay pocas en val/test | Ver la sección "Clases con pocas imágenes". |

Detalle en `auditoria/problemas_solapamiento.csv` y `auditoria/tamanos.csv`.

También reporta las **cajas anormalmente pequeñas** por clase (`caja_pequena` en el CSV). Suelen indicar que
en algunas imágenes se etiquetó solo el defecto, por ejemplo el orificio de broca, y en otras el grano entero.

### Verificación visual de cada clase

```bash
python 0b_mosaico_clases.py --data data.yaml --modelo ruta/best.pt
```

- Imprime la tabla índice → nombre en el `data.yaml` → nombre guardado en el modelo → número de etiquetas.
  Marca los índices sin etiquetas y los nombres que no coinciden.
- Genera un mosaico por índice en `auditoria/mosaicos/`. Todos los recortes usan la misma ventana en píxeles,
  así que también se compara el tamaño. Si el mosaico de "vinagre" no muestra granos vinagre, los nombres
  están desalineados con los índices de las etiquetas.

### Cruce etiquetas vs. modelo

```bash
python 0c_cruce_modelo.py --model ruta/best.pt --data data.yaml --split val --imgsz 640
```

Para cada caja etiquetada, muestra qué clase predice el modelo entrenado. Aunque el modelo acierte poco, un
desajuste sistemático (por ejemplo, las cajas "sanop" predichas sobre todo como "vinagre") indica que las
etiquetas del dataset están corridas o renombradas respecto a las que se usaron para entrenar.

> Medir en píxeles solo es válido si la cámara está siempre a la misma distancia. La auditoría avisa si hay
> imágenes con resoluciones distintas.

## sanog/sanop: medir el tamaño en vez de aprenderlo

YOLO no conserva bien el tamaño absoluto: el aumento `scale` reescala las imágenes durante el entrenamiento
y el clasificador recibe cada recorte reescalado a 224 px. Si dos clases solo se diferencian por tamaño,
es más fiable que el modelo diga "sano" y que el tamaño se mida:

```bash
python 2_recortar_granos.py --data data.yaml --out dataset_cls --fusionar sano=sanog,sanop --balancear 300
python 3_entrenar.py clasificador --data dataset_cls
python 4_dos_etapas.py evaluar --det ... --cls ... --data data.yaml --tamano "sano:ancho:UMBRAL:sanog:sanop"
```

`UMBRAL` y la métrica (`ancho`, `largo` o `area`) salen de la auditoría. El ancho es lo más parecido a
cómo clasifica una malla o tamiz. Los granos que no se pueden aislar porque están pegados se miden con la caja,
de forma aproximada, y se informa cuántos fueron.

## Clases con pocas imágenes (vinagre, broca)

- **En el pipeline de dos etapas el detector no se ve afectado**: para él todo es "grano". El desbalance
  solo pesa en el clasificador, donde `--balancear N` duplica recortes de las clases escasas (cada copia se
  aumenta distinto en cada época).
- Asegura que val y test tengan suficientes ejemplos de estas clases. Con 10–20 instancias, el recall de
  una clase cambia ±10 puntos por 2 granos.
- Lo que más ayuda es **capturar más imágenes de esas clases**. Por ejemplo, fotografiar bandejas solo con granos
  vinagre o broca separados de muestras ya clasificadas por un catador.
- Broca es un orificio pequeño: necesita resolución. Prueba el clasificador con `--imgsz 320`.

## Granos pegados

- Si los granos se tocan, la caja de uno incluye parte del vecino, y eso confunde tanto al detector como
  al recorte que ve el clasificador.
- La solución más efectiva está en la captura: separar los granos, por ejemplo con una bandeja con
  alvéolos o una vibración ligera.
- En el modelo, la alternativa es pasar a **segmentación** (`yolo11s-seg`). La máscara separa los granos
  aunque se toquen y permite medir el tamaño con exactitud. Requiere etiquetar con polígonos.

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
python 2_recortar_granos.py --data data.yaml --out dataset_cls --balancear 600 --max-por-clase 3000 --yaml-detector data_1clase.yaml
```

- Recorta cada caja **etiquetada** con un 15 % de margen y en formato cuadrado (`--margen`).
- `--balancear N` duplica recortes en *train* para las clases con menos de N ejemplos, y `--max-por-clase M`
  submuestrea la clase dominante a M recortes.
- `--yaml-detector` crea un `data.yaml` de una sola clase que reutiliza las mismas imágenes y etiquetas.
- Imprime cuántos recortes hay por clase y split, lo que sirve para ver el desbalance.

## 3. Entrenamiento

```bash
python 3_entrenar.py completo     --data data.yaml        --model yolo11s.pt     --imgsz 1280   # línea base 1 etapa
python 3_entrenar.py detector     --data data_1clase.yaml --model yolo11s.pt     --imgsz 1024
python 3_entrenar.py clasificador --data dataset_cls      --model yolo11s-cls.pt --imgsz 224
```

Ambos usan **aumentos de color suaves** (`hsv_h=0, hsv_s=0.1, hsv_v=0.15`) para no destruir el color
que distingue negro, oscuro, vinagre, oreado, cardenillo y blanqueado. También usan rotación libre y volteo
vertical, porque un grano no tiene orientación.

## 4. Evaluación y uso del pipeline

```bash
# matriz de confusión comparable a la de Ultralytics (mismo formato, IoU >= 0.5, conf >= 0.25)
# --iou ajusta el NMS (ver "Cajas que se tocan" en la auditoría); --tamano separa clases fusionadas
python 4_dos_etapas.py evaluar  --det runs/detect/det_granos/weights/best.pt \
                                --cls runs/classify/cls_granos/weights/best.pt --data data.yaml

# predicción sobre imágenes nuevas -> predicciones.csv (+ imágenes anotadas)
python 4_dos_etapas.py predecir --det ... --cls ... --source carpeta/ --guardar-imagenes
```

Para comparar de forma justa, contrasta la salida de `evaluar` con la fila `normal` del paso 1
(mismo umbral), no con la matriz original a conf 0.001.
