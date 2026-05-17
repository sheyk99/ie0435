# Proyecto 1 IE0435 — Clasificacion de Contaminaciones en Linea de Produccion

**Estudiante:** Sheyla Miller Segura — C34919  
**Curso:** IE0435 Inteligencia Artificial aplicada a ingenieria electrica — I-2026  
**Universidad de Costa Rica**

---

## Descripcion

Este repositorio contiene el pipeline completo para detectar contaminaciones (granos de arroz) en una linea de produccion simulada. El flujo va desde las fotos crudas tomadas con el celular hasta un modelo de clasificacion entrenado y exportado, pasando por preprocesamiento, generacion del dataset y entrenamiento.

---

## Estructura del repositorio

```
IE0435/
├── datasets/                        # Datasets de compañeros del grupo
│   ├── dataset_adrian_mendez.csv
│   ├── dataset_alex_viquez.csv
│   ├── dataset_eduardo_loria.csv
│   ├── dataset_sheyla_miller.csv
│   └── dataset-final.csv
│
├── fotos-raw/                       # Fotos originales tomadas con el celular
│
├── modelos/                         # Todos los modelos entrenados
│   ├── Arbol_Decision.joblib
│   ├── KNN.joblib
│   ├── Naive_Bayes.joblib
│   └── SVM_Lineal.joblib
│
├── procesadas/                      # Imagenes preprocesadas (128x128, grises)
│   ├── conArroz/                    # Imagenes con granos de arroz (label = 1)
│   ├── sinArroz/                    # Imagenes sin arroz (label = 0)
│   ├── dataset.csv                  # Dataset generado a partir de las imagenes
│   └── procesar-dataset.py          # Script para generar el dataset CSV
│
├── src/                             # Scripts principales
│   ├── procesar.py                  # Paso 1: preprocesar fotos crudas
│   └── entrenar_modelos.py          # Paso 3: entrenar y exportar modelos
│
├── C34919_sheyla_miller.joblib      # Mejor modelo (archivo de entrega)
├── DATASET.md
├── LICENSE
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

---

## Flujo de trabajo

### Paso 1 — Preprocesar fotos (`src/procesar.py`)

Las fotos en `fotos-raw/` fueron tomadas con camara de celular sobre una hoja blanca que simula la banda de produccion. El script las procesa automaticamente:

1. Detecta el contorno de la hoja blanca y recorta el fondo sobrante.
2. Convierte la imagen a escala de grises.
3. Redimensiona a 128x128 pixeles (interpolacion INTER_AREA).
4. Guarda las imagenes resultantes en `procesadas/`.

```bash
# Ejecutar desde la raiz del proyecto
python src/procesar.py
```

> **Nota sobre el umbral:** se ajusto el umbral de binarizacion a 200/255 para lograr una separacion consistente del fondo blanco en diferentes condiciones de iluminacion. Pixeles con valor mayor o igual a 200 se consideran fondo (1), el resto se consideran objeto (0).

---

### Paso 2 — Clasificacion manual

Las imagenes procesadas se clasificaron manualmente segun el criterio del proyecto:

- `procesadas/conArroz/` — imagenes con al menos un grano de arroz visible (**label = 1**)
- `procesadas/sinArroz/` — imagenes sin arroz; pueden tener aros, clips o estar vacias (**label = 0**)

---

### Paso 3 — Generar dataset (`procesadas/procesar-dataset.py`)

Lee las imagenes clasificadas, las binariza y genera el archivo CSV.

```bash
# Ejecutar desde la carpeta procesadas/
cd procesadas
python procesar-dataset.py
```

El script:
1. Lee cada imagen en escala de grises.
2. Aplica binarizacion con metodo Otsu (umbral calculado automaticamente).
3. Convierte pixeles: blanco (fondo) -> 1, negro (objeto) -> 0.
4. Aplana la matriz 128x128 a un vector de 16,384 valores.
5. Agrega la etiqueta al final (1 = arroz, 0 = no arroz).
6. Guarda todo en `dataset.csv` — una fila por imagen, 16,385 columnas.

---

### Paso 4 — Entrenar modelos (`src/entrenar_modelos.py`)

Carga el dataset propio y los de companeros, entrena 4 modelos con busqueda de hiperparametros y exporta el mejor.

```bash
# Ejecutar desde la raiz del proyecto
python src/entrenar_modelos.py
```

Modelos evaluados:
- KNN (K-Nearest Neighbors)
- Arbol de Decision
- Naive Bayes
- SVM (Support Vector Machine)

La seleccion del mejor modelo se basa en el **F1-score** obtenido en el conjunto de prueba (split 80/20, `random_state=42`).

**Resultados obtenidos:**

| Modelo        | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| Decision Tree | 0.8000   | 0.7368    | 0.9333 | **0.8235** |
| KNN           | 0.7333   | 0.6522    | 1.0000 | 0.7895   |
| SVM           | 0.7667   | 0.7222    | 0.8667 | 0.7879   |
| Naive Bayes   | 0.7333   | 0.7692    | 0.6667 | 0.7143   |

**Modelo seleccionado:** Decision Tree (`criterion=gini`, `max_depth=10`, `min_samples_split=2`)

---

## Inferencia con el modelo exportado

```python
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load("C34919_sheyla_miller.joblib")

# Cargar una imagen y convertirla al formato esperado
import cv2
img = cv2.imread("mi_imagen.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
_, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
vector = (binaria / 255).astype(int).flatten().reshape(1, -1)

# Predecir
prediccion = modelo.predict(vector)
print("Arroz detectado" if prediccion[0] == 1 else "Sin arroz")
```

---

## Instalacion de dependencias

```bash
pip install -r requirements.txt
```

---

## Prompts de IA utilizados

Los scripts de este repositorio fueron desarrollados con asistencia de Claude (Anthropic):

**Prompt 1 — Script de preprocesamiento:**
> "Necesito un script en Python con OpenCV que tome fotos de una hoja blanca con objetos encima, detecte el contorno de la hoja para recortarla, convierta la imagen a blanco y negro, la redimensione a 128x128 pixeles y guarde el resultado en una carpeta llamada procesadas."

**Prompt 2 — Ajuste del umbral de binarizacion:**
> "El script esta procesando las imagenes pero el fondo blanco no queda completamente blanco en algunas fotos por diferencias de iluminacion. Necesito que aplique un umbral de binarizacion donde pixeles con valor mayor o igual a 200 se consideren blanco."

**Prompt 3 — Script de vectorizacion:**
> "Necesito un script en Python con OpenCV que lea imagenes de dos carpetas conArroz y sinArroz, aplique binarizacion, aplane cada imagen de 128x128 a un vector de 16384 valores, agregue una etiqueta al final y guarde todo en dataset.csv."

**Prompt 4 — Script de entrenamiento:**
> "Necesito un script que cargue multiples CSV de datasets, entrene 4 modelos clasicos (KNN, Decision Tree, Naive Bayes, SVM) con GridSearchCV, muestre un resumen de metricas ordenado por F1-score y exporte el mejor modelo en formato joblib."