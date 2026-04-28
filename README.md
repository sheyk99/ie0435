# Procesador de Imágenes — Proyecto 1 IE0435

**Estudiante:** Sheyla Miller Segura — C34919  
**Curso:** IE0435 Inteligencia Artificial aplicada a ingeniería eléctrica — I-2026  
**Universidad de Costa Rica**

---

## Descripción

Este repositorio contiene el pipeline completo de preprocesamiento de imágenes para la detección de contaminaciones (granos de arroz) en una línea de producción simulada. El flujo va desde las fotos crudas tomadas con celular hasta un dataset en formato CSV de vectores binarios, listo para entrenar modelos de clasificación clásica.

---

## Estructura del repositorio

```
procesador/
├── fotos/                  # Fotos originales tomadas con el celular
├── procesadas/
│   ├── conArroz/           # Imágenes procesadas clasificadas manualmente: con arroz (label=1)
│   ├── sinArroz/           # Imágenes procesadas clasificadas manualmente: sin arroz (label=0)
│   ├── procesar.py         # Script 2: genera el dataset CSV desde las carpetas clasificadas
│   └── dataset.csv         # Dataset final de vectores binarios
└── procesar.py             # Script 1: convierte fotos crudas a imágenes procesadas en tamaño y colores adecuados
```

---

## Flujo de trabajo

### Paso 1 — Preprocesamiento de fotos (`procesar.py`)

Las fotos en `fotos/` fueron tomadas con cámara de celular sobre una hoja blanca que simula la banda de producción. El script `procesar.py` las procesa de forma automática:

1. **Detección y recorte de la hoja:** usa detección de contornos (OpenCV) para identificar el borde de la hoja blanca y recortar el fondo sobrante.
2. **Conversión a escala de grises:** elimina la información de color.
3. **Redimensionado a 128×128 píxeles:** estandariza el tamaño usando interpolación `INTER_AREA`.
4. **Guardado:** las imágenes resultantes se guardan automáticamente en `procesadas/` en formato PNG.

```bash
# Desde la carpeta procesador/
python procesar.py
```

> **Nota sobre el umbral de blanco:** durante el desarrollo se identificó que las fotos tomadas en diferentes condiciones de iluminación generaban variaciones en el nivel de blanco del fondo. Se ajustó el umbral de binarización a **200/255** (píxeles ≥ 200 se consideran fondo blanco → 1, el resto son objetos → 0) para lograr una separación consistente entre el fondo y los objetos en todas las imágenes del conjunto.

---

### Paso 2 — Clasificación manual

Una vez procesadas, las imágenes se clasificaron **manualmente** según los requerimientos del proyecto:

- `procesadas/conArroz/` → imágenes que contienen al menos un grano de arroz (**etiqueta = 1**)
- `procesadas/sinArroz/` → imágenes sin granos de arroz, pueden contener otros objetos como aros o clips, o estar vacías (**etiqueta = 0**)

El criterio de clasificación es estrictamente la **presencia o ausencia de granos de arroz**, independientemente de otros objetos en la imagen.

---

### Paso 3 — Generación del dataset (`procesadas/procesar.py`)

Con las imágenes ya clasificadas en sus carpetas, se corre el segundo script:

```bash
# Desde la carpeta procesadas/
python procesar.py
```

Este script:
1. Lee cada imagen en escala de grises.
2. Aplica binarización con umbral 200: píxeles ≥ 200 → `1` (fondo blanco), resto → `0` (objeto).
3. Aplana la matriz 128×128 a un **vector fila de 16,384 valores**.
4. Agrega la etiqueta al final del vector (`1` = arroz, `0` = no arroz).
5. Guarda todo en `dataset.csv` — una fila por imagen, **16,385 columnas** en total.

---

## Dependencias

```bash
pip install opencv-python numpy
```

| Librería | Versión mínima | Uso |
|---|---|---|
| `opencv-python` | 4.8+ | Lectura, recorte, escala de grises, redimensionado y binarización de imágenes |
| `numpy` | 1.24+ | Manejo de matrices y vectores |

---

## Prompts de IA utilizados

Los scripts de este repositorio fueron generados con asistencia de Claude (Anthropic). A continuación se listan los prompts principales utilizados:

**Prompt 1 — Script de preprocesamiento (`procesar.py`):**
> "Necesito un script en Python con OpenCV que tome fotos de una hoja blanca con objetos encima, detecte el contorno de la hoja para recortarla, convierta la imagen a blanco y negro, la redimensione a 128x128 píxeles y guarde el resultado en una carpeta llamada 'procesadas'. Las fotos están en una carpeta llamada 'fotos'."

**Prompt 2 — Ajuste del umbral de binarización:**
> "El script está procesando las imágenes pero el fondo blanco no queda completamente blanco en algunas fotos por diferencias de iluminación. Necesito que aplique un umbral de binarización donde píxeles con valor mayor o igual a 200 se consideren blanco y el resto negro, para que el fondo quede uniforme."

**Prompt 3 — Script de vectorización y dataset (`procesadas/procesar.py`):**
> "Necesito un script en Python con OpenCV que lea imágenes de dos carpetas: 'conArroz' y 'sinArroz', aplique un umbral de binarización (pixel >= 200 → 1, resto → 0), aplane cada imagen de 128x128 a un vector fila de 16384 valores, agregue una etiqueta al final (1 para conArroz, 0 para sinArroz) y guarde todo en un archivo dataset.csv donde cada fila sea una imagen."

**Prompt 4 — Ayuda para organizar un README formal:**
> "Ayúdeme a ordenar mi idea del script que tengo, añada los comandos de bash para correr scripts y dependencias, haga también una estructura de la carpeta del repositorio"

---
