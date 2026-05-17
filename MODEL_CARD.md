# MODEL_CARD.md

**Clasificador de Contaminaciones en Línea de Producción v1.0**  
Proyecto 1 — IE0435 Inteligencia Artificial, I-2026  
Estudiante: Sheyla Miller Segura — C34919

---

## Intended use

**Uso previsto:**  
Clasificación binaria de imágenes de una línea de producción simulada para detectar la presencia de granos de arroz (contaminación).

**Fuera del alcance:**  
- Imágenes de entornos industriales reales.
- Detección de múltiples tipos de contaminantes simultáneamente.
- Imágenes con fondos distintos a superficie blanca.
- Inferencia en tiempo real sobre video.

---

## Data summary

- **Recolección:** Imágenes tomadas manualmente con smartphone sobre hoja blanca.
- **Tamaño:** 150 imágenes de 5 estudiantes (30 por estudiante: 15 positivas + 15 negativas).
- **Variaciones:** Diferentes condiciones de luz, ángulos y cantidades de arroz.
- **Formato final:** Vectores de 16,384 valores binarios (128×128 px) y etiqueta.

---

## Labeling process

- Etiquetado manual por cada estudiante recolector.
- `label = 1`: presencia de granos de arroz en la imagen.
- `label = 0`: ausencia de arroz (puede haber otros objetos como aros o clips).
- Sin revisión de segunda persona; posible sesgo de etiquetado individual.

---

## Metrics

Se evaluaron 4 modelos con validación cruzada de 5 particiones (GridSearchCV) y split 80/20:

| Modelo        | Accuracy | Precision | Recall   | F1-score |
|---------------|----------|-----------|----------|----------|
| Decision Tree | 0.8000   | 0.7368    | 0.9333   | **0.8235** |
| KNN           | 0.7333   | 0.6522    | 1.0000   | 0.7895   |
| SVM           | 0.7667   | 0.7222    | 0.8667   | 0.7879   |
| Naive Bayes   | 0.7333   | 0.7692    | 0.6667   | 0.7143   |

**Modelo seleccionado:** Decision Tree (mayor F1-score: 0.8235)  
**Hiperparámetros óptimos:** `criterion=gini`, `max_depth=10`, `min_samples_split=2`  
**Métrica principal:** F1-score (balance entre precisión y recall en detección de contaminaciones)

---

## Ethical / safety notes

- **Sesgo por iluminación:** El modelo puede fallar bajo condiciones de luz muy diferentes a las de entrenamiento.
- **Sesgo por cámara:** Imágenes tomadas con distintos dispositivos móviles; puede no generalizar a cámaras industriales.
- **Sesgo de fondo:** Entrenado solo sobre fondo blanco; sensible a cambios de superficie.
- **Dataset pequeño:** 150 muestras es insuficiente para aplicaciones críticas de seguridad industrial.

---

## Limitations

- Granos de arroz muy pequeños o fuera de foco pueden no ser detectados.
- Alta dimensionalidad del vector (16,384) puede causar sobreajuste con datasets pequeños.
- No hay data augmentation aplicada.
- El umbral de binarización (200) fue fijado manualmente y puede no ser óptimo para todas las condiciones de luz.
- El modelo no distingue cantidad de arroz, solo presencia/ausencia.

---

## Reproducibility

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Preprocesar imágenes (desde procesador/)
python procesar.py        # Paso 1: recorte y escala de grises
python procesadas/procesar.py  # Paso 2: binarización y CSV

# 3. Entrenar y exportar modelo
python entrenar_modelos.py
```

**Modelo exportado:** `C34919_sheyla_miller.joblib`  
**Hardware usado:** Computadora personal, CPU, sin GPU requerida.  
**SO:** Windows 10/11  
**Python:** 3.13  
**Semilla aleatoria:** `random_state=42`
