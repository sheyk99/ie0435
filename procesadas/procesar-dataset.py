# =============================================================================
# procesar-dataset.py
# Proyecto 1 - IE0435 Inteligencia Artificial
# Estudiante: Sheyla Miller Segura - C34919
#
# PROPOSITO:
#   Lee las imagenes ya clasificadas manualmente en conArroz/ y sinArroz/,
#   las binariza (convierte cada pixel a 0 o 1), aplana cada imagen en un
#   vector de 16384 valores y agrega la etiqueta al final.
#   El resultado se guarda como dataset.csv, listo para entrenar modelos.
#
# USO:
#   Ejecutar desde la carpeta procesadas/:
#       python procesar-dataset.py
#
# SALIDA:
#   dataset.csv con formato:
#   - 16384 columnas de pixeles (0 = objeto, 1 = fondo blanco)
#   - 1 columna final 'label' (1 = arroz, 0 = no arroz)
# =============================================================================

import cv2
import numpy as np
import os

# -----------------------------------------------------------------------------
# CONFIGURACION
# Las carpetas son relativas a donde se corre el script (desde procesadas/).
# -----------------------------------------------------------------------------
CARPETA_CON_ARROZ = "conArroz"     # Imagenes con granos de arroz (label = 1)
CARPETA_SIN_ARROZ = "sinArroz"     # Imagenes sin arroz (label = 0)
ARCHIVO_SALIDA    = "dataset.csv"  # Archivo de salida
DIM               = 128            # Dimension esperada de cada imagen


def imagen_a_vector(ruta, etiqueta):
    """
    Convierte una imagen procesada en un vector binario de 16384 valores
    mas su etiqueta al final.

    Proceso:
    1. Leer la imagen en escala de grises
    2. Redimensionar a 128x128 por si hay variaciones de tamano
    3. Binarizar con metodo Otsu (calcula el umbral optimo automaticamente)
    4. Convertir: 255 (blanco) -> 1 (fondo), 0 (negro) -> 0 (objeto)
    5. Aplanar la matriz 128x128 a un vector de 16384 valores
    6. Agregar la etiqueta al final

    El metodo Otsu es preferible a un umbral fijo porque se adapta
    automaticamente a las variaciones de iluminacion entre fotos.

    Parametros:
        ruta: ruta al archivo de imagen
        etiqueta: 1 si hay arroz, 0 si no hay arroz

    Retorna:
        lista de 16385 valores (16384 pixeles + etiqueta), o None si falla
    """
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"    No se pudo leer: {ruta}")
        return None

    # Asegurar tamano estandar aunque la imagen ya deberia venir procesada
    img = cv2.resize(img, (DIM, DIM))

    # Binarizacion con Otsu: calcula automaticamente el mejor umbral
    # THRESH_BINARY: pixeles por encima del umbral -> 255, resto -> 0
    # THRESH_OTSU: el umbral se determina automaticamente segun el histograma
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convertir de escala 0-255 a binario 0-1
    # 255 (blanco, fondo) -> 1
    # 0   (negro, objeto/arroz) -> 0
    binaria_final = (binaria / 255).astype(int)

    # Aplanar la matriz 128x128 en un vector de 16384 elementos
    vector = binaria_final.flatten().tolist()

    # Agregar la etiqueta como ultima columna
    vector.append(etiqueta)

    return vector


def generar_dataset():
    """
    Funcion principal. Procesa todas las imagenes de ambas carpetas
    y genera el archivo CSV con todos los vectores etiquetados.
    """
    todos_los_vectores = []

    # Procesar cada carpeta con su etiqueta correspondiente
    for carpeta, etiqueta in [(CARPETA_CON_ARROZ, 1), (CARPETA_SIN_ARROZ, 0)]:
        if not os.path.exists(carpeta):
            print(f"Advertencia: no se encontro la carpeta '{carpeta}', saltando...")
            continue

        archivos = [
            f for f in os.listdir(carpeta)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"Procesando {len(archivos)} imagenes en '{carpeta}' (etiqueta={etiqueta})...")

        for nombre in sorted(archivos):
            ruta = os.path.join(carpeta, nombre)
            vector = imagen_a_vector(ruta, etiqueta)
            if vector is not None:
                todos_los_vectores.append(vector)

    if not todos_los_vectores:
        print("Error: no se encontraron imagenes para procesar.")
        return

    # Guardar como CSV con valores enteros (0 y 1, no decimales)
    # fmt='%d' asegura que se guarde "1" en lugar de "1.000000"
    np.savetxt(ARCHIVO_SALIDA, todos_los_vectores, fmt='%d', delimiter=',')

    n_filas = len(todos_los_vectores)
    n_cols  = len(todos_los_vectores[0])
    n_pos   = sum(1 for v in todos_los_vectores if v[-1] == 1)
    n_neg   = sum(1 for v in todos_los_vectores if v[-1] == 0)

    print(f"\nDataset generado: {ARCHIVO_SALIDA}")
    print(f"  Filas (imagenes)  : {n_filas}")
    print(f"  Columnas          : {n_cols} ({n_cols-1} pixeles + 1 etiqueta)")
    print(f"  Con arroz (1)     : {n_pos}")
    print(f"  Sin arroz (0)     : {n_neg}")


if __name__ == "__main__":
    generar_dataset()