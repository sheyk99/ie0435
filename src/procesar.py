# =============================================================================
# procesar.py
# Proyecto 1 - IE0435 Inteligencia Artificial
# Estudiante: Sheyla Miller Segura - C34919
#
# PROPOSITO:
#   Toma las fotos originales tomadas con el celular, detecta y recorta
#   la hoja blanca que sirve de fondo, convierte cada imagen a escala de
#   grises y la redimensiona a 128x128 pixeles. Las imagenes resultantes
#   se guardan en la carpeta 'procesadas/' listas para clasificarse
#   manualmente en conArroz/ y sinArroz/.
#
# USO:
#   Colocar las fotos en la carpeta fotos-raw/ y ejecutar desde src/:
#       python procesar.py
# =============================================================================

import cv2
import numpy as np
import os

# -----------------------------------------------------------------------------
# CONFIGURACION
# Ajustar estas rutas si se cambia la estructura del proyecto.
# Los paths son relativos a donde se corre el script (desde src/).
# -----------------------------------------------------------------------------
CARPETA_ENTRADA = "../fotos-raw"       # Fotos originales del celular
CARPETA_SALIDA  = "../procesadas"      # Imagenes procesadas (grises, 128x128)
TAMANIO         = (128, 128)           # Dimensiones estandar del proyecto


def recortar_hoja(img):
    """
    Detecta el contorno de la hoja blanca en la imagen y la recorta,
    eliminando el fondo de la mesa u otros elementos no deseados.

    El proceso es:
    1. Convierte a grises y aplica un desenfoque para reducir ruido
    2. Aplica un umbral para separar la hoja (blanca) del fondo
    3. Encuentra el contorno mas grande, que deberia ser la hoja
    4. Recorta la imagen al rectangulo que encierra ese contorno

    Si no encuentra un contorno valido, devuelve la imagen original.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Desenfoque gaussiano para eliminar ruido pequeño antes del umbral
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Umbral fijo: pixeles >= 200 se consideran blancos (la hoja)
    # pixeles < 200 se consideran fondo oscuro (mesa, sombras)
    _, umbral = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # Buscar todos los contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("    Advertencia: no se encontro contorno, usando imagen completa")
        return img

    # Tomar el contorno de mayor area (deberia ser la hoja)
    contorno_hoja = max(contornos, key=cv2.contourArea)

    # Verificar que el contorno sea lo suficientemente grande
    # Se exige que cubra al menos el 10% del area total de la imagen
    area = cv2.contourArea(contorno_hoja)
    area_minima = img.shape[0] * img.shape[1] * 0.1
    if area < area_minima:
        print("    Advertencia: contorno muy pequeno, usando imagen completa")
        return img

    # Obtener el rectangulo que encierra el contorno
    x, y, w, h = cv2.boundingRect(contorno_hoja)

    # Agregar un margen pequeno para no cortar los bordes de la hoja
    margen = 10
    x = max(0, x - margen)
    y = max(0, y - margen)
    w = min(img.shape[1] - x, w + 2 * margen)
    h = min(img.shape[0] - y, h + 2 * margen)

    # Devolver solo la region recortada
    return img[y:y+h, x:x+w]


def procesar_fotos():
    """
    Funcion principal. Recorre todas las fotos en CARPETA_ENTRADA,
    aplica el pipeline de procesamiento y guarda los resultados.
    """
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Buscar todos los archivos de imagen soportados
    extensiones = ('.jpg', '.jpeg', '.png', '.webp')
    fotos = [f for f in os.listdir(CARPETA_ENTRADA) if f.lower().endswith(extensiones)]

    if not fotos:
        print("Error: no se encontraron fotos en la carpeta '" + CARPETA_ENTRADA + "'")
        return

    print(f"Se encontraron {len(fotos)} fotos para procesar\n")

    for i, nombre in enumerate(fotos, 1):
        ruta = os.path.join(CARPETA_ENTRADA, nombre)
        print(f"[{i}/{len(fotos)}] {nombre}")

        # Leer la imagen en color
        img = cv2.imread(ruta)
        if img is None:
            print("    No se pudo leer, saltando...")
            continue

        # Paso 1: detectar y recortar la hoja blanca
        recortada = recortar_hoja(img)

        # Paso 2: convertir a escala de grises (eliminar color)
        gris = cv2.cvtColor(recortada, cv2.COLOR_BGR2GRAY)

        # Paso 3: redimensionar a 128x128 pixeles
        # INTER_AREA es el mejor metodo para reducir tamano de imagen
        final = cv2.resize(gris, TAMANIO, interpolation=cv2.INTER_AREA)

        # Guardar como PNG para evitar perdida de calidad
        nombre_salida = os.path.splitext(nombre)[0] + ".png"
        cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre_salida), final)
        print(f"    Guardada: {nombre_salida}")

    print(f"\nListo. {len(fotos)} fotos procesadas en '{CARPETA_SALIDA}'")
    print("Siguiente paso: clasificar manualmente en procesadas/conArroz/ y procesadas/sinArroz/")


if __name__ == "__main__":
    procesar_fotos()