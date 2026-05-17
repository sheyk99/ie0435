import cv2
import numpy as np
import os

# ── CONFIGURACIÓN ──────────────────────────────────────────
CARPETA_ENTRADA = "fotos"        # carpeta donde pones tus fotos
CARPETA_SALIDA  = "procesadas"   # carpeta donde se guardan los resultados
TAMAÑO          = (128, 128)
# ───────────────────────────────────────────────────────────

os.makedirs(CARPETA_SALIDA, exist_ok=True)

def recortar_hoja(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Desenfoque para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Umbral para separar hoja del fondo
    _, umbral = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        print("    No se encontró contorno, se usa la imagen completa")
        return img
    
    # Tomar el contorno más grande (que debería ser la hoja)
    contorno_hoja = max(contornos, key=cv2.contourArea)
    
    # Verificar que el contorno sea suficientemente grande
    area = cv2.contourArea(contorno_hoja)
    if area < img.shape[0] * img.shape[1] * 0.1:
        print("    Contorno muy pequeño, se usa la imagen completa")
        return img
    
    # Obtener el rectángulo del contorno
    x, y, w, h = cv2.boundingRect(contorno_hoja)
    
    # Pequeño margen para no cortar el borde de la hoja
    margen = 10
    x = max(0, x - margen)
    y = max(0, y - margen)
    w = min(img.shape[1] - x, w + 2 * margen)
    h = min(img.shape[0] - y, h + 2 * margen)
    
    return img[y:y+h, x:x+w]

# ── PROCESAMIENTO ──────────────────────────────────────────
extensiones = ('.jpg', '.jpeg', '.png', '.webp')
fotos = [f for f in os.listdir(CARPETA_ENTRADA) if f.lower().endswith(extensiones)]

if not fotos:
    print(" No se encontraron fotos en la carpeta 'fotos'")
else:
    print(f" Se encontraron {len(fotos)} fotos\n")
    
    for i, nombre in enumerate(fotos, 1):
        ruta = os.path.join(CARPETA_ENTRADA, nombre)
        print(f"[{i}/{len(fotos)}] Procesando: {nombre}")
        
        img = cv2.imread(ruta)
        if img is None:
            print("   No se pudo leer la imagen, saltando...")
            continue
        
        # 1. Recortar la hoja
        recortada = recortar_hoja(img)
        
        # 2. Convertir a blanco y negro
        byn = cv2.cvtColor(recortada, cv2.COLOR_BGR2GRAY)
        
        # 3. Redimensionar a 128x128
        final = cv2.resize(byn, TAMAÑO, interpolation=cv2.INTER_AREA)
        
        # Guardar con el mismo nombre
        nombre_salida = os.path.splitext(nombre)[0] + ".png"
        cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre_salida), final)
        print(f"   Guardada como {nombre_salida}")
    
    print(f"\n Listo! {len(fotos)} fotos procesadas en la carpeta '{CARPETA_SALIDA}'")