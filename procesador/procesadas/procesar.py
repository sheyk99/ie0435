import cv2
import numpy as np
import os

# ── CONFIGURACIÓN ──────────────────────────────────────────
CARPETA_CON_ARROZ  = "conArroz"
CARPETA_SIN_ARROZ  = "sinArroz"
ARCHIVO_SALIDA     = "dataset.csv"
DIM                = 128
# ───────────────────────────────────────────────────────────

def imagen_a_vector(ruta, etiqueta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Aseguramos el tamaño por si acaso
    img = cv2.resize(img, (DIM, DIM))

    # --- MÉTODO OTSU: Calcula el umbral automáticamente ---
    # Esto separa el objeto oscuro del fondo claro de forma inteligente
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convertir a 1s (blanco/fondo) y 0s (objeto/arroz)
    # cv2.threshold devuelve 255 para blanco, lo pasamos a 1
    binaria_final = (binaria / 255).astype(int)
    
    vector = binaria_final.flatten().tolist()
    vector.append(etiqueta) # Columna 16385
    return vector

def procesar():
    todos = []
    for carpeta, etiqueta in [(CARPETA_CON_ARROZ, 1), (CARPETA_SIN_ARROZ, 0)]:
        if not os.path.exists(carpeta): continue
        
        archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Procesando {len(archivos)} imágenes en {carpeta}...")
        
        for nombre in archivos:
            v = imagen_a_vector(os.path.join(carpeta, nombre), etiqueta)
            if v: todos.append(v)

    if todos:
        # Usamos fmt='%d' para que guarde 1 y 0 en lugar de 1.000000
        np.savetxt(ARCHIVO_SALIDA, todos, fmt='%d', delimiter=',')
        print(f"✅ CSV generado con {len(todos)} filas y {len(todos[0])} columnas.")
    else:
        print("❌ No se encontraron imágenes.")

if __name__ == "__main__":
    procesar()