# =============================================================================
# entrenar_modelos.py
# Proyecto 1 - IE0435 Inteligencia Artificial
# Estudiante: Sheyla Miller Segura - C34919
#
# PROPOSITO:
#   Carga el dataset propio (procesadas/dataset.csv) y los datasets de
#   companeros (datasets/*.csv), entrena 4 modelos de clasificacion clasica
#   con busqueda de hiperparametros, muestra un resumen de metricas y
#   exporta todos los modelos en formato .joblib.
#
# USO:
#   Ejecutar desde la raiz del proyecto (IE0435/):
#       python src/entrenar_modelos.py
#
# SALIDA:
#   modelos/KNN.joblib
#   modelos/Arbol_Decision.joblib
#   modelos/Naive_Bayes.joblib
#   modelos/SVM_Lineal.joblib
#   C34919_sheyla_miller.joblib  <- mejor modelo, archivo de entrega
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# CONFIGURACION
# Rutas relativas a la raiz del proyecto (IE0435/).
# -----------------------------------------------------------------------------
CSV_PROPIO       = "procesadas/dataset.csv"  # Dataset generado por esta estudiante
CARPETA_EXTRAS   = "datasets"                # Datasets de companeros
TEST_SIZE        = 0.20                      # 20% de los datos para prueba
RANDOM_STATE     = 42                        # Semilla para reproducibilidad
NOMBRE_MODELO    = "C34919_sheyla_miller.joblib"  # Archivo de entrega
N_PIXELES        = 16384                     # 128 x 128 pixeles por imagen


def leer_csv_flexible(archivo):
    """
    Lee un CSV de dataset sin importar si tiene encabezado o no.

    Algunos companeros generaron sus CSV con encabezado (pixel_0, pixel_1...),
    otros sin encabezado. Esta funcion detecta automaticamente el formato
    leyendo la primera celda: si es un numero, no hay encabezado.

    Siempre devuelve un DataFrame estandarizado con columnas p0, p1, ..., label
    o None si el archivo no tiene el formato esperado.
    """
    # Leer solo la primera linea para detectar si hay encabezado
    with open(archivo, 'r') as f:
        primera_celda = f.readline().strip().split(',')[0]

    try:
        float(primera_celda)
        # La primera celda es un numero: no hay encabezado
        df = pd.read_csv(archivo, header=None)
    except ValueError:
        # La primera celda es texto: hay encabezado
        df = pd.read_csv(archivo)

    # Eliminar filas vacias o corruptas
    df = df.dropna()

    # Verificar que tenga el numero correcto de columnas
    if df.shape[1] == N_PIXELES + 1:
        # Formato correcto: 16384 pixeles + 1 etiqueta
        df.columns = [f"p{i}" for i in range(N_PIXELES)] + ["label"]
        return df
    elif df.shape[1] == N_PIXELES:
        # Le falta la columna de etiqueta
        print(f"    Sin columna de etiqueta ({df.shape[1]} cols), se omite")
        return None
    else:
        print(f"    Dimension inesperada {df.shape}, se omite")
        return None


def cargar_todos_los_datasets():
    """
    Carga y combina el dataset propio con los de los companeros.

    El dataset propio es obligatorio. La carpeta de extras es opcional:
    si no existe, se trabaja solo con los datos propios.
    """
    dfs = []

    # Cargar dataset propio
    if os.path.exists(CSV_PROPIO):
        try:
            df = leer_csv_flexible(CSV_PROPIO)
            if df is not None:
                dfs.append(df)
                print(f"  OK  {CSV_PROPIO:45s} -> {df.shape[0]} muestras")
        except Exception as e:
            print(f"  Error al leer {CSV_PROPIO}: {e}")
    else:
        print(f"  Advertencia: no se encontro {CSV_PROPIO}")

    # Cargar datasets de companeros si existe la carpeta
    if os.path.isdir(CARPETA_EXTRAS):
        archivos = sorted(glob.glob(os.path.join(CARPETA_EXTRAS, "*.csv")))
        for archivo in archivos:
            try:
                df = leer_csv_flexible(archivo)
                if df is not None:
                    dfs.append(df)
                    print(f"  OK  {os.path.basename(archivo):45s} -> {df.shape[0]} muestras")
            except Exception as e:
                print(f"  Error al leer {os.path.basename(archivo)}: {e}")
    else:
        print(f"  Nota: no hay carpeta '{CARPETA_EXTRAS}/', usando solo dataset propio")

    if not dfs:
        raise ValueError("No se pudo cargar ningun dataset valido.")

    # Combinar todos los datasets en uno solo
    datos = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combinado: {datos.shape[0]} muestras x {datos.shape[1]} columnas")
    return datos


def preparar_datos(datos):
    """
    Separa las columnas de pixeles (X) de la columna de etiqueta (y)
    y divide en conjuntos de entrenamiento y prueba.

    Se usa stratify=y para garantizar que ambos conjuntos tengan
    la misma proporcion de positivos y negativos.
    """
    X = datos.iloc[:, :-1].values.astype(float)  # Todas las columnas menos la ultima
    y = datos.iloc[:, -1].values.astype(int)      # Ultima columna: etiqueta

    print(f"\n  Positivos (arroz)    : {int(y.sum())}")
    print(f"  Negativos (sin arroz): {int(len(y) - y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # Mantener proporcion de clases en ambos subconjuntos
    )

    print(f"\n  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba       : {len(X_test)} muestras")
    return X_train, X_test, y_train, y_test


def definir_modelos():
    """
    Define los 4 modelos a evaluar con sus respectivas grillas de
    hiperparametros para la busqueda exhaustiva (GridSearchCV).

    KNN y SVM usan StandardScaler porque son sensibles a la escala
    de las variables. Decision Tree y Naive Bayes no lo necesitan.

    La busqueda se hace con validacion cruzada de 5 particiones (cv=5)
    optimizando el F1-score, que balancea precision y recall.
    """
    return {
        "KNN": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),   # Normalizar pixeles
                ("clf", KNeighborsClassifier())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9],              # Numero de vecinos
                "clf__weights":     ["uniform", "distance"],   # Peso de los vecinos
                "clf__metric":      ["euclidean", "manhattan"] # Distancia a usar
            }
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__max_depth":         [3, 5, 10, None],  # Profundidad maxima
                "clf__min_samples_split": [2, 5, 10],        # Minimo de muestras para dividir
                "clf__criterion":         ["gini", "entropy"] # Criterio de division
            }
        },
        "Naive Bayes": {
            "pipeline": Pipeline([
                ("clf", GaussianNB())
            ]),
            "params": {
                # var_smoothing: suavizado para evitar probabilidades cero
                "clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]
            }
        },
        "SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),   # Normalizar pixeles
                ("clf", SVC(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__C":      [0.1, 1, 10],          # Parametro de regularizacion
                "clf__kernel": ["linear", "rbf"],     # Tipo de kernel
                "clf__gamma":  ["scale", "auto"]      # Parametro del kernel rbf
            }
        }
    }


def entrenar_y_evaluar(modelos, X_train, X_test, y_train, y_test):
    """
    Entrena cada modelo usando GridSearchCV con validacion cruzada de 5
    particiones, selecciona los mejores hiperparametros y evalua en el
    conjunto de prueba.

    Metricas calculadas:
    - Accuracy : proporcion de predicciones correctas
    - Precision: de los que predijo como arroz, cuantos realmente lo eran
    - Recall   : de los que eran arroz, cuantos detecto correctamente
    - F1-score : media armonica de precision y recall (metrica principal)
    """
    resultados      = []
    mejores_modelos = {}

    for nombre, config in modelos.items():
        print(f"\n  Entrenando {nombre}...")

        # GridSearchCV prueba todas las combinaciones de hiperparametros
        # con validacion cruzada y selecciona la que maximiza el F1-score
        grid = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=5,           # 5 particiones de validacion cruzada
            scoring="f1",   # Optimizar F1-score
            n_jobs=-1,      # Usar todos los nucleos del procesador
            verbose=0
        )
        grid.fit(X_train, y_train)

        # Guardar el mejor modelo encontrado
        mejor = grid.best_estimator_
        mejores_modelos[nombre] = mejor

        # Evaluar en el conjunto de prueba (datos que el modelo no vio)
        y_pred = mejor.predict(X_test)

        resultados.append({
            "modelo":    nombre,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        })

        print(f"    Mejores hiperparametros: {grid.best_params_}")
        print(f"    F1 en validacion cruzada: {grid.best_score_:.4f}")

    return resultados, mejores_modelos


def mostrar_resumen(resultados):
    """Imprime la tabla de resultados ordenada de mejor a peor por F1-score."""
    df = pd.DataFrame(resultados).sort_values("f1_score", ascending=False).reset_index(drop=True)
    sep = "=" * 68
    print(f"\n{sep}")
    print("RESUMEN DE RESULTADOS")
    print(sep)
    print(df.to_string(index=True, float_format="{:.6f}".format))
    print(sep)
    return df


def exportar_modelos(df_resultados, mejores_modelos):
    """
    Guarda todos los modelos entrenados en la carpeta modelos/ y
    exporta el mejor modelo con el nombre de entrega requerido.
    """
    CARPETA_MODELOS = "modelos"
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    # Nombres de archivo para cada modelo
    nombres_archivo = {
        "KNN":           "KNN.joblib",
        "Decision Tree": "Arbol_Decision.joblib",
        "Naive Bayes":   "Naive_Bayes.joblib",
        "SVM":           "SVM_Lineal.joblib",
    }

    print(f"\n  Guardando modelos en '{CARPETA_MODELOS}/'...")
    for nombre, modelo in mejores_modelos.items():
        nombre_archivo = nombres_archivo.get(nombre, f"{nombre.replace(' ', '_')}.joblib")
        ruta = os.path.join(CARPETA_MODELOS, nombre_archivo)
        joblib.dump(modelo, ruta)
        print(f"    Guardado: {nombre_archivo}")

    # Exportar el mejor modelo con el nombre de entrega
    mejor_nombre = df_resultados.iloc[0]["modelo"]
    joblib.dump(mejores_modelos[mejor_nombre], NOMBRE_MODELO)

    print(f"\n  Mejor modelo    : {mejor_nombre}")
    print(f"  Archivo entrega : {NOMBRE_MODELO}")
    print(f"  F1-score        : {df_resultados.iloc[0]['f1_score']:.6f}")
    print(f"  Accuracy        : {df_resultados.iloc[0]['accuracy']:.6f}")


def main():
    print("=" * 68)
    print("  PROYECTO 1 IE0435 - ENTRENAMIENTO DE MODELOS")
    print("=" * 68)

    print("\nCargando datasets...")
    datos = cargar_todos_los_datasets()

    print("\nDistribucion del dataset:")
    X_train, X_test, y_train, y_test = preparar_datos(datos)

    print("\nEntrenando modelos con GridSearchCV (5-fold)...")
    modelos = definir_modelos()
    resultados, mejores_modelos = entrenar_y_evaluar(modelos, X_train, X_test, y_train, y_test)

    df_resultados = mostrar_resumen(resultados)

    print("\nExportando modelos...")
    exportar_modelos(df_resultados, mejores_modelos)

    print("\nListo.")


if __name__ == "__main__":
    main()