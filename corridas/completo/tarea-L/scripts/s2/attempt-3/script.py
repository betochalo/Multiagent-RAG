# Control de Lectura 2 - Condicionamiento numerico y estabilidad (subtarea s2).
#
# Compara, para tres casos, dos procedimientos matematicamente equivalentes
# pero numericamente distintos:
#
#   (1) sigma_i(X)^2 : valores singulares de X via np.linalg.svd(X, compute_uv=False),
#       elevados al cuadrado y ordenados de mayor a menor.
#   (2) lambda_i(X.T @ X) : autovalores de G = X.T @ X (construida respetando el
#       dtype del caso) via np.linalg.eigvalsh, ordenados de mayor a menor.
#
# Casos:
#   A: separacion = 1e-2, dtype = np.float32
#   B: separacion = 1e-4, dtype = np.float32
#   C: separacion = 1e-4, dtype = np.float64
#
# Unico import numerico: NumPy. (json, de la biblioteca estandar, se usa solo
# para escribir el archivo de resultados resultados.json.)

import json

import numpy as np

# Tolerancia declarada para la columna "Coinciden?": los dos procedimientos se
# consideran coincidentes si su diferencia relativa queda por debajo de este
# umbral. En float32 el error absoluto de ambos algoritmos es del orden de
# eps32 * ||G|| ~ 1.2e-7 * 4 ~ 5e-7, lo que frente a un valor pequeno de ~1e-4
# permite diferencias relativas de hasta ~5e-3; el umbral 1e-2 declara
# coincidencia en A y C sin confundirla con el colapso a 0 del caso B.
TOLERANCIA_RELATIVA = 1e-2


def construir_X(separacion, dtype):
    """Construye X = [[1, 1+separacion], [1, 1-separacion]] con el dtype dado."""
    return np.array(
        [[1.0, 1.0 + separacion],
         [1.0, 1.0 - separacion]],
        dtype=dtype,
    )


def fmt(x):
    """Notacion cientifica con 10 decimales (11 cifras significativas)."""
    return np.format_float_scientific(float(x), precision=10, unique=False)


def fmt_vec(v):
    return "[" + ", ".join(fmt(x) for x in v) + "]"


def fmt_mat(M):
    return "[" + ", ".join(fmt_vec(fila) for fila in M) + "]"


def analizar_caso(nombre, separacion, dtype):
    """Ejecuta los dos procedimientos para un caso y devuelve todos los numeros."""
    X = construir_X(separacion, dtype)

    # Procedimiento 1: SVD directa; sigma^2 ordenado de mayor a menor.
    sigma = np.linalg.svd(X, compute_uv=False)
    sigma2_desc = np.sort(sigma ** 2)[::-1].astype(np.float64)

    # Procedimiento 2: G = X.T @ X (mismo dtype del caso); autovalores de
    # mayor a menor (eigvalsh los entrega en orden ascendente).
    G = X.T @ X
    autoval_desc = np.sort(np.linalg.eigvalsh(G))[::-1].astype(np.float64) + 0.0

    men_sigma2 = float(sigma2_desc[-1])
    men_lambda = float(autoval_desc[-1])
    denom = max(abs(men_sigma2), abs(men_lambda))
    dif_rel = abs(men_sigma2 - men_lambda) / denom if denom > 0.0 else 0.0
    coinciden = bool(dif_rel < TOLERANCIA_RELATIVA)

    detalle = {
        "separacion": float(separacion),
        "dtype": str(np.dtype(dtype)),
        "X": [[float(v) for v in fila] for fila in X],
        "G_XtX": [[float(v) for v in fila] for fila in G],
        "valores_singulares_svd": [float(v) for v in sigma],
        "sigma_cuadrado_svd_desc": [float(v) for v in sigma2_desc],
        "autovalores_XtX_desc": [float(v) for v in autoval_desc],
        "menor_sigma_cuadrado_svd": men_sigma2,
        "menor_autovalor_XtX": men_lambda,
        "diferencia_relativa": float(dif_rel),
        "coinciden": coinciden,
    }
    return detalle


def main():
    casos = [
        ("A", 1e-2, np.float32),
        ("B", 1e-4, np.float32),
        ("C", 1e-4, np.float64),
    ]

    resultados = {
        "descripcion": ("Comparacion del menor sigma^2 (SVD directa de X) con el menor "
                        "lambda(X.T @ X); ambas secuencias ordenadas de mayor a menor"),
        "tolerancia_diferencia_relativa": float(TOLERANCIA_RELATIVA),
        "criterio_coincidencia": "coinciden si diferencia_relativa < tolerancia_diferencia_relativa",
        "casos": {},
    }

    filas = []
    for nombre, separacion, dtype in casos:
        detalle = analizar_caso(nombre, separacion, dtype)
        resultados["casos"][nombre] = detalle
        filas.append((nombre, detalle))

    # ------------------------- tabla comparativa -------------------------
    print("=" * 118)
    print("Control de Lectura 2 - Condicionamiento numerico y estabilidad")
    print("Menor sigma^2 (SVD directa) frente a menor lambda(X.T @ X); secuencias ordenadas de mayor a menor")
    print(f"Criterio 'Coinciden?': diferencia relativa < {fmt(TOLERANCIA_RELATIVA)}")
    print("=" * 118)
    cab = (f"{'Caso':<6}{'separacion':>20}{'tipo':>10}"
           f"{'Menor sigma^2 (SVD)':>24}{'Menor lambda(X^T X)':>24}"
           f"{'Dif. relativa':>20}{'Coinciden?':>13}")
    print(cab)
    print("-" * len(cab))
    for nombre, d in filas:
        print(f"{nombre:<6}{d['separacion']:>20.10e}{d['dtype']:>10}"
              f"{fmt(d['menor_sigma_cuadrado_svd']):>24}"
              f"{fmt(d['menor_autovalor_XtX']):>24}"
              f"{fmt(d['diferencia_relativa']):>20}"
              f"{('SI' if d['coinciden'] else 'NO'):>13}")
    print("-" * len(cab))
    print()

    # ------------------------- detalle por caso -------------------------
    for nombre, d in filas:
        print(f"Detalle caso {nombre}: separacion = {fmt(d['separacion'])}, dtype = {d['dtype']}")
        print(f"  X = {fmt_mat(d['X'])}")
        print(f"  G = X.T @ X = {fmt_mat(d['G_XtX'])}")
        print(f"  sigma (SVD), desc       = {fmt_vec(d['valores_singulares_svd'])}")
        print(f"  sigma^2 (SVD), desc     = {fmt_vec(d['sigma_cuadrado_svd_desc'])}")
        print(f"  lambda(X.T @ X), desc   = {fmt_vec(d['autovalores_XtX_desc'])}")
        print(f"  Menor sigma^2 (SVD)     = {fmt(d['menor_sigma_cuadrado_svd'])}")
        print(f"  Menor lambda(X.T @ X)   = {fmt(d['menor_autovalor_XtX'])}")
        print(f"  Diferencia relativa     = {fmt(d['diferencia_relativa'])}")
        print(f"  Coinciden? (tol = {TOLERANCIA_RELATIVA:.1e}) -> {'SI' if d['coinciden'] else 'NO'}")
        print()

    # ------------------------- lectura del patron -------------------------
    print("Lectura del patron esperado:")
    print(" * Caso B (float32): al formar X.T @ X en float32, la entrada 2 + 2e-8 se redondea a 2.0,")
    print("   de modo que G = [[2, 2], [2, 2]] y su menor autovalor colapsa a 0; la SVD directa de X")
    print("   conserva un sigma_min^2 pequeno y positivo (~1e-8).")
    print(" * Caso C (float64): con la misma separacion nominal, ambos procedimientos coinciden en un")
    print("   valor pequeno no nulo (~1e-8): el colapso del caso B es un problema de estabilidad del")
    print("   procedimiento (formar X.T @ X), no del condicionamiento del problema.")
    print()

    # ------------------------- archivo de resultados -------------------------
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    print("Resultados escritos en resultados.json")


if __name__ == "__main__":
    main()
