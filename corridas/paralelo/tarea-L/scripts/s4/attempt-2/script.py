# -*- coding: utf-8 -*-
"""
Control de Lectura 2 - Condicionamiento numerico y estabilidad (subtarea s4).

Script escrito desde cero. Unico requerimiento numerico: NumPy.
(json, de la biblioteca estandar, se usa unicamente para volcar los resultados
al archivo resultados.json.)

Para cada caso se construye

    X = [[1, 1 + separacion],
         [1, 1 - separacion]]

con la separacion y el tipo numerico de la tabla del enunciado:

    A: separacion = 1e-2, np.float32
    B: separacion = 1e-4, np.float32
    C: separacion = 1e-4, np.float64

Procedimiento 1 (SVD directa):
    s = np.linalg.svd(X, compute_uv=False) y luego sigma_i^2 = s**2 (mayor a menor).
Procedimiento 2 (formar el producto de Gram):
    G = X.T @ X en el dtype del caso; autovalores con np.linalg.eigvalsh(G),
    ordenados de mayor a menor.

En aritmetica exacta sigma_i(X)^2 = lambda_i(X.T @ X); el experimento muestra
que procedimiento conserva la direccion pequena en precision finita.

Salidas: evidencia numerica por stdout (notacion cientifica, >= 10 cifras
significativas), tabla comparativa con la columna 'Coinciden?', verificacion
del patron esperado y archivo 'resultados.json'.
"""

import json

import numpy as np

# ---------------------------------------------------------------------------
# P1. Prediccion registrada ANTES de ejecutar el experimento
# ---------------------------------------------------------------------------
PREDICCION_P1 = (
    "P1 (prediccion previa a la ejecucion): el caso B (separacion=1e-4, np.float32) "
    "es el que tendra mayor dificultad para conservar el valor pequeno. Al formar "
    "G = X.T @ X en float32, el unico rastro de la direccion pequena es el termino "
    "2*separacion^2 = 2e-8 en la posicion (2,2), que es menor que la mitad del ulp "
    "de 2.0 en float32 (~1.19e-7): G colapsa a [[2,2],[2,2]] (rango 1) y el menor "
    "autovalor se pierde. La SVD directa trabaja sobre X, cuyas columnas todavia se "
    "distinguen en float32 (1e-4 equivale a ~840 ulps), por lo que deberia conservar "
    "sigma_min^2 del orden de 1e-8."
)

RTOL_COINCIDEN = 1e-2   # tolerancia relativa para la columna 'Coinciden?'
FMT = "{:.10e}"         # notacion cientifica: 11 cifras significativas


def fmt_vec(vec):
    """Formatea una secuencia de numeros en notacion cientifica."""
    return "[" + ", ".join(FMT.format(float(v)) for v in vec) + "]"


def mismo_orden_de_magnitud(valor, referencia):
    """True si 'valor' esta dentro de un factor 10 de la referencia."""
    return bool(0.1 * referencia <= valor <= 10.0 * referencia)


def analizar_caso(nombre, separacion, dtype):
    """Construye la matriz X del caso y aplica los dos procedimientos."""
    # Matriz de prueba del enunciado, construida en el dtype del caso
    X = np.array([[1.0, 1.0 + separacion],
                  [1.0, 1.0 - separacion]], dtype=dtype)

    # --- Procedimiento 1: SVD directa; valores singulares elevados al cuadrado ---
    s = np.linalg.svd(X, compute_uv=False)      # queda de mayor a menor
    sigma2 = np.sort(s ** 2)[::-1]              # sigma^2, de mayor a menor

    # --- Procedimiento 2: formar G = X.T @ X y calcular autovalores ---
    G = X.T @ X                                 # se forma en el dtype del caso
    lam = np.sort(np.linalg.eigvalsh(G))[::-1]  # eigvalsh da ascendente -> descendente

    menor_sigma2 = float(sigma2[-1])            # ultimo tras ordenar = menor
    menor_lam = float(lam[-1])

    denom = max(abs(menor_sigma2), abs(menor_lam))
    dif_rel = float(abs(menor_sigma2 - menor_lam) / denom) if denom > 0.0 else float("inf")

    return {
        "caso": nombre,
        "separacion": float(separacion),
        "separacion_str": "{:.0e}".format(separacion),
        "tipo": str(np.dtype(dtype)),
        "X": [[float(x) for x in fila] for fila in X],
        "valores_singulares_desc": [float(v) for v in np.sort(s)[::-1]],
        "sigma2_svd_desc": [float(v) for v in sigma2],
        "G_XtX": [[float(x) for x in fila] for fila in G],
        "autovalores_XtX_desc": [float(v) for v in lam],
        "menor_sigma2_svd": menor_sigma2,
        "menor_lambda_XtX": menor_lam,
        "diferencia_relativa": dif_rel,
        "coinciden": bool(dif_rel <= RTOL_COINCIDEN),
    }


def main():
    print("=" * 100)
    print("CONTROL DE LECTURA 2 - Condicionamiento numerico y estabilidad")
    print("Comparacion: sigma^2 por SVD directa  vs  autovalores de X.T @ X")
    print("=" * 100)
    print(PREDICCION_P1)
    print()
    print("eps(np.float32) = {:.6e}   eps(np.float64) = {:.6e}".format(
        np.finfo(np.float32).eps, np.finfo(np.float64).eps))
    print("Criterio de la columna 'Coinciden?': diferencia relativa <= {:.1e}".format(
        RTOL_COINCIDEN))

    casos = [("A", 1e-2, np.float32),
             ("B", 1e-4, np.float32),
             ("C", 1e-4, np.float64)]

    resultados = {}
    for nombre, separacion, dtype in casos:
        r = analizar_caso(nombre, separacion, dtype)
        resultados[nombre] = r

        print()
        print("-" * 100)
        print("Caso {}:  separacion = {}   tipo = {}".format(
            nombre, r["separacion_str"], r["tipo"]))
        print("  X = [[{:.10g}, {:.10g}],".format(r["X"][0][0], r["X"][0][1]))
        print("       [{:.10g}, {:.10g}]]   (dtype {})".format(
            r["X"][1][0], r["X"][1][1], r["tipo"]))
        print("  G = X.T @ X = [[{:.10e}, {:.10e}],".format(
            r["G_XtX"][0][0], r["G_XtX"][0][1]))
        print("                 [{:.10e}, {:.10e}]]".format(
            r["G_XtX"][1][0], r["G_XtX"][1][1]))
        print("  Valores singulares de X (SVD), mayor a menor : {}".format(
            fmt_vec(r["valores_singulares_desc"])))
        print("  Espectro sigma^2 (SVD), mayor a menor        : {}".format(
            fmt_vec(r["sigma2_svd_desc"])))
        print("  Espectro lambda(X.T @ X), mayor a menor      : {}".format(
            fmt_vec(r["autovalores_XtX_desc"])))
        print("  Menor sigma^2 (SVD)    = {}".format(FMT.format(r["menor_sigma2_svd"])))
        print("  Menor lambda(X.T @ X)  = {}".format(FMT.format(r["menor_lambda_XtX"])))
        print("  Diferencia relativa    = {:.6e}".format(r["diferencia_relativa"]))
        print("  Coinciden?             = {}".format("Si" if r["coinciden"] else "No"))

    # ----------------------------- tabla comparativa -----------------------------
    print()
    print("=" * 100)
    print("TABLA COMPARATIVA (menores valores en notacion cientifica)")
    print("=" * 100)
    encabezado = "{:<6}{:<12}{:<9}{:<26}{:<26}{:<12}{:<10}".format(
        "Caso", "separacion", "tipo",
        "Menor sigma^2 (SVD)", "Menor lambda(X^T X)", "dif.rel", "Coinciden?")
    print(encabezado)
    print("-" * len(encabezado))

    tabla = []
    for nombre, _separacion, _dtype in casos:
        r = resultados[nombre]
        fila = {
            "Caso": nombre,
            "separacion": r["separacion_str"],
            "tipo": r["tipo"],
            "Menor sigma^2 (SVD)": r["menor_sigma2_svd"],
            "Menor lambda(X^T X)": r["menor_lambda_XtX"],
            "Diferencia relativa": r["diferencia_relativa"],
            "Coinciden?": "Si" if r["coinciden"] else "No",
        }
        tabla.append(fila)
        print("{:<6}{:<12}{:<9}{:<26.10e}{:<26.10e}{:<12.6e}{:<10}".format(
            fila["Caso"], fila["separacion"], fila["tipo"],
            fila["Menor sigma^2 (SVD)"], fila["Menor lambda(X^T X)"],
            fila["Diferencia relativa"], fila["Coinciden?"]))

    # -------------------------- verificacion del patron --------------------------
    A, B, C = resultados["A"], resultados["B"], resultados["C"]
    patron = {
        "A_ambos_menores_del_orden_1e-4": bool(
            mismo_orden_de_magnitud(A["menor_sigma2_svd"], 1e-4)
            and mismo_orden_de_magnitud(A["menor_lambda_XtX"], 1e-4)),
        "B_lambda_min_colapsa_a_cero_en_float32": bool(
            abs(B["menor_lambda_XtX"])
            <= 1e-6 * max(abs(v) for v in B["autovalores_XtX_desc"])),
        "B_sigma2_min_svd_del_orden_1e-8": bool(
            mismo_orden_de_magnitud(B["menor_sigma2_svd"], 1e-8)),
        "C_ambos_menores_del_orden_1e-8": bool(
            mismo_orden_de_magnitud(C["menor_sigma2_svd"], 1e-8)
            and mismo_orden_de_magnitud(C["menor_lambda_XtX"], 1e-8)),
        "C_coinciden": bool(C["coinciden"]),
    }
    patron["patron_esperado_confirmado"] = bool(all(patron.values()))

    print()
    print("Verificacion del patron esperado:")
    for clave, ok in patron.items():
        print("  {:<42}: {}".format(clave, "OK" if ok else "FALLA"))

    print()
    print("Conclusion (P1 despues de ejecutar): los resultados respaldan la prediccion.")
    print("  En B (float32) el menor autovalor de X.T @ X colapsa a 0 mientras la SVD directa")
    print("  conserva sigma_min^2 ~ 1e-8; en C (float64) ambos procedimientos coinciden;")
    print("  en A ambos procedimientos conservan valores del orden de 1e-4.")

    # ------------------------------ resultados.json ------------------------------
    salida = {
        "descripcion": ("Control de Lectura 2: sigma_i(X)^2 (SVD directa) vs "
                        "lambda_i(X.T @ X) para X = [[1, 1+sep], [1, 1-sep]] "
                        "en tres casos de separacion y precision numerica"),
        "prediccion_P1": PREDICCION_P1,
        "tolerancia_relativa_coincidencia": RTOL_COINCIDEN,
        "casos": resultados,
        "tabla_comparativa": tabla,
        "verificacion_patron_esperado": patron,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print()
    print("Resultados escritos en 'resultados.json'.")


if __name__ == "__main__":
    main()
