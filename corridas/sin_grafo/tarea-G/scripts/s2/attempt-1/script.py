# -*- coding: utf-8 -*-
"""
SUBTASK s2 - Verificacion numerica de gradientes mediante diferencias centrales.

f(w)  = w**3 - 2*w
gA(w) = 3*w**2 - 2   (candidato A)
gB(w) = 3*w - 2      (candidato B)

Con w = 2.0 y h en [1e-1, 1e-5, 1e-16], todo en float de Python y sin redondear
calculos (solo se formatea la impresion), se calcula por caso:
aproximacion central, gA(2.0), gB(2.0), discrepancias absolutas |aprox-gA| y
|aprox-gB|, puntos perturbados w+h y w-h, y su igualdad real en float.

Salidas:
  - tabla en stdout con las tres filas
  - resultados_control_3.csv (una fila por h)
  - resultados.json (todos los numeros calculados)
"""

import csv
import json


def f(w):
    """Funcion objetivo: f(w) = w**3 - 2*w."""
    return w**3 - 2*w


def gA(w):
    """Candidato A al gradiente: gA(w) = 3*w**2 - 2."""
    return 3*w**2 - 2


def gB(w):
    """Candidato B al gradiente: gB(w) = 3*w - 2."""
    return 3*w - 2


def diferencia_central(f, w, h):
    """Aproximacion de f'(w) por diferencias centrales:
       (f(w+h) - f(w-h)) / (2*h).
       No usa la derivada analitica ni los candidatos gA/gB."""
    return (f(w + h) - f(w - h)) / (2 * h)


def fmt(v):
    """Formato de impresion/escritura: repr de float (sin redondeo, round-trip)."""
    if isinstance(v, bool):
        return str(v)
    return repr(float(v))


def main():
    w = 2.0
    pasos = [1e-1, 1e-5, 1e-16]

    filas = []
    for h in pasos:
        aprox = diferencia_central(f, w, h)      # float de Python, sin redondear
        valor_gA = gA(w)                          # gA(2.0) = 10.0
        valor_gB = gB(w)                          # gB(2.0) = 4.0
        disc_gA = abs(aprox - valor_gA)
        disc_gB = abs(aprox - valor_gB)
        w_mas_h = w + h
        w_menos_h = w - h
        iguales = (w_mas_h == w_menos_h)          # igualdad real en float
        filas.append({
            "h": h,
            "aproximacion": aprox,
            "gA": valor_gA,
            "gB": valor_gB,
            "disc_abs_gA": disc_gA,
            "disc_abs_gB": disc_gB,
            "w_mas_h": w_mas_h,
            "w_menos_h": w_menos_h,
            "perturbados_iguales": iguales,
        })

    # ---------- tabla en stdout ----------
    columnas = ["h", "aproximacion", "gA", "gB", "disc_abs_gA",
                "disc_abs_gB", "w_mas_h", "w_menos_h", "perturbados_iguales"]
    tabla = [[fmt(fila[c]) for c in columnas] for fila in filas]
    anchos = [max([len(c)] + [len(t[i]) for t in tabla])
              for i, c in enumerate(columnas)]

    def linea(celdas):
        return " | ".join(c.ljust(a) for c, a in zip(celdas, anchos))

    print("Verificacion numerica de gradientes por diferencias centrales")
    print("f(w) = w**3 - 2*w | gA(w) = 3*w**2 - 2 | gB(w) = 3*w - 2")
    print("w = 2.0, float de Python, calculos sin redondear")
    print()
    print(linea(columnas))
    print("-+-".join("-" * a for a in anchos))
    for t in tabla:
        print(linea(t))
    print()

    # ---------- CSV: resultados_control_3.csv ----------
    with open("resultados_control_3.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columnas)
        for fila in filas:
            writer.writerow([fmt(fila[c]) for c in columnas])

    # ---------- JSON: resultados.json ----------
    resultados = {
        "descripcion": "Verificacion numerica de gradientes por diferencias centrales",
        "f": "w**3 - 2*w",
        "gA": "3*w**2 - 2",
        "gB": "3*w - 2",
        "w": w,
        "casos": [
            {
                "h": fila["h"],
                "aproximacion": fila["aproximacion"],
                "gA": fila["gA"],
                "gB": fila["gB"],
                "disc_abs_gA": fila["disc_abs_gA"],
                "disc_abs_gB": fila["disc_abs_gB"],
                "w_mas_h": fila["w_mas_h"],
                "w_menos_h": fila["w_menos_h"],
                "perturbados_iguales": fila["perturbados_iguales"],
            }
            for fila in filas
        ],
    }
    with open("resultados.json", "w") as fh:
        json.dump(resultados, fh, indent=2)

    print("Archivos escritos: resultados_control_3.csv, resultados.json")


if __name__ == "__main__":
    main()
