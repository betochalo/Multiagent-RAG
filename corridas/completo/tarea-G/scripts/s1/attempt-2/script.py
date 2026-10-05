# -*- coding: utf-8 -*-
"""
CONTROL DE LECTURA 3 - Verificacion numerica de gradientes (subtarea s1).

f(w)  = w**3 - 2*w
gA(w) = 3*w**2 - 2   (candidato A)
gB(w) = 3*w - 2      (candidato B)

Se completa unicamente la expresion pendiente de diferencia_central(f, w, h)
con el metodo de diferencia central: (f(w+h) - f(w-h)) / (2*h).
Dentro de esa funcion NO se usa la derivada analitica ni los candidatos gA/gB.

Casos: w = 2.0 con h = 1e-1, 1e-5, 1e-16, con float nativo de Python y sin
redondear calculos intermedios (el formateo se aplica solo al mostrar/guardar).

Salidas:
  * tabla legible en stdout (tres filas, una por cada h)
  * resultados_control_3.csv (una fila por cada h)
  * resultados.json (todos los numeros calculados, como floats nativos)
"""

import csv
import json

# ---------------------------------------------------------------------------
# Tarea 1: prediccion P0 registrada ANTES de ejecutar (se conserva en evidencia)
# ---------------------------------------------------------------------------
PREDICCION_P0 = (
    "P0 (antes de ejecutar): al verificar un gradiente se comparan la "
    "aproximacion numerica por diferencia central con el valor de cada "
    "candidato en w. Un h menor no siempre mejora la aproximacion: reduce el "
    "error de truncamiento (~h^2), pero por debajo de cierto tamano domina el "
    "error de redondeo del float y la aproximacion se degrada o colapsa "
    "(p. ej., cuando w+h o w-h dejan de distinguirse de w)."
)

# ---------------------------------------------------------------------------
# Funciones proporcionadas por el enunciado
# ---------------------------------------------------------------------------
def f(w):
    return w**3 - 2*w


def gA(w):
    return 3*w**2 - 2


def gB(w):
    return 3*w - 2


# ---------------------------------------------------------------------------
# Tarea 2: funcion a completar. UNICAMENTE la expresion pendiente:
# metodo de diferencia central. Sin derivada analitica y sin gA/gB.
# ---------------------------------------------------------------------------
def diferencia_central(f, w, h):
    # Expresion pendiente (metodo de diferencia central):
    return (f(w + h) - f(w - h)) / (2 * h)


# ---------------------------------------------------------------------------
# Tarea 3: ejecucion de los tres casos (float de Python, sin redondear)
# ---------------------------------------------------------------------------
w = 2.0
pasos = [1e-1, 1e-5, 1e-16]

f_w = f(w)     # 4.0
gA_w = gA(w)   # 10.0
gB_w = gB(w)   # 4.0

casos = []
for h in pasos:
    aproximacion = diferencia_central(f, w, h)  # (f(w+h) - f(w-h)) / (2*h)
    disc_gA = abs(aproximacion - gA_w)
    disc_gB = abs(aproximacion - gB_w)
    w_mas_h = w + h        # punto perturbado tal como lo representa el float
    w_menos_h = w - h
    casos.append({
        "h": h,
        "aproximacion": aproximacion,
        "gA": gA_w,
        "gB": gB_w,
        "disc_gA": disc_gA,
        "disc_gB": disc_gB,
        "w_mas_h": w_mas_h,
        "w_menos_h": w_menos_h,
        "igualdad_w_mas_h": bool(w_mas_h == w),
        "igualdad_w_menos_h": bool(w_menos_h == w),
        "igualdad_f_perturbados": bool(f(w_mas_h) == f(w_menos_h)),
    })

# ---------------------------------------------------------------------------
# Tabla legible en stdout (el formateo/repr se aplica solo al mostrar)
# ---------------------------------------------------------------------------
encabezados = ["h", "aproximacion", "gA", "gB", "disc_gA", "disc_gB",
               "w+h (repr)", "w-h (repr)", "w+h==w", "w-h==w",
               "f(w+h)==f(w-h)"]

filas_tabla = []
for c in casos:
    filas_tabla.append([
        repr(c["h"]),
        repr(c["aproximacion"]),
        repr(c["gA"]),
        repr(c["gB"]),
        repr(c["disc_gA"]),
        repr(c["disc_gB"]),
        repr(c["w_mas_h"]),
        repr(c["w_menos_h"]),
        str(c["igualdad_w_mas_h"]),
        str(c["igualdad_w_menos_h"]),
        str(c["igualdad_f_perturbados"]),
    ])

anchos = [max([len(encabezados[i])] + [len(r[i]) for r in filas_tabla])
          for i in range(len(encabezados))]
separador = "-+-".join("-" * a for a in anchos)

print("CONTROL DE LECTURA 3 - Verificacion de gradientes por diferencia central")
print("f(w) = w**3 - 2*w   |   gA(w) = 3*w**2 - 2   |   gB(w) = 3*w - 2")
print("w = %r   ->   f(2.0) = %r, gA(2.0) = %r, gB(2.0) = %r"
      % (w, f_w, gA_w, gB_w))
print()
print(separador)
print(" | ".join(encabezados[i].ljust(anchos[i]) for i in range(len(encabezados))))
print(separador)
for r in filas_tabla:
    print(" | ".join(r[i].ljust(anchos[i]) for i in range(len(encabezados))))
print(separador)

# ---------------------------------------------------------------------------
# DataFrame -> resultados_control_3.csv (una fila por cada h)
# pandas puede no estar disponible en el sandbox; se usa si existe y, si no,
# se escribe un CSV equivalente con el modulo estandar csv.
# ---------------------------------------------------------------------------
columnas_csv = ["h", "aproximacion", "gA", "gB", "disc_gA", "disc_gB",
                "w_mas_h", "w_menos_h", "igualdad_w_mas_h",
                "igualdad_w_menos_h"]

csv_escrito = False
try:
    import pandas as pd
    df = pd.DataFrame([{k: c[k] for k in columnas_csv} for c in casos],
                      columns=columnas_csv)
    df.to_csv("resultados_control_3.csv", index=False)
    csv_escrito = True
except Exception:
    csv_escrito = False

if not csv_escrito:
    with open("resultados_control_3.csv", "w", newline="",
              encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(columnas_csv)
        for c in casos:
            escritor.writerow([
                repr(c["h"]),
                repr(c["aproximacion"]),
                repr(c["gA"]),
                repr(c["gB"]),
                repr(c["disc_gA"]),
                repr(c["disc_gB"]),
                repr(c["w_mas_h"]),
                repr(c["w_menos_h"]),
                str(c["igualdad_w_mas_h"]),
                str(c["igualdad_w_menos_h"]),
            ])

# ---------------------------------------------------------------------------
# resultados.json: todos los numeros calculados, como floats nativos de Python
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "s1_control_lectura_3_gradientes",
    "prediccion_P0": PREDICCION_P0,
    "w": w,
    "f_en_w": f_w,
    "gA_en_w": gA_w,
    "gB_en_w": gB_w,
    "casos": [
        {
            "h": c["h"],
            "aproximacion": c["aproximacion"],
            "gA": c["gA"],
            "gB": c["gB"],
            "disc_gA": c["disc_gA"],
            "disc_gB": c["disc_gB"],
            "w_mas_h": c["w_mas_h"],
            "w_menos_h": c["w_menos_h"],
            "repr_w_mas_h": repr(c["w_mas_h"]),
            "repr_w_menos_h": repr(c["w_menos_h"]),
            "igualdad_w_mas_h": c["igualdad_w_mas_h"],
            "igualdad_w_menos_h": c["igualdad_w_menos_h"],
            "igualdad_f_perturbados": c["igualdad_f_perturbados"],
        }
        for c in casos
    ],
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2)

print()
print("Contenido de resultados.json:")
print(json.dumps(resultados, indent=2))
print()
print("Archivos generados: resultados_control_3.csv, resultados.json")
