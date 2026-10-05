# -*- coding: utf-8 -*-
"""
CONTROL DE LECTURA 3 - Verificación numérica de gradientes mediante diferencias finitas.
Subtarea s1: script único de evidencia numérica (sin gráficos).

Datos del problema (literales del enunciado):
    f(w)  = w**3 - 2*w
    gA(w) = 3*w**2 - 2      (candidato A a la derivada)
    gB(w) = 3*w - 2         (candidato B a la derivada)
    punto de evaluación w = 2.0
    pasos h = 1e-1, 1e-5, 1e-16 (en ese orden)

Salidas:
    * tabla legible en stdout con toda la evidencia de los tres casos
      (se conservan los tres, incluido el caso degenerado h = 1e-16);
    * resultados_control_3.csv (UTF-8, separador coma);
    * resultados.json (todos los números calculados, claves descriptivas, floats planos).

Aritmética: float de Python, sin redondear cálculos intermedios.
"""

import csv
import json

# ---------------------------------------------------------------------------
# 1) Función objetivo y candidatos (definiciones literales del enunciado)
# ---------------------------------------------------------------------------

def f(w):
    """Función objetivo f(w) = w**3 - 2*w."""
    return w**3 - 2*w


def gA(w):
    """Candidato A a la derivada: gA(w) = 3*w**2 - 2."""
    return 3*w**2 - 2


def gB(w):
    """Candidato B a la derivada: gB(w) = 3*w - 2."""
    return 3*w - 2


# ---------------------------------------------------------------------------
# 2) Función a completar: diferencia central.
#    SOLO el método central: (f(w + h) - f(w - h)) / (2 * h).
#    Dentro de esta función NO se usa la derivada analítica ni los candidatos.
# ---------------------------------------------------------------------------

def diferencia_central(f, w, h):
    """Aproximación de la derivada por diferencia central."""
    return (f(w + h) - f(w - h)) / (2 * h)


# ---------------------------------------------------------------------------
# 3) Ejecución de los tres casos (float de Python, sin redondeos)
# ---------------------------------------------------------------------------

W = 2.0
PASOS = [1e-1, 1e-5, 1e-16]

casos = []
for h in PASOS:
    aproximacion = diferencia_central(f, W, h)          # float, sin redondear
    valor_gA = gA(W)                                    # candidato A en w
    valor_gB = gB(W)                                    # candidato B en w
    disc_gA = abs(aproximacion - valor_gA)              # discrepancia absoluta vs A
    disc_gB = abs(aproximacion - valor_gB)              # discrepancia absoluta vs B
    w_mas_h = W + h                                     # punto perturbado superior
    w_menos_h = W - h                                   # punto perturbado inferior
    casos.append({
        "h": float(h),
        "aproximacion": float(aproximacion),
        "gA": float(valor_gA),
        "gB": float(valor_gB),
        "disc_gA": float(disc_gA),
        "disc_gB": float(disc_gB),
        "w_mas_h": float(w_mas_h),
        "w_menos_h": float(w_menos_h),
        "igualdad_w_mas_h": bool(w_mas_h == W),
        "igualdad_w_menos_h": bool(w_menos_h == W),
    })

# ---------------------------------------------------------------------------
# 4) Tabla legible en stdout (evidencia completa de los tres casos)
# ---------------------------------------------------------------------------

SEP = "=" * 79
print(SEP)
print("CONTROL DE LECTURA 3 - Verificación numérica de gradientes")
print("Método central: diferencia_central(f, w, h) = (f(w+h) - f(w-h)) / (2*h)")
print("f(w) = w**3 - 2*w   |   gA(w) = 3*w**2 - 2   |   gB(w) = 3*w - 2")
print("w = 2.0   |   h = 1e-1, 1e-5, 1e-16   |   float de Python, sin redondeos")
print(SEP)

for i, c in enumerate(casos, start=1):
    print()
    print("Caso %d de %d:  h = %r" % (i, len(casos), c["h"]))
    print("-" * 79)
    print("  aproximacion = diferencia_central(f, 2.0, h) : %r" % c["aproximacion"])
    print("  gA(2.0) = 3*(2.0)**2 - 2                     : %r" % c["gA"])
    print("  gB(2.0) = 3*(2.0) - 2                        : %r" % c["gB"])
    print("  disc_gA = abs(aproximacion - gA(2.0))        : %r" % c["disc_gA"])
    print("  disc_gB = abs(aproximacion - gB(2.0))        : %r" % c["disc_gB"])
    print("  punto perturbado w + h   (repr exacto)       : %r" % c["w_mas_h"])
    print("  punto perturbado w - h   (repr exacto)       : %r" % c["w_menos_h"])
    print("  igualdad real (w + h == w)                   : %r" % c["igualdad_w_mas_h"])
    print("  igualdad real (w - h == w)                   : %r" % c["igualdad_w_menos_h"])

print()
print(SEP)
print("Resumen compacto (los tres casos se conservan como evidencia):")
print("%-9s %-24s %-24s %-24s %-8s %-8s" % (
    "h", "aproximacion", "disc_gA", "disc_gB", "w+h==w", "w-h==w"))
for c in casos:
    print("%-9r %-24r %-24r %-24r %-8r %-8r" % (
        c["h"], c["aproximacion"], c["disc_gA"], c["disc_gB"],
        c["igualdad_w_mas_h"], c["igualdad_w_menos_h"]))
print(SEP)

# ---------------------------------------------------------------------------
# 5) Guardar resultados_control_3.csv (UTF-8, separador coma)
# ---------------------------------------------------------------------------

COLUMNAS = ["h", "aproximacion", "gA", "gB", "disc_gA", "disc_gB",
            "w_mas_h", "w_menos_h", "igualdad_w_mas_h", "igualdad_w_menos_h"]

with open("resultados_control_3.csv", "w", newline="", encoding="utf-8") as fh:
    escritor = csv.writer(fh, delimiter=",")
    escritor.writerow(COLUMNAS)
    for c in casos:
        escritor.writerow([c[col] for col in COLUMNAS])

print()
print("Archivo escrito: resultados_control_3.csv (UTF-8, separador coma)")

# ---------------------------------------------------------------------------
# 6) Guardar resultados.json (todos los números, floats planos, indent=2)
# ---------------------------------------------------------------------------

resultados = {
    "control": "Control de Lectura 3 - verificación numérica de gradientes",
    "funcion_f": "w**3 - 2*w",
    "candidato_gA": "3*w**2 - 2",
    "candidato_gB": "3*w - 2",
    "metodo": "diferencia central: (f(w+h) - f(w-h)) / (2*h)",
    "w": 2.0,
    "pasos_h": [c["h"] for c in casos],
    "casos": casos,
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)

print("Archivo escrito: resultados.json")
