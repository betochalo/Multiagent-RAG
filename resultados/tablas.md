# Tablas de la Parte 2.b

Generado por `evaluacion/tablas.py` desde `resultados/*/resultados_solver.csv`, `resultados/*/resumen_solver.csv` y `corridas/*/tarea-*/traza.jsonl`.

## Por tarea

| Tarea | Variante | Comprobaciones | Procedencia | Subtareas no resueltas | Intentos de código | Tokens entrada | Tokens salida | Duración (s) | Status |
|---|---|---|---|---|---:|---:|---:|---:|---|
| A | Completo | 8/8 | 39/41 cifras respaldadas (0.95) | — | 3 | 46 392 | 52 965 | 214 | completado |
| B | Completo | 9/9 | sin cifras con dos o más decimales | — | 3 | 52 688 | 75 238 | 308 | completado |
| C | Completo | 8/8 | 23/23 cifras respaldadas (1.00) | — | 3 | 67 705 | 104 381 | 392 | completado |
| L | Completo | 4/7 | sin cifras con dos o más decimales | s3 (skipped) | 3 | 63 218 | 373 574 | 1558 | parcial |
| G | Completo | 8/8 | 9/9 cifras respaldadas (1.00) | — | 2 | 45 864 | 114 514 | 741 | completado |
| A | Sin grafo | 8/8 | 32/32 cifras respaldadas (1.00) | — | 4 | 28 754 | 41 982 | 307 | completado |
| B | Sin grafo | 9/9 | sin cifras con dos o más decimales | — | 3 | 25 054 | 27 622 | 354 | completado |
| C | Sin grafo | 8/8 | 15/15 cifras respaldadas (1.00) | — | 4 | 60 383 | 162 577 | 1753 | completado |
| L | Sin grafo | 2/7 | sin cifras con dos o más decimales | s2 (failed) | 3 | 51 617 | 417 085 | 2438 | parcial |
| G | Sin grafo | 8/8 | 4/4 cifras respaldadas (1.00) | — | 1 | 44 230 | 120 627 | 785 | completado |
| A | Paralelo (ext. A) | 8/8 | 51/57 cifras respaldadas (0.89) | — | 3 | 35 538 | 42 238 | 151 | completado |
| B | Paralelo (ext. A) | 8/9 | 0/1 cifras respaldadas (0.00) | — | 3 | 67 399 | 108 332 | 533 | parcial |
| C | Paralelo (ext. A) | 8/8 | 19/19 cifras respaldadas (1.00) | — | 5 | 80 152 | 171 403 | 2007 | completado |
| L | Paralelo (ext. A) | 5/7 | 26/26 cifras respaldadas (1.00) | — | 2 | 81 883 | 283 249 | 2168 | completado |
| G | Paralelo (ext. A) | 8/8 | 8/8 cifras respaldadas (1.00) | — | 1 | 31 488 | 76 335 | 431 | completado |

| Variante | Comprobaciones | Subtareas no resueltas | Intentos de código | Tokens (entrada + salida) | Duración total (min) |
|---|---|---:|---:|---:|---:|
| Completo | 37/40 | 1 | 14 | 996 539 | 53.6 |
| Sin grafo | 35/40 | 1 | 15 | 979 931 | 93.9 |
| Paralelo (ext. A) | 37/40 | 0 | 14 | 978 017 | 88.2 |

## Tokens por agente (entrada + salida, suma de las 5 tareas)

| Agente | Completo | Sin grafo | Paralelo (ext. A) |
|---|---:|---:|---:|
| indexer | 127 505 (29 llamadas) | — | 113 342 (29 llamadas) |
| indexer_community | 81 062 (39 llamadas) | — | 58 265 (32 llamadas) |
| indexer_course | — | — | — |
| planner | 38 594 (5 llamadas) | 37 857 (5 llamadas) | 48 827 (5 llamadas) |
| programmer | 462 244 (21 llamadas) | 518 174 (24 llamadas) | 434 363 (20 llamadas) |
| critic | 65 104 (10 llamadas) | 59 664 (10 llamadas) | 76 335 (11 llamadas) |
| writer | 222 030 (10 llamadas) | 364 236 (13 llamadas) | 246 885 (11 llamadas) |

## Tokens por agente y tarea (entrada + salida)

**Completo**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 17 691 | 18 549 | — | 3 843 | 22 155 | 10 198 | 26 921 |
| B | 18 579 | 19 157 | — | 3 544 | 34 385 | 17 413 | 34 848 |
| C | 21 338 | 18 464 | — | 9 389 | 66 725 | 26 613 | 29 557 |
| L | 51 537 | 11 557 | — | 11 801 | 303 261 | — | 58 636 |
| G | 18 360 | 13 335 | — | 10 017 | 35 718 | 10 880 | 72 068 |

**Sin grafo**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | — | — | — | 3 225 | 33 256 | 14 818 | 19 437 |
| B | — | — | — | 3 329 | 22 298 | 14 714 | 12 335 |
| C | — | — | — | 6 704 | 164 493 | 25 824 | 25 939 |
| L | — | — | — | 15 099 | 289 599 | — | 164 004 |
| G | — | — | — | 9 500 | 8 528 | 4 308 | 142 521 |

**Paralelo (ext. A)**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 17 840 | 9 365 | — | 4 026 | 22 726 | 14 164 | 9 655 |
| B | 33 027 | 12 328 | — | 4 487 | 35 491 | 17 638 | 72 760 |
| C | 18 455 | 17 185 | — | 7 593 | 156 921 | 25 549 | 25 852 |
| L | 24 715 | 8 674 | — | 17 252 | 203 116 | 12 703 | 98 672 |
| G | 19 305 | 10 713 | — | 15 469 | 16 109 | 6 281 | 39 946 |

