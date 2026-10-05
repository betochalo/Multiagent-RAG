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

## Tokens por agente (entrada / salida, suma de las 5 tareas)

| Agente | Completo | Sin grafo | Paralelo (ext. A) |
|---|---:|---:|---:|
| indexer | 16 682 / 110 823 (29 llamadas) | — | 16 546 / 96 796 (29 llamadas) |
| indexer_community | 50 506 / 30 556 (39 llamadas) | — | 35 480 / 22 785 (32 llamadas) |
| indexer_course | — | — | — |
| planner | 6 997 / 31 597 (5 llamadas) | 6 997 / 30 860 (5 llamadas) | 6 997 / 41 830 (5 llamadas) |
| programmer | 87 842 / 374 402 (21 llamadas) | 68 233 / 449 941 (24 llamadas) | 84 273 / 350 090 (20 llamadas) |
| critic | 38 638 / 26 466 (10 llamadas) | 38 455 / 21 209 (10 llamadas) | 44 830 / 31 505 (11 llamadas) |
| writer | 75 202 / 146 828 (10 llamadas) | 96 353 / 267 883 (13 llamadas) | 108 334 / 138 551 (11 llamadas) |

## Tokens por agente y tarea (entrada / salida)

**Completo**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3 157 / 14 534 | 12 066 / 6 483 | — | 1 192 / 2 651 | 10 613 / 11 542 | 7 414 / 2 784 | 11 950 / 14 971 |
| B | 3 151 / 15 428 | 12 893 / 6 264 | — | 1 185 / 2 359 | 10 743 / 23 642 | 9 759 / 7 654 | 14 957 / 19 891 |
| C | 3 458 / 17 880 | 10 234 / 8 230 | — | 1 493 / 7 896 | 20 828 / 45 897 | 16 770 / 9 843 | 14 922 / 14 635 |
| L | 4 181 / 47 356 | 7 535 / 4 022 | — | 1 912 / 9 889 | 37 362 / 265 899 | — | 12 228 / 46 408 |
| G | 2 735 / 15 625 | 7 778 / 5 557 | — | 1 215 / 8 802 | 8 296 / 27 422 | 4 695 / 6 185 | 21 145 / 50 923 |

**Sin grafo**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | — | — | — | 1 192 / 2 033 | 10 231 / 23 025 | 9 672 / 5 146 | 7 659 / 11 778 |
| B | — | — | — | 1 185 / 2 144 | 7 191 / 15 107 | 9 016 / 5 698 | 7 662 / 4 673 |
| C | — | — | — | 1 493 / 5 211 | 26 408 / 138 085 | 16 922 / 8 902 | 15 560 / 10 379 |
| L | — | — | — | 1 912 / 13 187 | 22 143 / 267 456 | — | 27 562 / 136 442 |
| G | — | — | — | 1 215 / 8 285 | 2 260 / 6 268 | 2 845 / 1 463 | 37 910 / 104 611 |

**Paralelo (ext. A)**

| Tarea | indexer | indexer_community | indexer_course | planner | programmer | critic | writer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3 157 / 14 683 | 5 692 / 3 673 | — | 1 192 / 2 834 | 10 669 / 12 057 | 8 166 / 5 998 | 6 662 / 2 993 |
| B | 3 755 / 29 272 | 8 280 / 4 048 | — | 1 185 / 3 302 | 10 542 / 24 949 | 9 889 / 7 749 | 33 748 / 39 012 |
| C | 3 458 / 14 997 | 10 679 / 6 506 | — | 1 493 / 6 100 | 33 939 / 122 982 | 16 101 / 9 448 | 14 482 / 11 370 |
| L | 3 441 / 21 274 | 4 696 / 3 978 | — | 1 912 / 15 340 | 25 245 / 177 871 | 6 323 / 6 380 | 40 266 / 58 406 |
| G | 2 735 / 16 570 | 6 133 / 4 580 | — | 1 215 / 14 254 | 3 878 / 12 231 | 4 351 / 1 930 | 13 176 / 26 770 |

