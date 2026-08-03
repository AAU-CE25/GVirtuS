#!/bin/bash
# Matriz completa de conformidad tras los cambios de A1/I13, I10 e I12 (puntos de
# observacion). Mismos brazos que el 2026-08-03 por la tarde, para comparar fila a fila.
set -u
for t in 1 4 8; do bash ~/conformidad_run.sh semantic $t 18 1; done
for t in 1 4;    do bash ~/conformidad_run.sh ptds     $t 18 1; done
CONF_ABLATE=1 bash ~/conformidad_run.sh semantic 8 18 1
echo "=== MATRIZ COMPLETA ==="
