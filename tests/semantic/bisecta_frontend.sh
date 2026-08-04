#!/bin/bash
# bisecta_frontend.sh -- ¿el volcado de core y el resumen que falta son de la lib NUEVA?
#
# A/B contra el UNICO fichero que se carga (lib/, no build/: van en ese orden en
# LD_LIBRARY_PATH y confundirlos ya costo un dia en esta campana). Tres corridas por brazo y
# por test, y se anota LA ULTIMA LINEA del cliente: ahi es donde sale "dumped core", que es
# justo lo que un grep de PASS/FAIL no ve.
set -u
H=/home/student.aau.dk/ll33pq
cd $H/GVirtuS
SAL=$H/bisecta_frontend.txt
: > "$SAL"

corre() {  # $1 = etiqueta del brazo
  for t in graphvis2 d2hpool graphvis; do
    for i in 1 2 3; do
      timeout 300 bash $H/gv_run_bin.sh "/opt/GVirtuS/tests/semantic/$t" >/dev/null 2>&1
      RES=$(grep -E "SUMMARY" /tmp/gv_bin.out | tail -1)
      ULT=$(tail -1 /tmp/gv_bin.out)
      CORE=$(grep -c "dumped core\|Segmentation\|Aborted" /tmp/gv_bin.out)
      echo "$1 | $t #$i | resumen='${RES:-<NINGUNO>}' | core=$CORE | ultima='${ULT:0:60}'" >> "$SAL"
    done
  done
}

echo "== brazo NUEVA (con el arreglo) ==" >> "$SAL"
cp -p $H/respaldo_libs_0804/../GVirtuS/build/libgvirtus-communicators-ucx.so lib/libgvirtus-communicators-ucx.so 2>/dev/null
md5sum lib/libgvirtus-communicators-ucx.so >> "$SAL"
corre NUEVA

echo "== brazo VIEJA (respaldo de antes del arreglo) ==" >> "$SAL"
cp -p $H/respaldo_libs_0804/lib_ucx.so.viejo lib/libgvirtus-communicators-ucx.so
md5sum lib/libgvirtus-communicators-ucx.so >> "$SAL"
corre VIEJA

# Se deja la NUEVA puesta: el brazo viejo es una medida, no un estado en el que quedarse.
cp -p build/libgvirtus-communicators-ucx.so lib/libgvirtus-communicators-ucx.so
echo "== restaurada la NUEVA ==" >> "$SAL"
md5sum lib/libgvirtus-communicators-ucx.so >> "$SAL"
echo "FIN" >> "$SAL"
