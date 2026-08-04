#!/bin/bash
# muestrea_hilos.sh <segundos> -- estado de CADA hilo del backend durante la ventana de bloqueo.
#
# Contesta la pregunta que ninguna traza de GVirtuS puede contestar: el hilo que se queda dentro
# de ucp_worker_progress, ¿esta DORMIDO en un futex (=cerrojo), dentro de un ioctl (=driver), o
# GIRANDO en espacio de usuario (=spinlock)? Cada respuesta apunta a un arreglo distinto.
#
# Campos: t pid tid estado syscall_nr arg0 arg1 wchan utime stime
DUR=${1:-25}
END=$(( $(date +%s) + DUR ))
PIDS=$(pgrep -f 'gvirtus-backend' | tr '\n' ' ')
echo "# pids=$PIDS dur=$DUR"
while [ "$(date +%s)" -lt "$END" ]; do
  T=$(date +%s.%N)
  for p in $PIDS; do
    [ -d "/proc/$p/task" ] || continue
    for t in /proc/"$p"/task/*; do
      tid=${t##*/}
      read -r _ _ st _ < "$t/stat" 2>/dev/null || continue
      sc=$(cat "$t/syscall" 2>/dev/null)
      nr=${sc%% *}; rest=${sc#* }; a0=${rest%% *}; rest2=${rest#* }; a1=${rest2%% *}
      wc=$(cat "$t/wchan" 2>/dev/null)
      cpu=$(awk '{print $14"/"$15}' "$t/stat" 2>/dev/null)
      echo "$T $p $tid $st $nr $a0 $a1 $wc $cpu"
    done
  done
  sleep 0.2
done
