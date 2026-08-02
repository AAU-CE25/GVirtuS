#!/bin/bash
# Captura de entorno para la campana ASPLOS. No modifica nada.
echo "=== HOST ==="
hostname; date -Is; uptime
echo "=== OS ==="
grep PRETTY /etc/os-release; uname -r
echo "=== CPU ==="
lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA node\(s\))"
echo "--- governor ---"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
grep "cpu MHz" /proc/cpuinfo | head -4
echo "=== MEM ==="
free -g | head -2
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,persistence_mode,compute_mode --format=csv
echo "--- MPS? ---"
pgrep -a nvidia-cuda-mps 2>/dev/null || echo "no MPS daemon"
echo "=== RDMA ==="
ibv_devinfo 2>/dev/null | grep -E "hca_id|fw_ver|state:|link_layer" | head -20
ibstat 2>/dev/null | grep -E "^CA|Rate|State" | head -12
echo "--- ofed ---"
ofed_info -s 2>/dev/null || echo "ofed_info n/a"
echo "=== NET ==="
ip -4 addr show ens1f1np1 2>/dev/null | grep -E "inet |mtu"
echo "=== TOPO ==="
nvidia-smi topo -m 2>/dev/null | head -12
echo "=== HOST TOOLCHAIN ==="
gcc --version|head -1; g++ --version|head -1; cmake --version|head -1; git --version
nvcc --version 2>/dev/null|tail -2 || echo "no host nvcc"
echo "=== UCX (host) ==="
ucx_info -v 2>/dev/null || echo "no host ucx_info"
echo "=== DOCKER ==="
docker --version
docker ps --format "{{.Names}}|{{.Image}}|{{.Status}}"
echo "=== APPORT/CORE ==="
cat /proc/sys/kernel/core_pattern
ulimit -c
echo "=== DISK ==="
df -h / /var /home 2>/dev/null | grep -v tmpfs
