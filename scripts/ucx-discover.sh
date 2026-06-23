#!/usr/bin/env bash
# ucx-discover.sh — Print hardware info relevant to GVirtuS UCX transport config.
# Run on the backend node (bare metal or inside the dev container).
# Usage: bash scripts/ucx-discover.sh

set -euo pipefail

section() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"; }
skip()    { printf "  [skipped] %s not found\n" "$1"; }

# ---------------------------------------------------------------------------
section "Network Interfaces (IPv4)"
if command -v ip &>/dev/null; then
    printf "  %-16s %-20s %s\n" "INTERFACE" "IP ADDRESS" "STATE"
    printf "  %-16s %-20s %s\n" "---------" "----------" "-----"
    ip -o -4 addr show 2>/dev/null | awk '{
        split($4, a, "/");
        printf "  %-16s %-20s %s\n", $2, a[1], $NF
    }'
else
    skip "ip"
fi

# ---------------------------------------------------------------------------
section "RDMA Devices → Network Interfaces"
if command -v ibdev2netdev &>/dev/null; then
    ibdev2netdev 2>/dev/null | sed 's/^/  /'
else
    skip "ibdev2netdev (install rdma-core / libibverbs-utils)"
fi

# ---------------------------------------------------------------------------
section "RoCE GID Table (v2 entries)"
if command -v show_gids &>/dev/null; then
    printf "  %-10s %-5s %-5s %-40s %-16s %s\n" "DEV" "PORT" "INDEX" "GID" "IPv4" "VER"
    printf "  %-10s %-5s %-5s %-40s %-16s %s\n" "---" "----" "-----" "---" "----" "---"
    show_gids 2>/dev/null | grep -i "v2" | awk '{
        printf "  %-10s %-5s %-5s %-40s %-16s %s\n", $1, $2, $3, $4, $5, $6
    }'
else
    skip "show_gids (install rdma-core / ibverbs-utils)"
fi

# ---------------------------------------------------------------------------
section "RDMA Link Speed"
if command -v ibstat &>/dev/null; then
    for dev in $(ibstat -l 2>/dev/null); do
        printf "  %s: " "$dev"
        ibstat "$dev" 2>/dev/null | grep -E "Rate|State" | head -4 | tr '\n' ' '
        printf "\n"
    done
else
    skip "ibstat"
fi

# ---------------------------------------------------------------------------
section "UCX Available Transports"
if command -v ucx_info &>/dev/null; then
    printf "  %-16s %s\n" "TRANSPORT" "DEVICE"
    printf "  %-16s %s\n" "---------" "------"
    ucx_info -d 2>/dev/null | grep -E "Transport:|Device:" | paste - - | \
        awk '{
            for(i=1;i<=NF;i++) {
                if($i=="Transport:") t=$(i+1)
                if($i=="Device:") d=$(i+1)
            }
            printf "  %-16s %s\n", t, d
        }'
else
    skip "ucx_info (install ucx or ucx-utils)"
fi

# ---------------------------------------------------------------------------
section "GPU"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | sed 's/^/  /'
else
    skip "nvidia-smi"
fi

# ---------------------------------------------------------------------------
section "nvidia-peermem (GPUDirect RDMA)"
if [ -f /proc/modules ]; then
    if grep -q nvidia_peermem /proc/modules 2>/dev/null; then
        printf "  \033[1;32m✓ loaded\033[0m\n"
    else
        printf "  \033[1;33m✗ not loaded\033[0m (run: sudo modprobe nvidia-peermem)\n"
    fi
else
    printf "  [skipped] /proc/modules not available\n"
fi

# ---------------------------------------------------------------------------
section "Current UCX / GVirtuS Environment"
found=0
for var in $(env | grep -E "^(UCX_|GVIRTUS_)" | sort); do
    printf "  %s\n" "$var"
    found=1
done
if [ "$found" -eq 0 ]; then
    printf "  (none set)\n"
fi

printf "\n"
