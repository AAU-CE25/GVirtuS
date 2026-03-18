#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# check_rdma.sh
#
# Validates that the BlueField-3 RDMA device is present, visible to libibverbs,
# and that the correct GID index (RoCEv2) is available for the RDMA demo.
#
# Usage:
#   chmod +x check_rdma.sh
#   ./check_rdma.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

PASS="${GREEN}[PASS]${NC}"; FAIL="${RED}[FAIL]${NC}"
INFO="${BLUE}[INFO]${NC}"; WARN="${YELLOW}[WARN]${NC}"
ERRORS=0

pass()  { echo -e "${PASS} $*"; }
fail()  { echo -e "${FAIL} $*"; ((ERRORS++)) || true; }
info()  { echo -e "${INFO} $*"; }
warn()  { echo -e "${WARN} $*"; }
header(){ echo -e "\n${BLUE}══ $* ══${NC}"; }

BF3_PCI="82:00"          # PCI slot of the BlueField-3
EXPECTED_DEV="mlx5_2"    # expected mlx5 name for BF3 — verify with check below
GID_INDEX=3              # RoCEv2 GID index used by rdma_common.h
IB_PORT=1

echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║       BF3 RDMA Device Validation         ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────────────────────
header "1. BlueField-3 visible on PCIe bus"
# ─────────────────────────────────────────────────────────────────────────────
if lspci | grep -qi "bluefield"; then
    BF3_LINE=$(lspci | grep -i bluefield | head -1)
    pass "BlueField-3 found: $BF3_LINE"
else
    fail "BlueField-3 NOT found in lspci output"
    warn "Check physical installation and PCIe slot"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
header "2. libibverbs installed"
# ─────────────────────────────────────────────────────────────────────────────
if command -v ibv_devinfo &>/dev/null; then
    pass "ibv_devinfo found (libibverbs is installed)"
else
    fail "ibv_devinfo not found"
    warn "Fix: sudo apt install ibverbs-utils infiniband-diags rdma-core"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
header "3. RDMA devices enumerated by libibverbs"
# ─────────────────────────────────────────────────────────────────────────────
NUM_DEVS=$(ibv_devinfo 2>/dev/null | grep -c "hca_id:" || true)
if [[ "$NUM_DEVS" -gt 0 ]]; then
    pass "$NUM_DEVS RDMA device(s) found"
    ibv_devinfo 2>/dev/null | grep "hca_id:" | while read -r _ dev; do
        info "  Device: $dev"
    done
else
    fail "No RDMA devices found by ibv_devinfo"
    warn "Is the RDMA stack loaded? Try: sudo modprobe mlx5_ib"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "4. Map mlx5 device names to PCI slots"
# ─────────────────────────────────────────────────────────────────────────────
info "Checking which mlx5_X is the BF3 (${BF3_PCI}):"
FOUND_DEV=""
for d in /sys/class/infiniband/mlx5_*/; do
    DEV=$(basename "$d")
    PCI=$(cat "$d/device/uevent" 2>/dev/null | grep PCI_SLOT | cut -d= -f2 || true)
    if [[ -n "$PCI" ]]; then
        echo "    $DEV  →  PCI $PCI"
        if [[ "$PCI" == *"${BF3_PCI}"* ]]; then
            FOUND_DEV="$DEV"
        fi
    fi
done

if [[ -n "$FOUND_DEV" ]]; then
    pass "BF3 at ${BF3_PCI} maps to RDMA device: $FOUND_DEV"
    if [[ "$FOUND_DEV" != "$EXPECTED_DEV" ]]; then
        warn "Expected '$EXPECTED_DEV' but found '$FOUND_DEV'"
        warn "Update DEVICE_NAME in rdma_common.h to: \"$FOUND_DEV\""
    else
        pass "DEVICE_NAME = \"$EXPECTED_DEV\" in rdma_common.h is correct"
    fi
else
    fail "Could not find BF3 (${BF3_PCI}) in /sys/class/infiniband/"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "5. BF3 port state"
# ─────────────────────────────────────────────────────────────────────────────
DEV="${FOUND_DEV:-$EXPECTED_DEV}"
PORT_STATE=$(ibv_devinfo -d "$DEV" 2>/dev/null | grep "state:" | head -1 | awk '{print $2}' || true)
if [[ "$PORT_STATE" == "PORT_ACTIVE" ]]; then
    pass "Port $IB_PORT state: PORT_ACTIVE"
else
    fail "Port $IB_PORT state: ${PORT_STATE:-unknown} (expected PORT_ACTIVE)"
    warn "Is the cable connected and the remote port UP?"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "6. RoCEv2 GID (index $GID_INDEX)"
# ─────────────────────────────────────────────────────────────────────────────
GID_TYPE=$(cat /sys/class/infiniband/"$DEV"/ports/${IB_PORT}/gid_attrs/types/${GID_INDEX} \
           2>/dev/null || true)
GID_VAL=$(cat /sys/class/infiniband/"$DEV"/ports/${IB_PORT}/gids/${GID_INDEX} \
          2>/dev/null || true)

if [[ "$GID_TYPE" == *"RoCE v2"* ]] || [[ "$GID_TYPE" == *"roce_v2"* ]]; then
    pass "GID index $GID_INDEX is RoCEv2"
    info "GID value: $GID_VAL"
elif [[ -n "$GID_TYPE" ]]; then
    warn "GID index $GID_INDEX type: $GID_TYPE (may not be RoCEv2)"
    info "All GIDs on port $IB_PORT:"
    for i in /sys/class/infiniband/"$DEV"/ports/${IB_PORT}/gids/*; do
        IDX=$(basename "$i")
        TYPE=$(cat /sys/class/infiniband/"$DEV"/ports/${IB_PORT}/gid_attrs/types/$IDX 2>/dev/null || echo "?")
        echo "    GID[$IDX]: $(cat $i 2>/dev/null)  type=$TYPE"
    done
else
    fail "Could not read GID index $GID_INDEX"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "7. rdma_common.h settings check"
# ─────────────────────────────────────────────────────────────────────────────
COMMON_H="$(dirname "$(readlink -f "$0")")/../rdma_common.h"
if [[ -f "$COMMON_H" ]]; then
    DEV_IN_FILE=$(grep 'DEVICE_NAME' "$COMMON_H" | grep -o '"mlx5_[0-9]*"' | tr -d '"' || true)
    GID_IN_FILE=$(grep 'GID_INDEX' "$COMMON_H" | grep -o '[0-9]*;' | tr -d ';' | head -1 || true)

    info "rdma_common.h  DEVICE_NAME = \"$DEV_IN_FILE\"  |  GID_INDEX = $GID_IN_FILE"

    [[ "$DEV_IN_FILE" == "$DEV" ]] && \
        pass "DEVICE_NAME matches detected BF3 device" || \
        fail "DEVICE_NAME mismatch: file has \"$DEV_IN_FILE\", BF3 is \"$DEV\""

    [[ "$GID_IN_FILE" == "$GID_INDEX" ]] && \
        pass "GID_INDEX = $GID_INDEX (RoCEv2) is correctly set" || \
        fail "GID_INDEX mismatch: file has $GID_IN_FILE, expected $GID_INDEX"
else
    warn "rdma_common.h not found at $COMMON_H"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}══ All RDMA checks passed — device is ready ══${NC}"
    exit 0
else
    echo -e "${RED}══ $ERRORS check(s) failed — see above ══${NC}"
    exit 1
fi
