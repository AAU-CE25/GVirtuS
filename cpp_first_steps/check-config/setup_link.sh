#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_link.sh
#
# Brings up the BF3 point-to-point link (ens7f1np1) with the correct IP.
# Run this on BOTH machines before starting any RDMA or TCP experiments.
#
# Requires sudo.
#
# Usage:
#   chmod +x setup_link.sh
#   sudo ./setup_link.sh          # auto-detects which node you are
#   sudo ./setup_link.sh --down   # tear the link back down
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IFACE="ens7f1np1"    # BF3 Port 1 — 25.25.25.x subnet

# Auto-detect which node we are
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"dpu-01"* ]]; then
    LOCAL_IP="25.25.25.1"
elif [[ "$HOSTNAME" == *"dpu-02"* ]]; then
    LOCAL_IP="25.25.25.3"
else
    echo "Unknown hostname '$HOSTNAME'. Set LOCAL_IP manually."
    echo "Usage: LOCAL_IP=25.25.25.X sudo ./setup_link.sh"
    exit 1
fi

PREFIX="24"   # /24 subnet mask

# ── Tear down mode ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--down" ]]; then
    echo "[setup_link] Bringing $IFACE DOWN..."
    ip addr del "${LOCAL_IP}/${PREFIX}" dev "$IFACE" 2>/dev/null || true
    ip link set "$IFACE" down
    echo "[setup_link] $IFACE is now DOWN"
    exit 0
fi

# ── Bring up ──────────────────────────────────────────────────────────────────
echo "[setup_link] Configuring $IFACE on $HOSTNAME..."

# Bring interface up (no-op if already up)
ip link set "$IFACE" up

# Remove any existing IP on this interface to avoid duplicates
ip addr flush dev "$IFACE" 2>/dev/null || true

# Assign the correct IP
ip addr add "${LOCAL_IP}/${PREFIX}" dev "$IFACE"

echo "[setup_link] Done."
echo ""
ip addr show "$IFACE"
echo ""
echo "[setup_link] Test with:"
echo "  ip link show $IFACE          # should say: state UP"
echo "  ./check_link.sh              # full validation"
