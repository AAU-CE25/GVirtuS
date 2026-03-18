#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# check_link.sh
#
# Validates the point-to-point Ethernet link between es-dpu-01 and es-dpu-02
# over the 25.25.25.x subnet (ens7f1np1 — BF3 Port 1, preferred RDMA link).
#
# Run this on EITHER machine. It will auto-detect which node it's on.
#
# Usage:
#   chmod +x check_link.sh
#   ./check_link.sh
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'  # No Colour

PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
INFO="${BLUE}[INFO]${NC}"
WARN="${YELLOW}[WARN]${NC}"

ERRORS=0

pass()  { echo -e "${PASS} $*"; }
fail()  { echo -e "${FAIL} $*"; ((ERRORS++)) || true; }
info()  { echo -e "${INFO} $*"; }
warn()  { echo -e "${WARN} $*"; }
header(){ echo -e "\n${BLUE}══ $* ══${NC}"; }

# ── Topology ──────────────────────────────────────────────────────────────────
IFACE="ens7f1np1"          # BF3 Port 1 — preferred RDMA/MPI link

# Auto-detect local/remote IP based on hostname
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"dpu-01"* ]]; then
    LOCAL_IP="25.25.25.1"
    REMOTE_IP="25.25.25.3"
    REMOTE_HOST="es-dpu-02"
elif [[ "$HOSTNAME" == *"dpu-02"* ]]; then
    LOCAL_IP="25.25.25.3"
    REMOTE_IP="25.25.25.1"
    REMOTE_HOST="es-dpu-01"
else
    warn "Unknown hostname '$HOSTNAME' — using manual fallback"
    LOCAL_IP="${LOCAL_IP:-25.25.25.1}"
    REMOTE_IP="${REMOTE_IP:-25.25.25.3}"
    REMOTE_HOST="${REMOTE_HOST:-es-dpu-02}"
fi

echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        BF3 Link Validation Script        ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"
info "Running on  : $HOSTNAME"
info "Local IP    : $LOCAL_IP  ($IFACE)"
info "Remote host : $REMOTE_HOST ($REMOTE_IP)"

# ─────────────────────────────────────────────────────────────────────────────
header "1. Interface exists"
# ─────────────────────────────────────────────────────────────────────────────
if ip link show "$IFACE" &>/dev/null; then
    pass "Interface $IFACE exists"
else
    fail "Interface $IFACE not found — check BF3 driver / PCI slot"
    echo "  Available interfaces:"
    ip link show | grep -E '^[0-9]+:' | awk '{print "    " $2}'
    exit 1   # no point continuing
fi

# ─────────────────────────────────────────────────────────────────────────────
header "2. Interface state"
# ─────────────────────────────────────────────────────────────────────────────
STATE=$(cat /sys/class/net/"$IFACE"/operstate 2>/dev/null || echo "unknown")
if [[ "$STATE" == "up" ]]; then
    pass "Interface $IFACE is UP"
else
    fail "Interface $IFACE is $STATE"
    warn "Fix: sudo ip link set $IFACE up"
    warn "Fix: sudo ip addr add ${LOCAL_IP}/24 dev $IFACE"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "3. IP address assigned"
# ─────────────────────────────────────────────────────────────────────────────
ASSIGNED_IP=$(ip -4 addr show "$IFACE" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1)
if [[ "$ASSIGNED_IP" == "$LOCAL_IP" ]]; then
    pass "IP $LOCAL_IP correctly assigned to $IFACE"
elif [[ -n "$ASSIGNED_IP" ]]; then
    warn "Interface has IP $ASSIGNED_IP, expected $LOCAL_IP"
else
    fail "No IP address assigned to $IFACE"
    warn "Fix: sudo ip addr add ${LOCAL_IP}/24 dev $IFACE"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "4. Link speed (ethtool)"
# ─────────────────────────────────────────────────────────────────────────────
if command -v ethtool &>/dev/null; then
    SPEED=$(ethtool "$IFACE" 2>/dev/null | grep "Speed:" | awk '{print $2}')
    LINK=$(ethtool  "$IFACE" 2>/dev/null | grep "Link detected:" | awk '{print $3}')
    if [[ "$LINK" == "yes" ]]; then
        pass "Link detected: yes  |  Speed: $SPEED"
    else
        fail "Link detected: no — cable unplugged or remote port is DOWN"
    fi
else
    warn "ethtool not installed (sudo apt install ethtool)"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "5. Ping remote host ($REMOTE_HOST)"
# ─────────────────────────────────────────────────────────────────────────────
if ping -c3 -W2 -I "$IFACE" "$REMOTE_IP" &>/dev/null; then
    RTT=$(ping -c3 -W2 -I "$IFACE" "$REMOTE_IP" 2>/dev/null \
          | tail -1 | awk -F'/' '{print $5}' )
    pass "Ping to $REMOTE_IP succeeded  (avg RTT: ${RTT}ms)"
else
    fail "Cannot ping $REMOTE_IP"
    warn "Make sure $REMOTE_HOST also has $IFACE UP with IP assigned"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "6. Hostname resolution"
# ─────────────────────────────────────────────────────────────────────────────
RESOLVED=$(getent hosts "$REMOTE_HOST" 2>/dev/null | awk '{print $1}')
if [[ "$RESOLVED" == "$REMOTE_IP" ]]; then
    pass "$REMOTE_HOST resolves to $REMOTE_IP (via /etc/hosts)"
elif [[ -n "$RESOLVED" ]]; then
    warn "$REMOTE_HOST resolves to $RESOLVED (expected $REMOTE_IP)"
else
    fail "$REMOTE_HOST does not resolve"
    warn "Add to /etc/hosts:  $REMOTE_IP  $REMOTE_HOST"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "7. TCP port 9000 (ping app)"
# ─────────────────────────────────────────────────────────────────────────────
if command -v nc &>/dev/null; then
    if nc -z -w2 "$REMOTE_IP" 9000 &>/dev/null; then
        pass "Port 9000 is open on $REMOTE_IP (helloServer is running)"
    else
        warn "Port 9000 not reachable — start helloServer on $REMOTE_HOST first"
    fi
else
    warn "nc (netcat) not available — skipping port check"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}══ All checks passed — link is ready ══${NC}"
    exit 0
else
    echo -e "${RED}══ $ERRORS check(s) failed — see above ══${NC}"
    exit 1
fi
