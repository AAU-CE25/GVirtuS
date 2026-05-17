#!/bin/bash
# Scrape the local host's UCX-relevant network configuration and emit a
# Make-style .env profile suitable for inclusion by the top-level Makefile.
#
# Usage:
#   ./generate-profile.sh                       # print to stdout
#   ./generate-profile.sh > $(hostname -s).env  # write the host's profile
#   ./generate-profile.sh --subnet 24.24.24.0/24 --peer 24.24.24.2
#
# What it tries to detect:
#   HOST_IP       — first IPv4 address on this host inside the chosen subnet
#   HOST_NETDEV   — the netdev that owns HOST_IP (used when UCX_TLS=tcp)
#   UCX_DEV       — the InfiniBand/RoCE device:port bound to HOST_NETDEV
#   UCX_GID_IDX   — the GID index whose IP matches HOST_IP and type is RoCE v2
#
# Fixed defaults written verbatim:
#   UCX_TLS=rc_verbs,tcp
#   UCX_PORT=7676
#   HOST_IP_PEER=<--peer arg or empty>
set -euo pipefail

SUBNET="24.24.24.0/24"
PEER=""
PORT=7676

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subnet) SUBNET="$2"; shift 2 ;;
        --peer)   PEER="$2";   shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Find the local IP inside the chosen subnet, and its owning netdev.
# ---------------------------------------------------------------------------
SUBNET_PREFIX="${SUBNET%/*}"
SP1=$(echo "$SUBNET_PREFIX" | cut -d. -f1)
SP2=$(echo "$SUBNET_PREFIX" | cut -d. -f2)
SP3=$(echo "$SUBNET_PREFIX" | cut -d. -f3)

LINE=""
ADDR_LIST=$(ip -o -4 addr show | awk '{print $2, $4}')
while read -r dev cidr; do
    ip="${cidr%/*}"
    o1=$(echo "$ip" | cut -d. -f1)
    o2=$(echo "$ip" | cut -d. -f2)
    o3=$(echo "$ip" | cut -d. -f3)
    if [[ "$o1" == "$SP1" && "$o2" == "$SP2" && "$o3" == "$SP3" ]]; then
        LINE="$dev $ip"
        break
    fi
done <<< "$ADDR_LIST"

if [[ -z "$LINE" ]]; then
    echo "ERROR: no local IPv4 address found in ${SUBNET}" >&2
    exit 1
fi
HOST_NETDEV="${LINE%% *}"
HOST_IP="${LINE##* }"

# ---------------------------------------------------------------------------
# 2. Map netdev → IB device:port via /sys/class/net/<dev>/device/infiniband/
# ---------------------------------------------------------------------------
UCX_DEV=""
IBDIR="/sys/class/net/${HOST_NETDEV}/device/infiniband"
if [[ -d "$IBDIR" ]]; then
    IBDEV=$(ls "$IBDIR" 2>/dev/null | head -1 || true)
    if [[ -n "$IBDEV" ]]; then
        # IB devices expose port numbers under /sys/class/infiniband/<ibdev>/ports/
        PORT_NUM=$(ls "/sys/class/infiniband/${IBDEV}/ports/" 2>/dev/null | head -1)
        UCX_DEV="${IBDEV}:${PORT_NUM:-1}"
    fi
fi
if [[ -z "$UCX_DEV" ]]; then
    echo "WARN: could not derive IB device for netdev '$HOST_NETDEV' — using mlx5_0:1" >&2
    UCX_DEV="mlx5_0:1"
fi

# ---------------------------------------------------------------------------
# 3. Find the GID index for HOST_IP on UCX_DEV with RoCEv2.
#    Prefer `show_gids`; otherwise walk /sys.
# ---------------------------------------------------------------------------
IBDEV_NAME="${UCX_DEV%:*}"
PORT_NUM="${UCX_DEV##*:}"
UCX_GID_IDX=""

if command -v show_gids >/dev/null 2>&1; then
    # show_gids columns: DEV PORT INDEX GID IPv4 VER DEV ...
    # Do not `exit` inside awk on a pipe — it triggers SIGPIPE upstream.
    SG_OUT=$(show_gids 2>/dev/null || true)
    UCX_GID_IDX=$(echo "$SG_OUT" \
        | awk -v d="$IBDEV_NAME" -v p="$PORT_NUM" -v ip="$HOST_IP" '
            $1==d && $2==p && $5==ip && $6 ~ /v2/ { print $3 }' \
        | head -1)
fi

if [[ -z "$UCX_GID_IDX" ]]; then
    # Fallback: scan /sys/class/infiniband/<dev>/ports/<p>/gids and gid_attrs/types
    GID_DIR="/sys/class/infiniband/${IBDEV_NAME}/ports/${PORT_NUM}/gids"
    TYPE_DIR="/sys/class/infiniband/${IBDEV_NAME}/ports/${PORT_NUM}/gid_attrs/types"
    if [[ -d "$GID_DIR" ]]; then
        # Encode HOST_IP into the bottom 4 bytes of a v6-mapped GID and match.
        IFS=. read -r o1 o2 o3 o4 <<< "$HOST_IP"
        HEX_IP=$(printf '%02x%02x:%02x%02x' "$o1" "$o2" "$o3" "$o4")
        for i in "$GID_DIR"/*; do
            idx=$(basename "$i")
            gid=$(cat "$i" 2>/dev/null || true)
            type=$(cat "$TYPE_DIR/$idx" 2>/dev/null || true)
            if [[ "$gid" == *"$HEX_IP" && "$type" == *"RoCE v2"* ]]; then
                UCX_GID_IDX="$idx"; break
            fi
        done
    fi
fi
UCX_GID_IDX="${UCX_GID_IDX:-1}"

# ---------------------------------------------------------------------------
# 4. Emit the profile.
# ---------------------------------------------------------------------------
cat <<EOF
# Auto-generated UCX profile for $(hostname -s) by etc/ucx/generate-profile.sh
# Subnet: ${SUBNET}    Peer (--peer): ${PEER:-<unset>}

UCX_DEV     := ${UCX_DEV}
UCX_GID_IDX := ${UCX_GID_IDX}
UCX_TLS     := rc_verbs,tcp

HOST_IP      := ${HOST_IP}
HOST_IP_PEER := ${PEER:-CHANGE_ME}

HOST_NETDEV  := ${HOST_NETDEV}
UCX_PORT     := ${PORT}
EOF
