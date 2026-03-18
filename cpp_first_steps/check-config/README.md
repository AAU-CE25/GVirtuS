# check-config

Scripts to validate the network and RDMA setup between `es-dpu-01` and `es-dpu-02` before running any experiments.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_link.sh` | Brings up `ens7f1np1` with the correct IP |
| `check_link.sh` | Validates the TCP/IP link between the two machines |
| `check_rdma.sh` | Validates the BlueField-3 RDMA device and RoCEv2 GID config |

## Quickstart — run before every experiment

### Step 1: Bring up the link (both machines, requires sudo)

```bash
# On es-dpu-01
sudo ./check-config/setup_link.sh

# On es-dpu-02
sudo ./check-config/setup_link.sh
```

### Step 2: Validate the link

```bash
# On either machine
./check-config/check_link.sh
```

Expected output:
```
══ 1. Interface exists ══
[PASS] Interface ens7f1np1 exists

══ 2. Interface state ══
[PASS] Interface ens7f1np1 is UP

══ 3. IP address assigned ══
[PASS] IP 25.25.25.1 correctly assigned to ens7f1np1

══ 4. Link speed (ethtool) ══
[PASS] Link detected: yes  |  Speed: 100000Mb/s

══ 5. Ping remote host (es-dpu-02) ══
[PASS] Ping to 25.25.25.3 succeeded  (avg RTT: 0.123ms)

══ 6. Hostname resolution ══
[PASS] es-dpu-02 resolves to 25.25.25.3 (via /etc/hosts)

══ All checks passed — link is ready ══
```

### Step 3: Validate RDMA device (before RDMA experiments)

```bash
./check-config/check_rdma.sh
```

Key things it checks:
- BF3 visible at PCI `82:00`
- Correct `mlx5_X` device name (must match `DEVICE_NAME` in `rdma_common.h`)
- Port state is `PORT_ACTIVE`
- GID index 3 is RoCEv2
- `rdma_common.h` settings are consistent with detected hardware

## Network topology

```
es-dpu-01                                 es-dpu-02
ens7f1np1 (25.25.25.1) ←── 1 cable ──→  ens7f1np1 (25.25.25.3)
         BlueField-3 Port 1                   BlueField-3 Port 1
```

## Tear down

```bash
sudo ./check-config/setup_link.sh --down
```
