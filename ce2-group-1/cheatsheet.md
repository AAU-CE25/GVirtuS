```bash
wg38up@student.aau.dk@es-dpu-02:~$ lspci | grep -i mellanox
02:00.0 Ethernet controller: Mellanox Technologies MT2894 Family [ConnectX-6 Lx]
02:00.1 Ethernet controller: Mellanox Technologies MT2894 Family [ConnectX-6 Lx]
82:00.0 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.1 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.2 DMA controller: Mellanox Technologies MT43244 BlueField-3 SoC Management Interface (rev 01)
wg38up@student.aau.dk@es-dpu-02:~$ lspci | grep -i bluefield
82:00.0 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.1 Ethernet controller: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller (rev 01)
82:00.2 DMA controller: Mellanox Technologies MT43244 BlueField-3 SoC Management Interface (rev 01)
```