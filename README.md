# GVirtuS

The GPU Virtualization Service (GVirtuS) presented in this work tries to fill the gap between in-house hosted computing clusters, equipped with GPGPUs devices, and pay-for-use high performance virtual clusters deployed via public or private computing clouds. gVirtuS allows an instanced virtual machine to access GPGPUs in a transparent and hypervisor independent way, with an overhead slightly greater than a real machine/GPGPU setup. The performance of the components of gVirtuS is assessed through a suite of tests in different deployment scenarios, such as providing GPGPU power to cloud computing based HPC clusters and sharing remotely hosted GPGPUs among HPC nodes.

**Read the official GVirtuS paper [here](https://link.springer.com/chapter/10.1007/978-3-642-15277-1_37).**

# 📄 Published Papers

You can view the full list of all GVirtuS published papers in [CITATIONS](CITATIONS.md).

# How To install GVirtuS Framework and Plugins

## 🧰 Prerequisites

**Tested OS**: Ubuntu 22.04 LTS

Before proceeding, ensure the following dependencies are installed on your system:

* `gcc` compiler and toolchain: _Tested with **v11.4.0** (latest verified working version)_

* [CUDA Drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html): _Tested with **v560.35.03** (latest verified working version)_

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads): _Tested with CUDA Toolkit **v12.6.3** (latest verified working version)_

* [cuDNN](https://developer.nvidia.com/cudnn-downloads): _Tested with cuDNN **v9.5.1** (latest verified working version)_

* [Docker](https://docs.docker.com/engine/install/): _Ensure Docker Engine is properly installed and running. Latest verified working version **v26.1.3**_

> [!NOTE]
> **CUDA Drivers**, **CUDA Toolkit**, and **cuDNN** only need to be installed on the **host machine** running the **GVirtuS backend**.
> Machines acting as **frontends** do **not** require these installations.

## 📦 Installation

1) `git clone` the **GVirtuS** main repository: 

```   
git clone https://github.com/ecn-aau/GVirtuS.git
```

2) CD into the repo directory:

```
cd GVirtuS
```

# 📊 GVirtuS Implementation Status

🗂️ Looking for function coverage? Check out the [**STATUS**](docs/STATUS.md) file for a detailed breakdown of which CUDA functions are:
- 🛠️ Implemented
- 🧪 Tested
- ⚙️ Working

This file tracks progress across major CUDA libraries (e.g., cuBLAS, cuDNN, cuRAND) and helps monitor GVirtuS coverage.


# 🔬 Testing GVirtuS

To test GVirtuS, follow the steps below. This setup runs the GVirtuS backend inside a Docker container with all dependencies pre-installed, and mounts your local source and test files into the container for easy development and debugging.

**1. Start the GVirtuS Backend**

Use the script below to start the GVirtuS backend. It builds GVirtuS from source inside a Docker container and launches the backend process:

```
make run-gvirtus-backend-dev
```

**2. Run the Tests**

Once the backend is running, you can run the tests using the following script. This script creates a new process inside the same container that acts as the frontend and runs all test files located in the tests/ directory:

```
make run-gvirtus-tests
```

**3. Adding Tests**

To add new tests, simply place your test code in any existing .cu file inside the tests directory. You can also create new .cu files if you wish; just make sure to include them as source files in [tests/CMakeLists.txt](tests/CMakeLists.txt#L32).

**4. Updating and Restarting**

After making local changes to the source or tests:

Stop the currently running GVirtuS backend:

```
make stop-gvirtus
```

Ensure your changes are saved.

Restart the backend and re-run the tests using the scripts above.

---

> [!NOTE]
> The GVirtuS backend and frontend communicate over localhost (127.0.0.1), so both processes must run on the same machine.

> [!NOTE]
> The `make run-gvirtus-backend-dev` command starts a Docker container with all necessary GVirtuS dependencies and mounts your local repository files required. This means your local changes are automatically used inside the container, making development and testing fast and efficient—no need to git push or docker push. The `make run-gvirtus-tests` command does **not** start a new container. Instead, it opens a new shell inside the already running backend container and executes the GVirtuS tests there.

> [!IMPORTANT]
> If you make **any changes** to the test files, you must restart the GVirtuS backend using `make stop-gvirtus` followed by `make run-gvirtus-backend-dev`. Otherwise, your test changes will not be picked up.

--- 

# 📦 GVirtuS–OpenPose Integration

The below instructions explains how to set up and run **OpenPose integrated with GVirtuS**, enabling CUDA workloads to be executed on a backend GPU server while running the application on a non-GPU frontend device.

---

## 🚀 Setup Instructions

### 1. Clone the GVirtuS Repository

Clone the GVirtuS project:

```bash
git clone https://github.com/ecn-aau/GVirtuS.git
```

### 2. Navigate into the Repository

```bash
cd GVirtuS
```

### 3. Start the GVirtuS Backend

Launch the backend service on the GPU-enabled device:

```bash
make run-gvirtus-backend-dev
```

> **Note**:
>
> * Before running the OpenPose–GVirtuS integrated application on a **frontend (non-GPU) device**, ensure that the `properties.json` configuration file in the frontend contains the correct **IP address**, **port**, and **endpoint suite** of the backend.

> [!TIP]
> **Controlling log verbosity** — Both the backend and frontend read the `GVIRTUS_LOGLEVEL` environment variable to set the log4cplus log level.
> Set it **before** launching the process:
>
> ```bash
> # Example: enable DEBUG output for the backend
> export GVIRTUS_LOGLEVEL=10000
> make run-gvirtus-backend-dev
> ```
>
> | Level | Value | What you will see |
> |-------|------:|-------------------|
> | TRACE | `0` | Everything (most verbose) |
> | DEBUG | `10000` | Debug messages + all below |
> | INFO | `20000` | Startup / shutdown info (**default**) |
> | WARN | `30000` | Warnings + errors only |
> | ERROR | `40000` | Errors + fatal only |
> | FATAL | `50000` | Fatal errors only |
> | OFF | `60000` | Logging disabled |
>
> The same variable works on the **frontend** side (`src/frontend/Frontend.cpp` reads it identically).
> Inside Docker the Dockerfiles already set a default — override it with `docker run -e GVIRTUS_LOGLEVEL=10000 …` or by editing the `entrypoint.sh`.

### Case 1: Distributed Setup (Different Devices)

If the **GVirtuS backend** is running on a GPU server (or edge device) and the **frontend** is on a different non-GPU device:

* Update the **frontend configuration file** with the backend’s IP and port.
* File to update:

  ```
  {ROOT_FOLDER}/GVirtuS/examples/openpose/properties.json
  ```

Here, the {ROOT_FOLDER} would be your Present Working Directory(PWD) where GVirtuS repo has been installed (ex: home/darshan)

* Get the backend device’s IP address using "ifconfig" command on Backend device.

* Modify {ROOT_FOLDER}/GVirtuS/examples/openpose/properties.json 
Example configuration: 

  ```json
  {
      "suite": "tcp/ip",                   // Replace rdma-roce if you're using rdma
      "protocol": "tcp",                   // Replace roce if you're using rdma
      "server_address": "130.225.243.38",    //Replace with Backend device IP
      "port": "8888"
  }
  ```

---

### Case 2: Local Setup (Same Device)

If both the **GVirtuS backend** and **frontend** are running on the **same server or edge device**:

* No need to find the external IP address.

* Instead, set the server address to **127.0.0.1** in:

  ```
  {ROOT_FOLDER}/GVirtuS/examples/openpose/properties.json
  ```

* Example configuration:

  ```json
  {
      "suite": "tcp/ip",                        // Replace rdma-roce if you're using rdma
      "protocol": "tcp",                        // Replace roce if you're using rdma
      "server_address": "127.0.0.1",
      "port": "8888"
  }
  ```

---

### 4. Run the OpenPose–GVirtuS Demo Application

On the frontend device, start the integrated OpenPose application:

```bash
make run-openpose-test
```

This will connect to the GVirtuS backend component, **transparently redirect all CUDA calls** to the GPU-enabled backend system, and return the results to the frontend application.

---


# 🚀 UCX Communicator (High-Performance Transport)

GVirtuS includes a UCX-based communicator that auto-negotiates TCP, RoCE, or InfiniBand per connection. It supports GPUDirect RDMA for zero-copy GPU transfers over the network.

See the full setup and tuning guide: **[docs/UCX_GUIDE.md](docs/UCX_GUIDE.md)**

Additional UCX documentation:
- [UCX Properties Guide](docs/UCX_PROPERTIES_GUIDE.md) — Makefile variable reference
- [docs/UCX_OPTIMIZATIONS.md](docs/UCX_OPTIMIZATIONS.md) — Optimization stack details
- [docs/GPUDIRECT.md](docs/GPUDIRECT.md) — GPUDirect RDMA implementation

---

# ⚠️ Disclaimers

> [!IMPORTANT]
> GVirtuS is currently not production-ready.
> It is **not thread-safe** and has known **memory leaks**. Use it with caution in experimental or non-critical environments.



