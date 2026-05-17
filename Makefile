.PHONY: docker-build-push-dev docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-backend-tcp run-gvirtus-backend-ucx run-gvirtus-backend-ucx-tcp run-gvirtus-backend-ucx-rdma run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test docker-build-ucx-benchmark run-ucx-benchmark-server-tcp run-ucx-benchmark-server-ucx run-ucx-benchmark-tcp run-ucx-benchmark-ucx stop-ucx-benchmark run-ucx-matrix-single run-matrix-bench-tcp run-matrix-bench-ucx-tcp run-matrix-bench-ucx-rdma
USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= aauce25

GVIRTUS_LOG_LEVEL ?= 20000

# ---------------------------------------------------------------------------
# UCX per-host profile
#
# Auto-loaded from etc/ucx/<hostname>.env if it exists. Each profile pins the
# RDMA device name and GID index used by UCX at runtime. Ports and server
# addresses live exclusively in the JSON config files under examples/ucx_benchmark/.
# Override on the CLI to bypass: make ... UCX_PROFILE=etc/ucx/es-dpu-02.env
# ---------------------------------------------------------------------------
SHORT_HOST  := $(shell hostname -s)
UCX_PROFILE ?= etc/ucx/$(SHORT_HOST).env
ifneq ("$(wildcard $(UCX_PROFILE))","")
    include $(UCX_PROFILE)
    $(info [ucx] loaded profile: $(UCX_PROFILE) (UCX_DEV=$(UCX_DEV) UCX_GID_IDX=$(UCX_GID_IDX) UCX_TLS=$(UCX_TLS)))
else
    $(warning [ucx] no profile for host '$(SHORT_HOST)' at $(UCX_PROFILE); using defaults)
endif

# Defaults if no profile was loaded (override on CLI).
UCX_DEV          ?= mlx5_0:1
UCX_GID_IDX      ?= 1
UCX_TLS          ?= rc_verbs,tcp
HOST_NETDEV      ?= ens1f0np0
# Eager/rendezvous crossover (bytes). Below this size UCX uses EAGER (1-copy
# send); at/above it switches to RENDEZVOUS (zero-copy RDMA READ for the
# RDMA transports, or larger TCP buffers for tcp). Override on CLI:
#   make run-matrix-bench-ucx-rdma UCX_RNDV_THRESH=65536
# Set to 0 to force rendezvous for every message.
UCX_RNDV_THRESH  ?= 0

# Single source of truth for GVirtuS endpoint config (server_address + port live
# inside the JSON; edit those files by hand if you need to change them).
GVIRTUS_TCP_CONFIG := $(PWD)/examples/ucx_benchmark/properties_tcp.json
GVIRTUS_UCX_CONFIG := $(PWD)/examples/ucx_benchmark/properties_ucx.json

DOCKER_REPO_DEV := $(DOCKER_HUB_USERNAME)/gvirtus-dev
DOCKER_REPO_TEST := $(DOCKER_HUB_USERNAME)/gvirtus-test
DOCKER_REPO_PROD := $(DOCKER_HUB_USERNAME)/gvirtus

docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t $(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-push-prod:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/prod/Dockerfile \
		-t $(DOCKER_REPO_PROD):cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-push-docker-test:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/test/Dockerfile \
		-t $(DOCKER_REPO_TEST):latest \
		.

run-docker-gvirtus-test:
	docker run \
		--rm \
		--name gvirtus-test-$(USER) \
		-it $(DOCKER_REPO_TEST):latest

stop-docker-gvirtus-test:
	docker stop gvirtus-test-$(USER) || true

run-gvirtus-backend-dev:
	docker run \
		--rm \
		-it \
		--network host \
		--privileged \
		-v ./cmake:/gvirtus/cmake/ \
		-v ./etc:/gvirtus/etc/ \
		-v ./include:/gvirtus/include/ \
		-v ./plugins:/gvirtus/plugins/ \
		-v ./src:/gvirtus/src/ \
		-v ./tools:/gvirtus/tools/ \
		-v ./tests:/gvirtus/tests/ \
		-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
		-v ./docker/dev/entrypoint.sh:/entrypoint.sh \
		-v ./examples:/gvirtus/examples/ \
		--entrypoint /entrypoint.sh \
		--name gvirtus-$(USER) \
		--runtime=nvidia \
		--shm-size=8G \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		$(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04

attach-gvirtus-bash:
		docker exec -it gvirtus-$(USER) bash

# GVirtuS endpoint config:
#   - Single source of truth per protocol, hand-maintained, lives in examples/ucx_benchmark/.
#   - server_address + port are inside the JSON; the Makefile never edits them.
#   - Backend targets mount the JSON read-only into /opt/gvirtus-config.json
#     and point GVIRTUS_CONFIG at it. Client targets mount the same JSON into
#     /opt/GVirtuS/etc/properties.json so the frontend stub picks it up.

run-gvirtus-backend-ucx:
	docker run \
		--rm \
		-it \
		--network host \
		--privileged \
		-v ./cmake:/gvirtus/cmake/ \
		-v ./etc:/gvirtus/etc/ \
		-v ./include:/gvirtus/include/ \
		-v ./plugins:/gvirtus/plugins/ \
		-v ./src:/gvirtus/src/ \
		-v ./tools:/gvirtus/tools/ \
		-v ./tests:/gvirtus/tests/ \
		-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
		-v ./docker/dev/entrypoint.sh:/entrypoint.sh \
		-v ./examples:/gvirtus/examples/ \
		-v $(GVIRTUS_UCX_CONFIG):/opt/gvirtus-config.json:ro \
		--entrypoint /entrypoint.sh \
		--name gvirtus-ucx-$(USER) \
		--runtime=nvidia \
		--shm-size=8G \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e GVIRTUS_CONFIG=/opt/gvirtus-config.json \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_DEV) \
		-e UCX_IB_GID_INDEX=$(UCX_GID_IDX) \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04

run-gvirtus-tests:
	docker exec \
		-it gvirtus-$(USER) \
		bash -c \
		'export LD_LIBRARY_PATH=$$GVIRTUS_HOME/lib/frontend:$$LD_LIBRARY_PATH && \
			cd /gvirtus/build && \
			ctest --output-on-failure'

stop-gvirtus:
	docker stop gvirtus-$(USER) || true


docker-build-openpose:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f examples/openpose/Dockerfile \
		-t $(DOCKER_HUB_USERNAME)/openpose_gvirtus:cuda12.6 \
		examples/openpose	


run-openpose-test: 
	docker run --rm \
		--name openpose_container-$(USER) \
		--network host \
		-v ./examples/openpose/media:/opt/openpose/examples/media \
		-v ./examples/openpose:/opt/openpose/examples/gvirtus \
		-v ./examples/openpose/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/openpose/entrypoint.sh:/entrypoint.sh \
		$(DOCKER_HUB_USERNAME)/openpose_gvirtus:cuda12.6 \
		bash /entrypoint.sh

stop-openpose-test:
	docker stop openpose_container-$(USER) || true

docker-build-2d-human-parsing:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f examples/2d-human-parsing/Dockerfile \
		-t $(DOCKER_HUB_USERNAME)/human-parsing_gvirtus:cuda12.6 \
		examples/2d-human-parsing	

run-2d-human-parsing-test: 
	docker run --rm \
		--name human_parsing_test_container-$(USER) \
		--network host \
		--shm-size=8G \
		-v ./examples/2d-human-parsing/inference_acc_00.py:/opt/2D-Human-Parsing/inference/inference_acc_00.py \
		-v ./examples/2d-human-parsing/demo_imgs:/opt/2D-Human-Parsing/demo_imgs \
		-v ./examples/2d-human-parsing/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/2d-human-parsing/entrypoint.sh:/entrypoint.sh \
		$(DOCKER_HUB_USERNAME)/human-parsing_gvirtus:cuda12.6 \
		bash /entrypoint.sh

stop-2d-human-parsing-test:
	docker stop human_parsing_test_container-$(USER) || true

docker-build-simple-matrix:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f examples/simple_matrix/Dockerfile \
		-t $(DOCKER_HUB_USERNAME)/simple_matrix_gvirtus:cuda12.6 \
		.	

run-simple-matrix-test: 
	docker run --rm \
		--name simple_matrix_test_container-$(USER) \
		--network host \
		-v ./examples/simple_matrix:/opt/GVirtuS/examples \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json \
		$(DOCKER_HUB_USERNAME)/simple_matrix_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/frontend.sh

stop-simple-matrix-test:
	docker stop simple_matrix_test_container-$(USER) || true

# ===== UCX Benchmark =====

# Benchmark configuration (override on command line)
BENCH_TEST ?= all
BENCH_RUNS ?= 10
BENCH_DATA_SIZES ?= 1024,8192,65536,262144,1048576,4194304,16777216
BENCH_MATRIX_NS ?= 64,128,256,512,1024,2048
BENCH_SERVER ?= 24.24.24.1
BENCH_PORT ?= 5555

docker-build-ucx-benchmark:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		-f examples/ucx_benchmark/Dockerfile \
		-t $(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		.

# Start the data_copy echo server (run on the backend/GPU node)
run-ucx-benchmark-server-tcp:
	docker run --rm \
		--name ucx_bench_server_tcp-$(USER) \
		--network host \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash -c "g++ -O2 -o /tmp/data_copy_bench /opt/GVirtuS/examples/ucx_benchmark/data_copy_bench.cpp -lpthread && /tmp/data_copy_bench server tcp $(BENCH_PORT)"

run-ucx-benchmark-server-ucx:
	docker run --rm \
		--name ucx_bench_server_ucx-$(USER) \
		--network host \
		-e UCX_TLS=rc_x,tcp \
		-e UCX_NET_DEVICES=all \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash -c "g++ -O2 -DUSE_UCX -o /tmp/data_copy_bench /opt/GVirtuS/examples/ucx_benchmark/data_copy_bench.cpp -lucp -lucs -lpthread && /tmp/data_copy_bench server ucx $(BENCH_PORT)"

# Run benchmarks as client (data_copy uses its own transport, matrix_mul uses GVirtuS)
run-ucx-benchmark-tcp:
	docker run --rm \
		--name ucx_benchmark_tcp-$(USER) \
		--network host \
		-v ./examples/ucx_benchmark/results:/opt/GVirtuS/examples/ucx_benchmark/results \
		-v ./examples/ucx_benchmark/properties_tcp.json:/opt/GVirtuS/etc/properties.json \
		-e BENCH_TEST=$(BENCH_TEST) \
		-e BENCH_RUNS=$(BENCH_RUNS) \
		-e BENCH_DATA_SIZES=$(BENCH_DATA_SIZES) \
		-e BENCH_MATRIX_NS=$(BENCH_MATRIX_NS) \
		-e BENCH_TAG=tcp \
		-e BENCH_TRANSPORT=tcp \
		-e BENCH_SERVER=$(BENCH_SERVER) \
		-e BENCH_PORT=$(BENCH_PORT) \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend.sh

run-ucx-benchmark-ucx:
	docker run --rm \
		--name ucx_benchmark_ucx-$(USER) \
		--network host \
		-v ./examples/ucx_benchmark/results:/opt/GVirtuS/examples/ucx_benchmark/results \
		-v ./examples/ucx_benchmark/properties_ucx.json:/opt/GVirtuS/etc/properties.json \
		-e BENCH_TEST=$(BENCH_TEST) \
		-e BENCH_RUNS=$(BENCH_RUNS) \
		-e BENCH_DATA_SIZES=$(BENCH_DATA_SIZES) \
		-e BENCH_MATRIX_NS=$(BENCH_MATRIX_NS) \
		-e BENCH_TAG=ucx \
		-e BENCH_TRANSPORT=ucx \
		-e BENCH_SERVER=$(BENCH_SERVER) \
		-e BENCH_PORT=$(BENCH_PORT) \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e UCX_TLS=rc_x,tcp \
		-e UCX_NET_DEVICES=all \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend.sh

stop-ucx-benchmark:
	docker stop ucx_bench_server_tcp-$(USER) || true
	docker stop ucx_bench_server_ucx-$(USER) || true
	docker stop ucx_benchmark_tcp-$(USER) || true
	docker stop ucx_benchmark_ucx-$(USER) || true
	docker stop ucx_matrix_ucx-$(USER) || true

# Smoke test: single run of matrix_mul_bench through GVirtuS over UCX.
# Requires `make run-gvirtus-backend-ucx` to be running on the GPU node.
# Override N / RUNS on the CLI: `make run-ucx-matrix-single N=256 RUNS=1`
N    ?= 128
RUNS ?= 1
run-ucx-matrix-single:
	docker run --rm \
		--name ucx_matrix_ucx-$(USER) \
		--network host \
		--runtime=nvidia \
		--privileged \
		--ulimit memlock=-1 \
		--shm-size=8G \
		-v ./examples/ucx_benchmark:/opt/GVirtuS/examples/ucx_benchmark \
		-v ./examples/ucx_benchmark/properties_ucx.json:/opt/GVirtuS/etc/properties.json \
		-e N=$(N) \
		-e RUNS=$(RUNS) \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_DEV) \
		-e UCX_IB_GID_INDEX=$(UCX_GID_IDX) \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend_matrix_only.sh
# ===== Matrix-mul transport comparison sweep =====
#
# Three back-to-back-runnable benchmarks; same matrix_mul_bench client, three
# different GVirtuS communicators. The backend has to be (re)started in the
# matching mode between runs:
#
#   GPU node                                Client node
#   --------                                -----------
#   make run-gvirtus-backend-tcp            make run-matrix-bench-tcp
#   make run-gvirtus-backend-ucx-tcp        make run-matrix-bench-ucx-tcp
#   make run-gvirtus-backend-ucx-rdma       make run-matrix-bench-ucx-rdma
#
# Each client run appends a CSV under examples/ucx_benchmark/results/ tagged
# with the transport name and a timestamp. Override the sweep parameters via:
#   make run-matrix-bench-tcp BENCH_NS=128,256,512 BENCH_RUNS=20
BENCH_NS   ?= 128,256,512,1024,2048,4096
BENCH_RUNS ?= 10

# ---- backends ----

# Pure TCP backend (uses TcpCommunicator). Mirrors run-gvirtus-backend-dev but
# mounts the regenerated TCP properties file so the bench client and the
# backend agree on host:port from the per-host profile.
run-gvirtus-backend-tcp:
	docker run \
		--rm \
		-it \
		--network host \
		--privileged \
		-v ./cmake:/gvirtus/cmake/ \
		-v ./etc:/gvirtus/etc/ \
		-v ./include:/gvirtus/include/ \
		-v ./plugins:/gvirtus/plugins/ \
		-v ./src:/gvirtus/src/ \
		-v ./tools:/gvirtus/tools/ \
		-v ./tests:/gvirtus/tests/ \
		-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
		-v ./docker/dev/entrypoint.sh:/entrypoint.sh \
		-v ./examples:/gvirtus/examples/ \
		-v $(GVIRTUS_TCP_CONFIG):/opt/gvirtus-config.json:ro \
		--entrypoint /entrypoint.sh \
		--name gvirtus-tcp-$(USER) \
		--runtime=nvidia \
		--shm-size=8G \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e GVIRTUS_CONFIG=/opt/gvirtus-config.json \
		$(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04

# UCX backend forced to TCP transport (no RDMA): isolates UCX framing overhead
# from the network medium. The UCX `tcp` transport works on a kernel netdev,
# not an IB device, so we override UCX_DEV to HOST_NETDEV here.
run-gvirtus-backend-ucx-tcp:
	$(MAKE) run-gvirtus-backend-ucx UCX_TLS=tcp UCX_DEV=$(HOST_NETDEV)

# UCX backend over RDMA (RoCE v2 via rc_verbs).
run-gvirtus-backend-ucx-rdma:
	$(MAKE) run-gvirtus-backend-ucx UCX_TLS=rc_verbs

# ---- clients ----

run-matrix-bench-tcp:
	docker run --rm \
		--name matrix_bench_tcp-$(USER) \
		--network host \
		--runtime=nvidia \
		--shm-size=8G \
		-v ./examples/ucx_benchmark:/opt/GVirtuS/examples/ucx_benchmark \
		-v ./examples/ucx_benchmark/properties_tcp.json:/opt/GVirtuS/etc/properties.json \
		-e BENCH_NS=$(BENCH_NS) \
		-e RUNS=$(BENCH_RUNS) \
		-e BENCH_TAG=tcp \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend_matrix_sweep.sh

run-matrix-bench-ucx-tcp:
	docker run --rm \
		--name matrix_bench_ucx_tcp-$(USER) \
		--network host \
		--runtime=nvidia \
		--privileged \
		--ulimit memlock=-1 \
		--shm-size=8G \
		-v ./examples/ucx_benchmark:/opt/GVirtuS/examples/ucx_benchmark \
		-v ./examples/ucx_benchmark/properties_ucx.json:/opt/GVirtuS/etc/properties.json \
		-e BENCH_NS=$(BENCH_NS) \
		-e RUNS=$(BENCH_RUNS) \
		-e BENCH_TAG=ucx-tcp \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e UCX_TLS=tcp \
		-e UCX_NET_DEVICES=$(HOST_NETDEV) \
		-e UCX_IB_GID_INDEX=$(UCX_GID_IDX) \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend_matrix_sweep.sh

run-matrix-bench-ucx-rdma:
	docker run --rm \
		--name matrix_bench_ucx_rdma-$(USER) \
		--network host \
		--runtime=nvidia \
		--privileged \
		--ulimit memlock=-1 \
		--shm-size=8G \
		-v ./examples/ucx_benchmark:/opt/GVirtuS/examples/ucx_benchmark \
		-v ./examples/ucx_benchmark/properties_ucx.json:/opt/GVirtuS/etc/properties.json \
		-e BENCH_NS=$(BENCH_NS) \
		-e RUNS=$(BENCH_RUNS) \
		-e BENCH_TAG=ucx-rdma \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e UCX_TLS=rc_verbs \
		-e UCX_NET_DEVICES=$(UCX_DEV) \
		-e UCX_IB_GID_INDEX=$(UCX_GID_IDX) \
		-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH) \
		$(DOCKER_HUB_USERNAME)/ucx_benchmark_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/ucx_benchmark/frontend_matrix_sweep.sh
