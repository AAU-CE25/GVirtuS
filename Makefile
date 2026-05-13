.PHONY: \
	docker-build-push-dev local-docker-build-backend docker-build-push-prod \
	docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test \
	run-gvirtus-backend-dev attach-gvirtus-bash run-gvirtus-tests stop-gvirtus \
	docker-build-openpose run-openpose-test stop-openpose-test \
	docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test \
	docker-build-simple-matrix local-docker-build-simple-matrix \
	run-simple-matrix-test run-simple-matrix-reconnect-test stop-simple-matrix-test \
	run-gvirtus-backend-async-dev run-async-memset-test \
	docker-build-rapids-matrix run-rapids-matrix-test stop-rapids-matrix-test

USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= ll33pq # change username for local dev!

GVIRTUS_LOG_LEVEL ?= 10000
GVIRTUS_UCX_DATAPATH ?= tag-framed
UCX_TLS ?= tcp,self
UCX_NET_DEVICES ?= ens1f1np1
UCX_LOG_LEVEL ?= info
UCX_SOCKADDR_TLS_PRIORITY ?= tcp
UCX_IB_GID_INDEX ?=
SIMPLE_MATRIX_GPU_FLAGS ?=
UCX_RECONNECT_LOOPS ?= 10

DOCKER_REPO_DEV := $(DOCKER_HUB_USERNAME)/gvirtus-dev
DOCKER_REPO_TEST := $(DOCKER_HUB_USERNAME)/gvirtus-test
DOCKER_REPO_PROD := $(DOCKER_HUB_USERNAME)/gvirtus

# Async memset/fire-and-forget test image and config
ASYNC_TEST_IMAGE ?= $(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04
ASYNC_CONFIG ?= /gvirtus/etc/properties_ucx_async_test.json

# Shared Docker mounts for GVirtuS dev/test containers
DOCKER_GVIRTUS_MOUNTS = \
	-v ./cmake:/gvirtus/cmake/ \
	-v ./etc:/gvirtus/etc/ \
	-v ./include:/gvirtus/include/ \
	-v ./plugins:/gvirtus/plugins/ \
	-v ./src:/gvirtus/src/ \
	-v ./tools:/gvirtus/tools/ \
	-v ./tests:/gvirtus/tests/ \
	-v ./examples:/gvirtus/examples/ \
	-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt

# Shared Docker flags for UCX/RDMA-capable runs
DOCKER_UCX_FLAGS = \
	--network host \
	--privileged \
	--runtime=nvidia \
	--shm-size=8G \
	--device /dev/infiniband \
	--cap-add IPC_LOCK \
	--ulimit memlock=-1

# Shared environment for GVirtuS UCX tests
GVIRTUS_UCX_ENV = \
	-e GVIRTUS_HOME=/gvirtus \
	-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
	-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
	-e UCX_TLS=$(UCX_TLS) \
	-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
	-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
	-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
	$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX))

docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t $(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04 \
		.

local-docker-build-backend:
	docker buildx build \
		--platform linux/amd64 \
		--load \
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
		-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		$(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04

attach-gvirtus-bash:
	docker exec -it gvirtus-$(USER) bash

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

local-docker-build-simple-matrix:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		--no-cache \
		-f examples/simple_matrix/Dockerfile \
		-t $(DOCKER_REPO_DEV)/simple_matrix_gvirtus:cuda12.6 \
		.

run-simple-matrix-test:
	docker run --rm \
		--name simple_matrix_test_container-$(USER) \
		$(SIMPLE_MATRIX_GPU_FLAGS) \
		--network host \
		--device /dev/infiniband \
		--cap-add IPC_LOCK \
		--ulimit memlock=-1 \
		-e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
		-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		-v ./examples/simple_matrix:/opt/GVirtuS/examples \
		-v ./etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json \
		$(DOCKER_REPO_DEV)/simple_matrix_gvirtus:cuda12.6 \
		bash /opt/GVirtuS/examples/frontend.sh

run-simple-matrix-reconnect-test:
	@set -e; \
	loops="$${LOOPS:-$(UCX_RECONNECT_LOOPS)}"; \
	i=1; \
	while [ "$$i" -le "$$loops" ]; do \
		echo "[UCX reconnect] iteration $$i/$$loops"; \
		$(MAKE) run-simple-matrix-test; \
		i=$$((i + 1)); \
	done; \
	echo "[UCX reconnect] completed $$loops iterations"

stop-simple-matrix-test:
	docker stop simple_matrix_test_container-$(USER) || true

run-gvirtus-backend-async-dev:
	docker rm -f gvirtus-async-backend-$(USER) || true
	docker run \
		--rm \
		-it \
		$(DOCKER_UCX_FLAGS) \
		--entrypoint bash \
		--name gvirtus-async-backend-$(USER) \
		$(DOCKER_GVIRTUS_MOUNTS) \
		$(GVIRTUS_UCX_ENV) \
		$(ASYNC_TEST_IMAGE) \
		-lc '\
			set -e; \
			cd /gvirtus; \
			cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/gvirtus; \
			cmake --build build --target gvirtus-backend gvirtus-plugin-cudart gvirtus-communicators-ucx -j$$(nproc); \
			mkdir -p /gvirtus/lib; \
			cp -av /gvirtus/build/libgvirtus-common.so /gvirtus/lib/ || true; \
			cp -av /gvirtus/build/libgvirtus-communicators.so /gvirtus/lib/ || true; \
			cp -av /gvirtus/build/libgvirtus-communicators-ucx.so /gvirtus/lib/; \
			cp -av /gvirtus/build/plugins/cudart/libgvirtus-plugin-cudart.so /gvirtus/lib/; \
			export LD_LIBRARY_PATH=/gvirtus/lib:/gvirtus/build:/gvirtus/build/external/lib:$$LD_LIBRARY_PATH; \
			/gvirtus/build/gvirtus-backend $(ASYNC_CONFIG); \
		'

run-async-memset-test:
	docker rm -f gvirtus-async-frontend-$(USER) || true
	docker run \
		--rm \
		-it \
		$(DOCKER_UCX_FLAGS) \
		--entrypoint bash \
		--name gvirtus-async-frontend-$(USER) \
		$(DOCKER_GVIRTUS_MOUNTS) \
		$(GVIRTUS_UCX_ENV) \
		-e GVIRTUS_CONFIG=$(ASYNC_CONFIG) \
		$(ASYNC_TEST_IMAGE) \
		-lc '\
			set -e; \
			cd /gvirtus; \
			cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/gvirtus; \
			cmake --build build --target gvirtus-frontend cudart gvirtus-communicators-ucx -j$$(nproc); \
			mkdir -p /gvirtus/lib /gvirtus/lib/frontend_async_test; \
			cp -av /gvirtus/build/libgvirtus-common.so /gvirtus/lib/ || true; \
			cp -av /gvirtus/build/libgvirtus-communicators.so /gvirtus/lib/ || true; \
			cp -av /gvirtus/build/libgvirtus-communicators-ucx.so /gvirtus/lib/; \
			cp -av /gvirtus/build/libgvirtus-frontend.so /gvirtus/lib/; \
			cp -av /gvirtus/build/plugins/cudart/libcudart.so* /gvirtus/lib/frontend_async_test/; \
			cd /gvirtus/examples/testing; \
			g++ -std=c++17 test_async_memset.cpp \
				-I/usr/local/cuda/include \
				-L/gvirtus/lib/frontend_async_test \
				-L/gvirtus/lib \
				-L/gvirtus/build \
				-lcudart \
				-lgvirtus-frontend \
				-Wl,-rpath,/gvirtus/lib/frontend_async_test \
				-Wl,-rpath,/gvirtus/lib \
				-Wl,-rpath,/gvirtus/build \
				-Wl,-rpath,/gvirtus/build/external/lib \
				-o test_async_memset; \
			export LD_LIBRARY_PATH=/gvirtus/lib/frontend_async_test:/gvirtus/lib:/gvirtus/build:/gvirtus/build/external/lib:$$LD_LIBRARY_PATH; \
			echo "=== Linked GVirtuS libraries ==="; \
			ldd ./test_async_memset | grep -Ei "libcudart|libgvirtus"; \
			./test_async_memset; \
		'


RAPIDS_MATRIX_IMAGE ?= $(DOCKER_HUB_USERNAME)/rapids_matrix_gvirtus:cuda12.6
RAPIDS_MATRIX_CONTAINER_NAME ?= rapids_matrix_test_container-$(USER)

docker-build-rapids-matrix:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		-f docker/dev/RAPIDS/Dockerfile \
		-t $(RAPIDS_MATRIX_IMAGE) \
		.

run-rapids-matrix-test:
	docker run --rm \
		--name $(RAPIDS_MATRIX_CONTAINER_NAME) \
		--network host \
		--device /dev/infiniband \
		--cap-add IPC_LOCK \
		--ulimit memlock=-1 \
		--entrypoint bash \
		-e GVIRTUS_HOME=/opt/GVirtuS \
		-e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
		-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		-v ./etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json \
		-v ./lib:/opt/GVirtuS/lib \
		-v ./build:/opt/GVirtuS/build \
		-v ./examples/rapids_matrix:/opt/GVirtuS/examples/rapids_matrix \
		$(RAPIDS_MATRIX_IMAGE) \
		-lc '\
			export GVIRTUS_HOME=/opt/GVirtuS; \
			export LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/build/plugins/cudadr:/opt/GVirtuS/build/plugins/cudart:/opt/GVirtuS/build/plugins/cublas:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/cuda/lib64:$$LD_LIBRARY_PATH; \
			export LD_PRELOAD=/opt/GVirtuS/lib/frontend/libcuda.so.1:/opt/GVirtuS/lib/frontend/libcudart.so.12; \
			python3 /opt/GVirtuS/examples/rapids_matrix/rapids_matrix.py \
		'

stop-rapids-matrix-test:
	docker stop $(RAPIDS_MATRIX_CONTAINER_NAME) || true
