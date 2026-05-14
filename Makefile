.PHONY: docker-build-push-dev local-docker-build-backend docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test local-docker-build-simple-matrix
USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= entor# change username for local dev!


GVIRTUS_LOG_LEVEL ?= 10000
GVIRTUS_UCX_DATAPATH ?= am
UCX_LOG_LEVEL ?= info


# UCX_TLS = tcp,self
# UCX_NET_DEVICES = ens1f1np1
# UCX_SOCKADDR_TLS_PRIORITY = tcp
# UCX_IB_GID_INDEX ?= # empty by default; set to 3 for RoCEv2

UCX_TLS=rc_mlx5,ud_mlx5,self
UCX_NET_DEVICES=mlx5_1:1
UCX_SOCKADDR_TLS_PRIORITY=rdmacm
UCX_IB_GID_INDEX=3

SIMPLE_MATRIX_GPU_FLAGS ?=

DOCKER_REPO_DEV := $(DOCKER_HUB_USERNAME)/gvirtus-dev
DOCKER_REPO_TEST := $(DOCKER_HUB_USERNAME)/gvirtus-test
DOCKER_REPO_PROD := $(DOCKER_HUB_USERNAME)/gvirtus

local-docker-build-backend:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t $(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04 \
		.

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
		$(DOCKER_REPO_DEV):cuda12.6.3-cudnn-ubuntu22.04

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