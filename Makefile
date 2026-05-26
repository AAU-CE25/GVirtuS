.PHONY: docker-build-push-dev local-docker-build-backend docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test local-docker-build-simple-matrix
USER := ll33pq
DOCKER_HUB_USERNAME ?= ll33pq# change username for local dev!


GVIRTUS_LOG_LEVEL ?= 10000
GVIRTUS_UCX_DATAPATH ?= am
UCX_TLS ?=
UCX_NET_DEVICES ?=
UCX_LOG_LEVEL ?= info
UCX_SOCKADDR_TLS_PRIORITY ?=
UCX_IB_GID_INDEX ?= # empty by default; set to 3 for RoCEv2
SIMPLE_MATRIX_GPU_FLAGS ?=


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
		$(if $(BACKEND_CONFIG),-e BACKEND_CONFIG=$(BACKEND_CONFIG)) \
		$(if $(UCX_TLS),-e UCX_TLS=$(UCX_TLS)) \
		$(if $(UCX_NET_DEVICES),-e UCX_NET_DEVICES=$(UCX_NET_DEVICES)) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		$(if $(UCX_SOCKADDR_TLS_PRIORITY),-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY)) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		$(if $(GVIRTUS_RMA_ZEROCOPY),-e GVIRTUS_RMA_ZEROCOPY=$(GVIRTUS_RMA_ZEROCOPY)) \
		$(if $(GVIRTUS_GPUDIRECT),-e GVIRTUS_GPUDIRECT=$(GVIRTUS_GPUDIRECT)) \
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
		$(if $(UCX_TLS),-e UCX_TLS=$(UCX_TLS)) \
		$(if $(UCX_NET_DEVICES),-e UCX_NET_DEVICES=$(UCX_NET_DEVICES)) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		$(if $(UCX_SOCKADDR_TLS_PRIORITY),-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY)) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		$(if $(UCX_RNDV_THRESH),-e UCX_RNDV_THRESH=$(UCX_RNDV_THRESH)) \
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