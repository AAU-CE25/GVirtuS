.PHONY: docker-build-push-dev docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test
USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= ul11nh# change username for local dev!



GVIRTUS_LOG_LEVEL ?= 10000
GVIRTUS_UCX_DATAPATH ?= am
GVIRTUS_CONFIG_FILE ?= properties_ucx.json
MATRIX_N ?= 512
UCX_TLS ?= tcp,self
UCX_NET_DEVICES ?= ens1f1np1
UCX_LOG_LEVEL ?= info
UCX_SOCKADDR_TLS_PRIORITY ?= tcp
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
		-e GVIRTUS_CONFIG_FILE=$(GVIRTUS_CONFIG_FILE) \
		-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
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
		--gpus all \
		--network host \
		--device /dev/infiniband \
		--cap-add IPC_LOCK \
		--ulimit memlock=-1 \
		-e GVIRTUS_CONFIG=/opt/GVirtuS/etc/$(GVIRTUS_CONFIG_FILE) \
		-e GVIRTUS_UCX_DATAPATH=$(GVIRTUS_UCX_DATAPATH) \
		-e GVIRTUS_CONFIG_FILE=$(GVIRTUS_CONFIG_FILE) \
		-e MATRIX_N=$(MATRIX_N) \
		-e UCX_TLS=$(UCX_TLS) \
		-e UCX_NET_DEVICES=$(UCX_NET_DEVICES) \
		-e UCX_LOG_LEVEL=$(UCX_LOG_LEVEL) \
		-e UCX_SOCKADDR_TLS_PRIORITY=$(UCX_SOCKADDR_TLS_PRIORITY) \
		$(if $(UCX_IB_GID_INDEX),-e UCX_IB_GID_INDEX=$(UCX_IB_GID_INDEX)) \
		-v ./examples/simple_matrix:/opt/GVirtuS/examples \
		-v ./etc/$(GVIRTUS_CONFIG_FILE):/opt/GVirtuS/etc/$(GVIRTUS_CONFIG_FILE) \
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