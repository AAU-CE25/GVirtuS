.PHONY: docker-build-push-dev docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test run-spark-local-cpu run-spark-local-rapids run-spark-local docker-build-spark-local run-spark-docker-local run-spark-docker-local-cpu run-spark-docker-local-rapids stop-spark-simple-matrix docker-build-spark-gvirtus run-spark-docker-gvirtus test-spark-gvirtus
USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= aauce25

GVIRTUS_LOG_LEVEL ?= 0

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

# ═══════════════════════════════════════════════════════════════════════════════
# spark_simple_matrix - Matrix multiplication benchmark with Spark + RAPIDS
#
# Three execution modes:
#   1. Local native   - Run directly on host (make run-spark-local-*)
#   2. Docker local   - Run in Docker with local GPU (make run-spark-docker-local-*)
#   3. Docker GVirtuS - Run in Docker with remote GPU (make run-spark-docker-gvirtus)
# ═══════════════════════════════════════════════════════════════════════════════

SPARK_MATRIX_DIR := examples/spark_simple_matrix

# ── 1. LOCAL NATIVE (run directly on host) ──
# Requirements: Python 3.10+, Java 17+, NVIDIA GPU + driver

run-spark-local-cpu:
	cd $(SPARK_MATRIX_DIR)/src && python3 simple_matrix.py local \
		--mode cpu \
		--overwrite yes

run-spark-local-rapids: 
	cd $(SPARK_MATRIX_DIR)/src && python3 simple_matrix.py local \
		--mode rapids \
		--overwrite yes \
		--minimal


# ── 2. DOCKER LOCAL GPU  ──
# Requirements: Docker, NVIDIA Container Toolkit, local GPU

DOCKER_SPARK_LOCAL := $(DOCKER_HUB_USERNAME)/spark_simple_matrix:local

docker-build-push-spark-local:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f $(SPARK_MATRIX_DIR)/Dockerfile.local \
		-t $(DOCKER_SPARK_LOCAL) \
		$(SPARK_MATRIX_DIR)
	
run-spark-docker-cpu:
	docker run --rm \
		--name spark-simple-matrix-$(USER) \
		--network host \
		-v ./$(SPARK_MATRIX_DIR)/src:/app/src \
		-v ./$(SPARK_MATRIX_DIR)/results:/app/results \
		-v ./$(SPARK_MATRIX_DIR)/jars:/app/jars \
		-e PYSPARK_PYTHON=python3 \
		-e PYSPARK_DRIVER_PYTHON=python3 \
		--shm-size=8G \
		$(DOCKER_SPARK_LOCAL) --mode cpu --overwrite yes

run-spark-docker-rapids:
	docker run --rm \
		-it \
		--network host \
		--privileged \
		-v ./$(SPARK_MATRIX_DIR)/src:/app/src \
		-v ./$(SPARK_MATRIX_DIR)/results:/app/results \
		-v ./$(SPARK_MATRIX_DIR)/jars:/app/jars \
		-e PYSPARK_PYTHON=python3 \
		-e PYSPARK_DRIVER_PYTHON=python3 \
		--name spark-simple-matrix-$(USER) \
		--runtime=nvidia \
		--shm-size=8G \
		$(DOCKER_SPARK_LOCAL) --mode rapids --overwrite yes

stop-spark-simple-matrix:
	docker stop spark-simple-matrix-$(USER) || true
	docker stop spark-simple-matrix-gvirtus-$(USER) || true

# ── 3. DOCKER GVIRTUS (Docker with GVirtuS frontend, no local GPU needed) ──
# Requirements: Docker, GVirtuS backend running on remote host

DOCKER_SPARK_GVIRTUS := $(DOCKER_HUB_USERNAME)/spark_simple_matrix:gvirtus

docker-build-push-spark-gvirtus:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f $(SPARK_MATRIX_DIR)/Dockerfile.gvirtus \
		-t $(DOCKER_SPARK_GVIRTUS) \
		.

run-spark-gvirtus:
	docker run --rm \
		--name spark-simple-matrix-gvirtus-$(USER) \
		--network host \
		-v ./$(SPARK_MATRIX_DIR)/src:/app/src \
		-v ./$(SPARK_MATRIX_DIR)/results:/app/results \
		-v ./$(SPARK_MATRIX_DIR)/jars:/app/jars \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json \
		-e PYSPARK_PYTHON=python3 \
		-e PYSPARK_DRIVER_PYTHON=python3 \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL)  \
		--shm-size=8G \
		$(DOCKER_SPARK_GVIRTUS) --mode rapids --overwrite yes

# Quick GVirtuS connectivity test (no Spark, just cudaGetDeviceCount etc.)
test-spark-gvirtus:
	docker run --rm \
		--name spark-gvirtus-test-$(USER) \
		--network host \
		-v ./$(SPARK_MATRIX_DIR)/src:/app/src \
		-v ./$(SPARK_MATRIX_DIR)/jars:/app/jars \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		$(DOCKER_SPARK_GVIRTUS) --test
