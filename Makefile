.PHONY: docker-build-push-dev docker-build-push-prod docker-build-push-docker-test run-docker-gvirtus-test stop-docker-gvirtus-test run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-2d-human-parsing-test docker-build-simple-matrix run-simple-matrix-test stop-simple-matrix-test run-spark-local-cpu run-spark-local-rapids docker-build-spark run-spark-docker-cpu run-spark-docker-rapids run-spark-gvirtus test-spark-gvirtus stop-spark-simple-matrix docker-build-frontend run-simple-matrix-frontend run-spark-frontend
USER := $(shell whoami | cut -d'@' -f1 | tr -d '.')
DOCKER_HUB_USERNAME ?= aauce25

GVIRTUS_LOG_LEVEL ?= 10000

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
# The switch between local GPU and GVirtuS happens at the Spark config level:
#   - Local GPU: JVM loads real CUDA libs from /usr/local/cuda/lib64
#   - GVirtuS:   JVM loads GVirtuS stubs via spark.executor.extraLibraryPath
# ═══════════════════════════════════════════════════════════════════════════════

SPARK_MATRIX_DIR := examples/spark_simple_matrix
DOCKER_SPARK := $(DOCKER_HUB_USERNAME)/spark_simple_matrix:latest
RAPIDS_JARS := ../jars
# ── Build the unified image ──
docker-build-spark:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f $(SPARK_MATRIX_DIR)/Dockerfile \
		-t $(DOCKER_SPARK) \
		.

# ── 1. LOCAL NATIVE (run directly on host, no Docker) ──
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


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend-only image (no GPU/CUDA runtime needed on client)
# ═══════════════════════════════════════════════════════════════════════════════

DOCKER_FRONTEND := $(DOCKER_HUB_USERNAME)/gvirtus-frontend:latest

docker-build-frontend:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/frontend/Dockerfile \
		-t $(DOCKER_FRONTEND) \
		.

run-simple-matrix-frontend:
	docker run --rm \
		--name simple-matrix-frontend-$(USER) \
		--network host \
		-v ./examples/simple_matrix:/opt/GVirtuS/examples \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		$(DOCKER_FRONTEND) \
		bash /opt/GVirtuS/examples/frontend.sh

DOCKER_SPARK_FRONTEND := $(DOCKER_HUB_USERNAME)/spark-gvirtus-frontend:latest

docker-build-spark-frontend:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f examples/spark_simple_matrix/Dockerfile.frontend \
		-t $(DOCKER_SPARK_FRONTEND) \
		examples/spark_simple_matrix

run-spark-frontend:
	mkdir -p examples/spark_simple_matrix/logs/frontend
	docker run --rm \
		--name spark-frontend-$(USER) \
		--network host \
		-v ./examples/spark_simple_matrix/src:/app/src \
		-v ./examples/spark_simple_matrix/results:/app/results \
		-v ./examples/spark_simple_matrix/logs/frontend:/app/logs \
		-v $$(pwd)/$(RAPIDS_JARS):/app/jars \
		-v ./examples/spark_simple_matrix/entrypoint.sh:/entrypoint.sh \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json \
		-e GVIRTUS_LOGLEVEL=$(GVIRTUS_LOG_LEVEL) \
		-e PYSPARK_PYTHON=python3 \
		-e PYSPARK_DRIVER_PYTHON=python3 \
		--shm-size=8G \
		$(DOCKER_SPARK_FRONTEND) \
		gvirtus --mode rapids --overwrite yes --minimal
