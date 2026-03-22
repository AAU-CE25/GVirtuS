.PHONY: docker-build-dev-local run-gvirtus-backend-dev-local docker-build-openpose-local run-openpose-test-local docker-build-openpose-overlay run-openpose-test-overlay docker-build-push-dev docker-build-push-prod run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-openpose run-openpose-test stop-openpose-test docker-build-2d-human-parsing run-2d-human-parsing-test stop-human-parsing-test docker-build-simple-matrix run-simple-matrix-test

# Backend local

docker-build-dev-local:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t gvirtus_backend \
		.

run-gvirtus-backend-dev-local:
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
		--name gvirtus \
		--runtime=nvidia \
		--shm-size=8G \
		gvirtus_backend

# Frontend local

docker-build-openpose-local:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		-f examples/openpose/Dockerfile-local \
		-t openpose_local \
		examples/openpose	

run-openpose-test-local: 
	docker run --rm \
		--name openpose_container \
		--network host \
		--privileged \
		-v ./examples/openpose/media:/opt/openpose/examples/media \
		-v ./examples/openpose:/opt/openpose/examples/gvirtus \
		-v ./examples/openpose/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/openpose/entrypoint.sh:/entrypoint.sh \
		openpose_local \
		bash /entrypoint.sh

docker-build-openpose-overlay:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		-f examples/openpose/Dockerfile-overlay \
		-t openpose_overlay \
		.

run-openpose-test-overlay:
	docker run --rm \
		--name openpose_container \
		--network host \
		--privileged \
		-v ./examples/openpose/media:/opt/openpose/examples/media \
		-v ./examples/openpose:/opt/openpose/examples/gvirtus \
		-v ./examples/openpose/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/openpose/entrypoint.sh:/entrypoint.sh \
		openpose_overlay \
		bash /entrypoint.sh


################################


docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-push-prod:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/prod/Dockerfile \
		-t taslanidis/gvirtus:cuda12.6.3-cudnn-ubuntu22.04 \
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
		--name gvirtus \
		--runtime=nvidia \
		--shm-size=8G \
		tinghui8576/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04

attach-gvirtus-bash:
		docker exec -it gvirtus bash

run-gvirtus-tests:
	docker exec \
		-it gvirtus \
		bash -c \
		'export LD_LIBRARY_PATH=$$GVIRTUS_HOME/lib/frontend:$$LD_LIBRARY_PATH && \
			cd /gvirtus/build && \
			ctest --output-on-failure'

stop-gvirtus:
	docker stop gvirtus


docker-build-openpose:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f examples/openpose/Dockerfile \
		-t darsh916/openpose_gvirtus:cuda12.6 \
		examples/openpose	


run-openpose-test: 
	docker run --rm \
		--name openpose_container \
		--network host \
		-v ./examples/openpose/media:/opt/openpose/examples/media \
		-v ./examples/openpose:/opt/openpose/examples/gvirtus \
		-v ./examples/openpose/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/openpose/entrypoint.sh:/entrypoint.sh \
		darsh916/openpose_gvirtus:cuda12.6 \
		bash /entrypoint.sh

stop-openpose-test:
	docker stop openpose_test_container || true



docker-build-2d-human-parsing:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f examples/2d-human-parsing/Dockerfile \
		-t darsh916/human-parsing_gvirtus:cuda12.6 \
		examples/2d-human-parsing	


run-2d-human-parsing-test: 
	docker run --rm \
		--name human_parsing_test_container \
		--network host \
		--shm-size=8G \
		-v ./examples/2d-human-parsing/inference_acc_00.py:/opt/2D-Human-Parsing/inference/inference_acc_00.py \
		-v ./examples/2d-human-parsing/demo_imgs:/opt/2D-Human-Parsing/demo_imgs \
		-v ./examples/2d-human-parsing/properties.json:/opt/GVirtuS/etc/properties.json \
		-v ./examples/2d-human-parsing/entrypoint.sh:/entrypoint.sh \
		darsh916/human-parsing_gvirtus:cuda12.6 \
		bash /entrypoint.sh

stop-2d-human-parsing-test:
	docker stop human-parsing_test_container || true


docker-build-simple-matrix:
	docker buildx build \
		--platform linux/amd64 \
		--load \
		-f examples/simple_matrix/Dockerfile \
		-t simple_matrix_gvirtus_local:cuda12.6 \
		.

run-simple-matrix-test:
	docker run --rm \
		--name simple_matrix_test_container \
		--network host \
		--privileged \
		--device /dev/infiniband \
		-v ./etc/properties.json:/opt/GVirtuS/etc/properties.json:ro \
		simple_matrix_gvirtus_local:cuda12.6 \
		bash -lc 'set -e; \
			export GVIRTUS_HOME=/opt/GVirtuS; \
			export GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties.json; \
			export LD_LIBRARY_PATH=/opt/GVirtuS/lib:/opt/GVirtuS/lib/frontend:$$LD_LIBRARY_PATH; \
			cd /opt/GVirtuS/examples/simple_matrix; \
			nvcc simple_matrix.cu -o simple_matrix -L/opt/GVirtuS/lib/frontend -L/opt/GVirtuS/lib -lcuda -lcudart -lcublas; \
			./simple_matrix'
