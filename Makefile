.PHONY: docker/build  docker/test tests

DOCKER_IMAGE_TAG=movie_classification


docker/build:
	docker build -t ${DOCKER_IMAGE_TAG} .

docker/tests:
	docker run --rm --entrypoint '' ${DOCKER_IMAGE_TAG} python3.8 -m pytest tests -s -vv -ra

tests:
	python -m pytest tests -s -vv -ra