# Define variables
PYTHON := python3
ENV := .env
VENV := ${ENV}/bin/activate

# Check if running in a Docker container
DOCKER_ENV := $(shell [ -f /.dockerenv ] && echo 1 || echo 0)

install:
ifeq ($(DOCKER_ENV), 0)
	$(PYTHON) -m venv ${ENV} && \
	. ${VENV} && \
	pip install -r requirements.txt
else
	pip install -r requirements.txt
endif

run:
ifeq ($(DOCKER_ENV), 0)
	source $(VENV) && \
	cd src/ && $(PYTHON) main.py -m knn
else
	cd src/ && $(PYTHON) main.py -m knn
endif

app:
ifeq ($(DOCKER_ENV), 0)
	source $(VENV) && \
	cd src/ && $(PYTHON) app.py
else
	cd src/ && $(PYTHON) app.py
endif
