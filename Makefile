# Define variables
PYTHON := python3
ENV := .env
VENV := ${ENV}/bin/activate

install:
	$(PYTHON) -m venv ${ENV} && \
	. ${VENV} && \
	pip install -r requirements.txt

lint:
	pylint --disable=R,C ./src

format:
	black src/*.py

run:
	source $(VENV) && cd src/ && \
    $(PYTHON) main.py -m knn -t

app:
	source $(VENV) && cd src/ && \
    $(PYTHON) app.py
