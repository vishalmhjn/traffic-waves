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

test:	
	# python scopuscaller/test_call_scopus.py

run:
	source $(VENV) && cd src/ && \
    $(PYTHON) call_data_api.py && \
    $(PYTHON) process_data.py && \
    $(PYTHON) train.py && \
    $(PYTHON) predict.py && \
    $(PYTHON) app.py