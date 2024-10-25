# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd -r modeler && useradd -r -g modeler modeler

ENV HOME=/home/modeler

USER modeler

WORKDIR $HOME

# Install dependencies
COPY --chown=modeler:modeler requirements.txt .
RUN python -m pip install --user -r requirements.txt

# Copy the application files
COPY --chown=modeler:modeler . .

# Create folder for saving outputs
RUN mkdir ./predictions
RUN mkdir ./model_output

# Expose the application port
EXPOSE 5000

WORKDIR $HOME/src

CMD ["sh", "-c", "python main.py -t -m knn && python app.py"]
