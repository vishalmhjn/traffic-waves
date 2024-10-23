# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copy the application files
COPY . .

RUN mkdir ./predictions
RUN mkdir ./model_output

# Expose the application port
EXPOSE 5000

WORKDIR /app/src

CMD ["sh", "-c", "python main.py -t -m knn && python app.py"]
