# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install build tools, including `make`
RUN apt-get update && apt-get install -y build-essential make && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project directory into the container at /app
COPY . /app/

# Expose the port the app runs on (5000 for Flask)
EXPOSE 5000
