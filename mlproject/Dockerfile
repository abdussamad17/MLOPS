# syntax=docker/dockerfile:1

FROM python:3.11.10

WORKDIR /mlproject
ARG MLFLOW_TRACKING_URI

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y \
    libev-dev \
    libsm6 \
    libxext6

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Install supervisor
RUN apt-get install -y supervisor

# Create supervisor configuration directory
RUN mkdir -p /etc/supervisor/conf.d

# Add supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose both ports
EXPOSE 9510 9610

# Change CMD to run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]