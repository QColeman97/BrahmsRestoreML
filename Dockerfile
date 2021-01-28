FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Quinn Coleman

WORKDIR /home/docker_thesis/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
