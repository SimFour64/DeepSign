# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
FROM python:3.10

#librairie libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

#      Slim version
# FROM python:3.10-slim

#      Tensorflow version (attention: won't run on Apple Silicon)
# FROM tensorflow/tensorflow:2.16.1

# Copy and install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Installer google-cloud (utilisé pour GCP)
#RUN pip install --no-cache-dir google-cloud

COPY secrets/deepsign-454016-6c19001d719b.json /app/key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"

# Copy everything we need into the image
COPY deepsign deepsign
COPY api api
COPY models models
COPY setup.py setup.py
COPY params.py params.py
# COPY credentials.json credentials.json

# Install package
RUN pip install .

# Make directories that we need, but that are not included in the COPY
RUN mkdir /raw_data
# RUN mkdir /models

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'
EXPOSE 8080
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
