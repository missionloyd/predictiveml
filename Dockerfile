# Use the official Python base image
# FROM python:3.11-slim-buster
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy the main.py file to the working directory
COPY main.py /app/main.py

# Create logging folders inside the container
RUN mkdir /app/main_log \
    && mkdir /app/error_log \
    && mkdir /app/flask_log

# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/requirements.txt

## Copy the env file to the working directory
# COPY ./env_config.yml /tmp/env_config.yml

# Install dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl gnupg libtinfo6 tzdata nvidia-container-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA Docker
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && apt-get update && apt-get install -y nvidia-docker2

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh && \
    bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
    
## ADD CONDA PATH TO LINUX PATH 
ENV PATH /opt/conda/bin:$PATH

# ## COPY ENV REQUIREMENTS FILES
COPY ./env_config.yml /tmp/env_config.yml

# # Install Conda dependencies
# RUN conda install --yes -c conda-forge prophet && \
#     conda install --yes numpy pandas matplotlib flask

## CREATE CONDA ENVIRONMENT USING YML FILE
RUN conda update conda \
    && conda env create -f /tmp/env_config.yml

## ADD CONDA ENV PATH TO LINUX PATH 
ENV PATH /opt/conda/envs/tf-gpu/bin:$PATH
ENV CONDA_DEFAULT_ENV tf-gpu
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib/:$CUDNN_PATH/lib"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

## MAKE ALL BELOW RUN COMMANDS USE THE NEW CONDA ENVIRONMENT
SHELL ["conda", "run", "-n", "tf-gpu", "/bin/bash", "-c"]

# Install nvidia-cudnn-cu11
RUN python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*

# Copy the Flask script to the working directory
COPY app.py /app/app.py

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy the activation script to the container
COPY env_vars.sh /etc/conda/activate.d/env_vars.sh

# Activate the environment
RUN chmod +x /etc/conda/activate.d/env_vars.sh \
    && echo "source /etc/conda/activate.d/env_vars.sh" >> /root/.bashrc

# Copy the remaining project files
COPY . .

# Create empty log files
RUN touch /app/main_log/0.log \
    && touch /app/error_log/0.log \
    && touch /app/flask_log/0.log

# run the crontab file
# RUN apt-get -y install cron vim
# COPY crontab /etc/cron.d/crontab
# RUN chmod 0644 /etc/cron.d/crontab
# RUN /usr/bin/crontab /etc/cron.d/crontab
# RUN echo $PYTHONPATH

EXPOSE 8080

# run the main.py script, and start the Flask server
# CMD python3 main.py --preprocess >> /app/main_log/0.log 2>> /app/error_log/0.log & flask run --host=0.0.0.0 --port=8080 >> /app/flask_log.log 2>&1
CMD python3 -m flask run --host=0.0.0.0 --port=8080 > /app/flask_log/0.log 2>&1
# CMD ["cron", "-f"]