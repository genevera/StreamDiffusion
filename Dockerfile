FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
        CUDA_HOME=/usr/local/cuda-12.6
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -yqq --no-install-recommends \
        make \
        wget \
        tar \
        build-essential \
        libgl1-mesa-dev \
        curl \
        unzip \
        git \
        python3.10-dev \
        libglib2.0-0 \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-12.6" >> /etc/bash.bashrc

RUN pip3 install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    huggingface-hub==0.25.2 \
    xformers \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY . /streamdiffusion
WORKDIR /streamdiffusion

RUN /usr/bin/python3.10 setup.py develop easy_install streamdiffusion[tensorrt] \
    && /usr/bin/python3.10 -m streamdiffusion.tools.install-tensorrt

WORKDIR /home/ubuntu/streamdiffusion

