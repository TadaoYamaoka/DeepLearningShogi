ARG CUDA_VERSION=10.2
ARG UBUNTU_VERSION=18.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

MAINTAINER RyotoSawada <qhapaq49@yahoo.co.jp>

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    emacs \
    software-properties-common \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvparsers-dev \
    libnvonnxparsers-dev \
    libboost-dev \
    libboost-python-dev \
    libboost-numpy-dev \
    zlib1g-dev \
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update -y  && apt-get install -y python3.7 libpython3.7 python3.7-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.7 get-pip.py && rm -f get-pip.py

RUN pip3 install torch chainer && \
    rm -r ~/.cache/pip

RUN ln -f -n -s /usr/bin/python3.7 /usr/bin/python

ENV DEBIAN_FRONTEND=noninteractive

ARG uid=1000
ARG uname=shogi

RUN groupadd -g ${uid} ${uname}
RUN useradd -u ${uid} -g ${uid} -m -d /home/${uname} --create-home --shell /usr/bin/bash ${uname} && \
    echo "${uname} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R ${uname}:${uname} /home/${uname}

ENV HOME /home/${uid}
USER ${uid}
