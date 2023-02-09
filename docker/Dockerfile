FROM nvcr.io/nvidia/tensorrt:22.08-py3

ENV DEBIAN_FRONTEND noninteractive
RUN ln -s -f /bin/true /usr/bin/chfn
RUN echo "resolvconf resolvconf/linkify-resolvconf boolean false" | debconf-set-selections

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
    build-essential libopenblas-dev \
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 timm -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install cython scipy && \
    rm -r ~/.cache/pip
