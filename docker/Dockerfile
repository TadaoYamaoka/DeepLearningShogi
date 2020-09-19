FROM nvcr.io/nvidia/tensorrt:20.03-py3

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
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update -y  && apt-get install -y python3.7 libpython3.7 python3.7-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.7 get-pip.py && rm -f get-pip.py

RUN pip3 install torch attrs==19.1.0 json-log-plots==0.0.1 fire==0.1.3 matplotlib==3.0.3 numpy==1.16.2 tqdm==4.31.1 && \
    rm -r ~/.cache/pip

RUN ln -f -n -s /usr/bin/python3.7 /usr/bin/python
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update -y  && apt-get install -y --no-install-recommends python3-tk && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y libboost-all-dev libboost-dev libboost-python-dev libboost-numpy-dev zlib1g-dev python-dev

RUN rm -rf /var/lib/apt/lists/* &&\
    rm /tmp/* -rf
