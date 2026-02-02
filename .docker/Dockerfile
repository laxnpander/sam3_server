FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    tar \
    curl \
    zip \
    unzip \
    pkg-config \
    git \
    cmake \
    m4 \
    ninja-build \
    gfortran \
    gperf \
    autoconf \
    autoconf-archive \
    automake \
    libtool \
    pkg-config \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-jinja2 \
    python3.12-venv \
    python3-msgpack \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV

RUN pip3 install fastapi \
    uvicorn \
    einops \
    decord \
    pycocotools \
    psutil \
    opencv-python-headless


WORKDIR /opt
RUN git clone https://github.com/laxnpander/sam3.git \
    && cd sam3 && pip3 install .

RUN mkdir /opt/weights
RUN curl -L "https://cloud.tu-braunschweig.de/s/b7STzAn7roz3fS7/download" -o /opt/weights/sam3.pt

WORKDIR /workspace
COPY . .

#RUN pip3 install matplotlib pandas scikit-image scikit-learn

RUN pip3 install -e .

WORKDIR /workspace/src
ENV PYTHONPATH=/workspace/src
CMD ["python", "sam3_server/server.py"]