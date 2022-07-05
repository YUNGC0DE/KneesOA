FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 as prod
ENV TZ="Europe/Moscow"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq tzdata --no-install-recommends \
    && apt-get install software-properties-common -y --no-install-recommends \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install -yq --no-install-recommends \
    git \
    cron \
    swig \
    wget \
    cmake \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m ensurepip --upgrade && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel awscli && \
    pip3 cache purge

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH /KneesOA/:$PYTHONPATH

COPY requirements/ requirements/
RUN pip3 install --no-cache-dir -r requirements/prod.txt
COPY setup.py .
COPY model_release/ model_release/
COPY KneesOA/ KneesOA/
RUN pip3 install --no-cache-dir -e .
WORKDIR KneesOA/app/