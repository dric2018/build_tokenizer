# get the base image from the hub
FROM nvidia/cuda:11.0-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin/:${PATH}"
ARG PATH="/root/miniconda3/bin/:${PATH}"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update  && \
    apt-get install -yq apt-utils lsb-release dpkg apt-utils python3-dev python-pip htop wget unzip nano

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN mkdir -p root/tokenizers/save root/tokenizers/data

RUN conda create -y -n ml python=3.8

RUN wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip \
    -O /root/tokenizers/data &&\
    cd /root/tokenizers/data && unzip wikitext-103-raw-v1.zip


COPY . /tokenizers/

RUN cd && conda activate ml && \
    pip install -r requirements.txt

WORKDIR /tokenizers

#RUN python build_tokenizer --save_dir /tokenizers/save