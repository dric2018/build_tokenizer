# get the base image from the hub
FROM nvidia/cuda:11.0-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update  && \
    apt-get install -y python3-dev python-pip htop wget unzip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    sh Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN /bin/bash -c "pip install tokenizers"
RUN mkdir -p root/tokenizers/save

RUN conda create -y -n ml python=3.8

WORKDIR /tokenizers

RUN wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip &&\
    unzip wikitext-103-raw-v1.zip


COPY . /tokenizers/

RUN /bin/bash -c "conda activate ml && pip install -r requirements.txt"

#RUN python build_tokenizer --save_dir /tokenizers/save