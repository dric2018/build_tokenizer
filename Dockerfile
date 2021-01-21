FROM nvidia/cuda:11.1.1-base-ubuntu18.04

ENV PATH = "/root/miniconda3/bin:${PATH}"
ENV PATH = "/root/miniconda3/bin:${PATH}"

RUN apt-get update  && \
    apt-get install -y python3-dev python-pip htop wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    sh Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN pip install tokenizers
RUN mkdir -p root/tokenizers/save

WORKDIR /tokenizers

RUN wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip &&\
    unzip wikitext-103-raw-v1.zip

COPY build_tokenizer.py /tokenizers/build_tokenizer.py

#RUN python build_tokenizer --save_dir /tokenizers/save