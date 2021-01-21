# get the base image from the hub
FROM nvidia/cuda:11.0-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin/:${PATH}"
ARG PATH="/root/miniconda3/bin/:${PATH}"

RUN apt-get update  && \
    apt-get install -y python3-dev python-pip htop wget curl unzip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh

RUN mkdir -p root/tokenizers/save root/tokenizers/data

RUN conda create -y -n ml python=3.8

#RUN curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip \
#    -o /root/tokenizers/data/wikitext-103-raw-v1.zip &&\
#    cd /root/tokenizers/data && unzip wikitext-103-raw-v1.zip && rm -rf wikitext-103-raw-v1.zip


COPY . /tokenizers/
RUN cd 
RUN source activate ml && \
    pip install -r requirements.txt

WORKDIR /tokenizers

RUN nvidia-smi && python build_tokenizer.py --save_dir /tokenizers/save --data_dir /tokenizers/data