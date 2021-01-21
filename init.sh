echo "====Building image build_tokenizer ===="
docker build -f Dockerfile -t build_tokenizer .

echo "====Runing a container from the new build_tokenizer image ====\n"
docker run -v $(pwd):/root/tokenizers -ti --gpus 1 build_tokenizer /bin/bash
