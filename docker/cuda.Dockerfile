# numpy.distutils is deprecated in NumPy 1.23.0 and will remain in Python <= 3.11 until Oct 2025
FROM python:3.11.11-bookworm
WORKDIR /app

COPY . ./

ENV FORCE_CUDA=1

RUN apt-get update &&\
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb &&\
    dpkg -i cuda-keyring_1.1-1_all.deb &&\
    apt-get update &&\
    apt-get -y install cuda-toolkit-12-8 &&\
    apt-get -y install cudnn &&\
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 &&\
    pip install --no-cache-dir faiss-cpu pandas tqdm