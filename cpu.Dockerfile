# numpy.distutils is deprecated in NumPy 1.23.0 and will remain in Python <= 3.11 until Oct 2025
FROM python:3.11.11-bookworm
WORKDIR /app

COPY . ./

RUN apt-get update &&\
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu &&\
    pip install --no-cache-dir chamferdist faiss-cpu pandas tqdm
