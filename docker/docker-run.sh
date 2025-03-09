gpus=""
if [ "$1" = "cuda" ]; then
    gpus="--gpus=all"
fi
docker run --ipc=host $gpus -it -v "$(pwd):/app" labelpointnetae bash
