echo ${1:-cuda}
docker build --file=${1:-cuda}.Dockerfile -t kpconv .