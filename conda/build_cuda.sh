#/bin/sh
docker build -t tvm-cuda101-forge . -f Dockerfile.cuda100
docker run -v `pwd`/..:/workspace tvm-cuda100-forge
docker build -t tvm-cuda92-forge . -f Dockerfile.cuda92
docker run -v `pwd`/..:/workspace tvm-cuda92-forge
sudo chown -R `whoami` pkg
