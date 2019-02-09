#/bin/sh
condadir=`dirname $0`
condadir=`readlink -f $condadir`
srcdir=`dirname $condadir`

docker build -t tvm-cuda100-forge $condadir -f $condadir/Dockerfile.cuda100
docker run --rm -v $srcdir:/workspace tvm-cuda100-forge
docker build -t tvm-cuda92-forge $condadir -f $condadir/Dockerfile.cuda92
docker run --rm -v $srcdir:/workspace tvm-cuda92-forge
sudo chown -R `whoami` $condadir/pkg
