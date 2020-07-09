#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
set -o pipefail

# Prerequisite
apt-get update --fix-missing

export DEBIAN_FRONTEND=noninteractive
apt-get install -y tzdata
ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
dpkg-reconfigure --frontend noninteractive tzdata
 
apt-get install libprotobuf-dev -y
apt-get install libleveldb-dev -y
apt-get install libsnappy-dev -y
apt-get install libopencv-dev -y
apt-get install libhdf5-serial-dev -y
apt-get install protobuf-compiler -y
apt-get install libgflags-dev -y
apt-get install libgoogle-glog-dev -y
apt-get install liblmdb-dev -y
apt-get install libatlas-base-dev -y
apt-get install --no-install-recommends libboost-all-dev -y
apt-get install gfortran -y

cd /
mkdir caffe
cd caffe
wget  https://github.com/weiliu89/caffe/archive/ssd.zip -O ssd.zip
unzip ssd.zip
rm ssd.zip
cd caffe-ssd

echo "CPU_ONLY := 1" >> Makefile.config
echo "OPENCV_VERSION := 3" >> Makefile.config
echo "BLAS := open" >> Makefile.config
echo "PYTHON_LIBRARIES := boost_python3 python3.6m" >> Makefile.config
echo "PYTHON_INCLUDE := /usr/include/python3.6m /usr/lib/python3.6/dist-packages/numpy/core/include /usr/local/lib/python3.6/dist-packages/numpy/core/include" >> Makefile.config
echo "PYTHON_LIB := /usr/lib" >> Makefile.config
echo "WITH_PYTHON_LAYER := 1" >> Makefile.config
echo INCLUDE_DIRS := $\(PYTHON_INCLUDE\) /usr/local/include /usr/include/hdf5/serial >> Makefile.config
echo LIBRARY_DIRS := $\(PYTHON_LIB\) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial >> Makefile.config
echo "BUILD_DIR := build" >> Makefile.config
echo "DISTRIBUTE_DIR := distribute" >> Makefile.config
echo "Q ?= @" >> Makefile.config
 
 
make -j8
make test -j8
make runtest -j8
 
echo export PYTHONPATH=/caffe/caffe-ssd/python >> /etc/profile
cd ./python

rm requirements.txt
echo "Cython>=0.19.2" >> requirements.txt
echo "numpy>=1.7.1" >> requirements.txt
echo "scipy>=0.13.2" >> requirements.txt
echo "scikit-image>=0.9.3" >> requirements.txt
echo "h5py>=2.2.0" >> requirements.txt
echo "leveldb>=0.191" >> requirements.txt
echo "networkx>=1.8.1" >> requirements.txt
echo "nose>=1.3.0" >> requirements.txt
echo "pandas>=0.12.0" >> requirements.txt
echo "python-dateutil>=2.6.0" >> requirements.txt
echo "protobuf>=2.5.0" >> requirements.txt
echo "python-gflags>=2.0" >> requirements.txt
echo "pyyaml>=3.10" >> requirements.txt
echo "Pillow>=2.3.0" >> requirements.txt
echo "six>=1.1.0" >> requirements.txt

for req in $(cat requirements.txt); do pip3 install $req -i https://pypi.tuna.tsinghua.edu.cn/simple some-package; done

cd ..
make pycaffe
