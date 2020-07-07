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

# Copyright (c) 2020 T-Head Semiconductor Co., Ltd. All rights reserved.

set -e
set -u
set -o pipefail

# Prerequisite
apt update
apt install python3-dev
apt install python3-pip
pip3 install numpy
apt-get install libprotobuf-dev
apt-get install libleveldb-dev
apt-get install libsnappy-dev
apt-get install libopencv-dev
apt-get install libhdf5-serial-dev
apt-get install protobuf-compiler
apt-get install libgflags-dev
apt-get install libgoogle-glog-dev
apt-get install liblmdb-dev
apt-get install libatlas-base-dev
apt-get install --no-install-recommends libboost-all-dev
apt-get install gfortran

# Download and build SSD-Caffe
cd ~
git clone https://github.com/weiliu89/caffe.git
cd caffe
git checkout ssd

echo "CPU_ONLY := 1" >> Makefile.config
echo "OPENCV_VERSION := 3" >> Makefile.config
echo "BLAS := open" >> Makefile.config
echo "PYTHON_LIBRARIES := boost_python3 python3.6m" >> Makefile.config
echo "PYTHON_INCLUDE := /usr/include/python3.6m /usr/lib/python3.6/dist-packages/numpy/core/include /usr/local/lib/python3.6/dist-packages/numpy/core/include" >> Makefile.config
echo "PYTHON_LIB := /usr/lib" >> Makefile.config
echo "WITH_PYTHON_LAYER := 1" >> Makefile.config
echo "INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial" >> Makefile.config
echo "LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial" >> Makefile.config
echo "BUILD_DIR := build" >> Makefile.config
echo "DISTRIBUTE_DIR := distribute" >> Makefile.config
echo "Q ?= @" >> Makefile.config


make -j8
make test -j8
make runtest -j8

echo "export PYTHONPATH=~/caffe/python:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
cd caffe/python
for req in $(cat requirements.txt); do sudo pip3 install $req; done
make pycaffe
