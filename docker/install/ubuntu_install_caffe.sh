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

apt-get update --fix-missing

# Install dependencies
apt-get install -y --no-install-recommends libboost-filesystem-dev libboost-python-dev \
    libboost-system-dev libboost-thread-dev libboost-regex-dev protobuf-compiler \
    libprotobuf-dev libhdf5-serial-dev libopenblas-dev libgflags-dev libgoogle-glog-dev
rm -rf /var/lib/apt/lists/*

# install python packages
pip install "numpy>=1.7.1"
pip install "protobuf>=2.5.0"
pip install "scikit-image>=0.9.3"
pip install "six>=1.1.0"

# Build the Caffe and the python wrapper
echo "Downloading Caffe"
CAFFE_HOME="/opt/caffe"
git clone --branch=ssd --depth 1 https://github.com/weiliu89/caffe /caffe_src
cd /caffe_src

echo "Building Caffe"
mkdir /caffe_src/build && cd /caffe_src/build
cmake .. -DCMAKE_INSTALL_PREFIX=${CAFFE_HOME} -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=1 \
    -Dpython_version=3 -DUSE_OPENCV=OFF -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF -DBUILD_docs=OFF -DBLAS=open
make all -j`nproc`
make pycaffe -j`nproc`
make test -j`nproc`
make runtest -j`nproc`
make pytest -j`nproc`

echo "Installing Caffe to /opt/caffe"
make install

echo "Removing build directory"
cd / && rm -rf /caffe_src

PYCAFFE_ROOT=${CAFFE_HOME}/python
echo "${CAFFE_HOME}/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
ln -s ${PYCAFFE_ROOT}/caffe /usr/local/lib/python3.6/dist-packages/caffe
