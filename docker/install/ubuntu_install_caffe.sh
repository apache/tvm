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

set -euxo pipefail

if [ -z "${TVM_VENV+x}" ]; then
    echo "ERROR: expect TVM_VENV env var to be set"
    exit 2
fi

apt-get update --fix-missing

# # Install dependencies
apt-install-and-clear -y --no-install-recommends protobuf-compiler \
    libprotobuf-dev libhdf5-serial-dev libopenblas-dev libgflags-dev libgoogle-glog-dev


# install python packages
pip install "numpy" "protobuf" "scikit-image" "six"

# Build the Caffe and the python wrapper
echo "Downloading Caffe"
CAFFE_HOME="/opt/caffe"
git clone --branch=ssd --depth 1 https://github.com/weiliu89/caffe /caffe_src
cd /caffe_src


echo "Building Caffe"
mkdir /caffe_src/build && cd /caffe_src/build
cmake -DCMAKE_INSTALL_PREFIX=${CAFFE_HOME}\
    -DCMAKE_BUILD_TYPE=Release \
    -DCPU_ONLY=1 \
    -Dpython_version=3 \
    -DUSE_OPENCV=OFF \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=OFF \
    -DBUILD_docs=OFF \
    -DBLAS=open \
    ..

make all -j$(expr $(nproc) - 1)
make pycaffe -j$(expr $(nproc) - 1)
make test -j$(expr $(nproc) - 1)

echo "Installing Caffe to /opt/caffe"
make install

echo "Removing build directory"
cd / && rm -rf /caffe_src

PYCAFFE_ROOT=${CAFFE_HOME}/python
echo "${CAFFE_HOME}/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
site_packages=$("${TVM_VENV}/bin/python3" -c 'import site; print(site.getsitepackages()[0])')
ln -s ${PYCAFFE_ROOT}/caffe "${site_packages}/caffe"
