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

# Minimum docker image for demo purposes
# prebuilt-image: tvmai/demo-cpu
FROM ubuntu:16.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

# Python: basic dependencies
RUN apt-get update && apt-get install -y python3-dev python3-pip
RUN pip3 install numpy nose-timer cython decorator scipy

# LLVM
RUN echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main \
     >> /etc/apt/sources.list.d/llvm.list && \
     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
     apt-get update && apt-get install -y --force-yes llvm-6.0

# Jupyter notebook.
RUN pip3 install matplotlib Image Pillow jupyter[notebook]

# Deep learning frameworks
RUN pip3 install mxnet tensorflow keras gluoncv

# Build TVM
COPY install/install_tvm_cpu.sh /install/install_tvm_cpu.sh
RUN bash /install/install_tvm_cpu.sh

# Environment variables
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:/usr/tvm/nnvm/python/:/usr/tvm/vta/python:${PYTHONPATH}
