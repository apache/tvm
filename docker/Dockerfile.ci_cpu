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

# CI docker CPU env
# tag: v0.41
FROM ubuntu:16.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

COPY install/ubuntu_install_llvm.sh /install/ubuntu_install_llvm.sh
RUN bash /install/ubuntu_install_llvm.sh

# SGX deps (build early; changes infrequently)
COPY install/ubuntu_install_sgx.sh /install/ubuntu_install_sgx.sh
RUN bash /install/ubuntu_install_sgx.sh
ENV LD_LIBRARY_PATH /opt/sgxsdk/lib64:${LD_LIBRARY_PATH}

# Rust env (build early; takes a while)
COPY install/ubuntu_install_rust.sh /install/ubuntu_install_rust.sh
RUN bash /install/ubuntu_install_rust.sh
ENV RUSTUP_HOME /opt/rust
ENV CARGO_HOME /opt/rust
ENV RUSTC_WRAPPER sccache

# AutoTVM deps
COPY install/ubuntu_install_redis.sh /install/ubuntu_install_redis.sh
RUN bash /install/ubuntu_install_redis.sh

# Golang environment
COPY install/ubuntu_install_golang.sh /install/ubuntu_install_golang.sh
RUN bash /install/ubuntu_install_golang.sh

# NNPACK deps
COPY install/ubuntu_install_nnpack.sh /install/ubuntu_install_nnpack.sh
RUN bash /install/ubuntu_install_nnpack.sh

ENV PATH $PATH:$CARGO_HOME/bin:/usr/lib/go-1.10/bin


# ANTLR deps
COPY install/ubuntu_install_java.sh /install/ubuntu_install_java.sh
RUN bash /install/ubuntu_install_java.sh

COPY install/ubuntu_install_antlr.sh /install/ubuntu_install_antlr.sh
RUN bash /install/ubuntu_install_antlr.sh
