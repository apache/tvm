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

old_dir=`pwd`

cd /usr/local/
current_dir=`pwd`

rls_tag=$(curl -s https://github.com/oneapi-src/oneDNN/releases/latest \
    | cut -d '"' -f 2 \
    | grep -o '[^\/]*$')
dnnl_ver=`echo ${rls_tag} | sed 's/v//g'`
echo "The latest oneDNN release is version ${dnnl_ver} with tag '${rls_tag}'"

tar_file="${rls_tag}.tar.gz"
src_dir="${current_dir}/oneDNN-${dnnl_ver}"

if [ -d ${src_dir} ]; then
    echo "source files exist."
else
    if [ -f ${tar_file} ]; then
        echo "${tar_file} exists, skip downloading."
    else 
        echo "downloading ${tar_file}."
        tar_url="https://github.com/oneapi-src/oneDNN/archive/refs/tags/${tar_file}"
        wget ${tar_url} ${tar_file}
    fi
    tar -xzvf ${tar_file}
fi

install_dir="/usr/local/"
cd ${src_dir}
cmake . -DCMAKE_INSTALL_PREFIX=${install_dir} && make -j16 && make install

cd ${old_dir}
