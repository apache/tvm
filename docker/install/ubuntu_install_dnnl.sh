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

pre_dir=`pwd`
tmpdir=$(mktemp -d)

rls_tag="v2.6"

dnnl_ver=`echo ${rls_tag} | sed 's/v//g'`
echo "Using oneDNN release version ${dnnl_ver} with tag '${rls_tag}'"

archive_name="${rls_tag}.tar.gz"
archive_url="https://github.com/oneapi-src/oneDNN/archive/refs/tags/${archive_name}"
archive_folder="${tmpdir}/oneDNN-${dnnl_ver}"
archive_hash="4cb7b80bfe16920bc096e18e7d8caa56b9ab7a4dab2a091a230bcf562c09533392f4a4ccd4db22754a10293670efdea20382db0994dc47949005a4c77f14b64c"

cd "${tmpdir}"

curl -sL "${archive_url}" -o "${archive_name}"
echo "$archive_hash" ${archive_name} | sha512sum -c
tar xf "${archive_name}"

cd "${archive_folder}"
cmake . -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib
make -j"$(nproc)"
make install

cd ${pre_dir}
rm -rf "${tmpdir}"
