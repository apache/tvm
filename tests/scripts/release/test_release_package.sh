#!/usr/bin/env bash
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
set -exu

######################################################
# Write current test version and rc number here
######################################################
# NOTE about "rc":
# 1. Required for test candidate, such as "rc0"
# 2. Not required for release, leave blank ""
######################################################
version="v0.16.0"
rc="rc0"

######################################################
# This script is used test release (cancdidate)
# packages uploading to apache.org, github.com
# before release vote starts or release.
#
# The release (candidate) package contains files
# below:
#    1. apache-tvm-src-${version_rc}.tar.gz.asc
#    2. apache-tvm-src-${version_rc}.tar.gz.sha512
#    3. apache-tvm-src-${version_rc}.tar.gz
######################################################
version_rc="${version}"
apache_prefix="${version}"
if [ "$rc" != "" ]; then
    apache_prefix="${version_rc}-${rc}"
    version_rc="${version_rc}.${rc}"
fi
mkdir test_tvm_${version_rc}
cd test_tvm_${version_rc}

echo "[1/9] Downloading from apache.org ..."
mkdir apache
cd apache
wget -c https://dist.apache.org/repos/dist/dev/tvm/tvm-${apache_prefix}/apache-tvm-src-${version_rc}.tar.gz.sha512
wget -c https://dist.apache.org/repos/dist/dev/tvm/tvm-${apache_prefix}/apache-tvm-src-${version_rc}.tar.gz.asc
wget -c https://dist.apache.org/repos/dist/dev/tvm/tvm-${apache_prefix}/apache-tvm-src-${version_rc}.tar.gz
md5sum ./* > ./md5sum.txt
cd -

echo "[2/9] Downloading from github.com ..."
mkdir github
cd github
wget -c https://github.com/apache/tvm/releases/download/${version_rc}/apache-tvm-src-${version_rc}.tar.gz.sha512
wget -c https://github.com/apache/tvm/releases/download/${version_rc}/apache-tvm-src-${version_rc}.tar.gz.asc
wget -c https://github.com/apache/tvm/releases/download/${version_rc}/apache-tvm-src-${version_rc}.tar.gz
md5sum ./* > ./md5sum.txt
cd -

echo "[3/9] Check difference between github.com and apache.org ..."
diff github/md5sum.txt ./apache/md5sum.txt

echo "[4/9] Checking asc ..."
cd github
gpg --verify ./apache-tvm-src-${version_rc}.tar.gz.asc ./apache-tvm-src-${version_rc}.tar.gz

echo "[5/9] Checking sha512 ..."
sha512sum -c ./apache-tvm-src-${version_rc}.tar.gz.sha512

echo "[6/9] Unzip ..."
tar -zxf apache-tvm-src-${version_rc}.tar.gz

echo "[7/9] Checking whether binary in source code ..."
output=`find apache-tvm-src-${version_rc} -type f -exec file {} + | grep -w "ELF\|shared object"`
if [[ -n "$output" ]]; then
    echo "Error: ELF or shared object files found:"
    echo "$output"
    exit 1
fi

echo "[8/9] Compile and Python Import on Linux ..."
cd apache-tvm-src-${version_rc}
mkdir build
cd build
cp ../cmake/config.cmake .
cmake ..
make -j4
cd ..

echo "[9/9] Import TVM and print path ..."
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
python3 -c "import tvm; print(tvm.__path__)"
