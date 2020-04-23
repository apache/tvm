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

# Download, build and install flatbuffers
git clone --branch=v1.12.0 --depth=1 --recursive https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make install -j8
cd ..
rm -rf flatbuffers

# Install flatbuffers python packages.
pip3 install flatbuffers
pip2 install flatbuffers

# Setup tflite from schema
mkdir tflite
cd tflite
wget -q https://raw.githubusercontent.com/tensorflow/tensorflow/r2.1/tensorflow/lite/schema/schema.fbs
flatc --python schema.fbs

cat <<EOM >setup.py
import setuptools

setuptools.setup(
    name="tflite",
    version="2.1.0",
    author="google",
    author_email="google@google.com",
    description="TFLite",
    long_description="TFLite",
    long_description_content_type="text/markdown",
    url="https://www.tensorflow.org/lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
EOM

cat <<EOM >__init__.py
name = "tflite"
EOM

# Install tflite over python2 and python3
python3 setup.py install
python2 setup.py install

cd ..
rm -rf tflite
