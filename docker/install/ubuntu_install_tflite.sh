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
set -x
set -o pipefail

# The tflite version should have matched versions to the tensorflow
# version installed from pip in ubuntu_install_tensorflow.sh
TENSORFLOW_VERSION=$(python3 -c "import tensorflow; print(tensorflow.__version__)" 2> /dev/null)

# Download, build and install flatbuffers
git clone --branch=v25.1.24 --depth=1 --recursive https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-class-memaccess"
make install -j8
cd ..

# Install flatbuffers python packages.
pip3 install flatbuffers

# Build the TFLite static library, necessary for building with TFLite ON.
# The library is built at:
# tensorflow/tensorflow/lite/tools/make/gen/*/lib/libtensorflow-lite.a.
git clone https://github.com/tensorflow/tensorflow --branch=v${TENSORFLOW_VERSION} --depth 1

mkdir -p /opt/tflite
cd /opt/tflite
cmake \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  /tensorflow/tensorflow/lite

cmake --build .
cd -


# Setup tflite from schema
mkdir tflite
if [ -f tensorflow/tensorflow/compiler/mlir/lite/schema/schema.fbs ] ; then
  cp tensorflow/tensorflow/compiler/mlir/lite/schema/schema.fbs tflite
else
  cp tensorflow/tensorflow/lite/schema/schema.fbs tflite
fi

cd tflite
flatc --python schema.fbs

cat <<EOM >setup.py
import setuptools

setuptools.setup(
    name="tflite",
    version="${TENSORFLOW_VERSION}",
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

# Install tflite over python3
python3 setup.py install

cd ..
rm -rf tflite
