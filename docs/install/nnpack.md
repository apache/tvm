<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# NNPACK Contrib Installation

[NNPACK](https://github.com/Maratyszcza/NNPACK) is an acceleration package
for neural network computations, which can run on x86-64, ARMv7, or ARM64 architecture CPUs.
Using NNPACK, higher-level libraries like _MXNet_ can speed up
the execution on multi-core CPU computers, including laptops and mobile devices.

***Note***: AS TVM already has natively tuned schedules, NNPACK is here mainly for reference and comparison purpose.
For regular use prefer native tuned TVM implementation.

_TVM_ supports NNPACK for forward propagation (inference only) in convolution, max-pooling, and fully-connected layers.
In this document, we give a high level overview of how to use NNPACK with _TVM_.

## Conditions
The underlying implementation of NNPACK utilizes several acceleration methods,
including [fft](https://arxiv.org/abs/1312.5851) and [winograd](https://arxiv.org/abs/1509.09308).
These algorithms work better on some special `batch size`, `kernel size`, and `stride` settings than on other,
so depending on the context, not all convolution, max-pooling, or fully-connected layers can be powered by NNPACK.
When favorable conditions for running NNPACKS are not met,

NNPACK only supports Linux and OS X systems. Windows is not supported at present.

## Build/Install NNPACK

If the trained model meets some conditions of using NNPACK,
you can build TVM with NNPACK support.
Follow these simple steps:
* Build NNPACK shared library with the following commands. _TVM_ will link NNPACK dynamically.

Note: The following NNPACK installation instructions have been tested on Ubuntu 16.04.

### Build [Ninja](https://ninja-build.org/)

NNPACK need a recent version of Ninja. So we need to install ninja from source.
```bash
git clone git://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
```

Set the environment variable PATH to tell bash where to find the ninja executable. For example, assume we cloned ninja on the home directory ~. then we can added the following line in ~/.bashrc.
```bash
export PATH="${PATH}:~/ninja"
```

### Build [NNPACK](https://github.com/Maratyszcza/NNPACK)

The new CMAKE version of NNPACK download [Peach](https://github.com/Maratyszcza/PeachPy) and other dependencies alone

```bash
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
# Add PIC option in CFLAG and CXXFLAG to build NNPACK shared library
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt
mkdir build
cd build
# Generate ninja build rule and add shared library in configuration
cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install

# Add NNPACK lib folder in your ldconfig
echo "/usr/local/lib" > /etc/ld.so.conf.d/nnpack.conf
sudo ldconfig
```

## Build TVM with NNPACK support

```bash
git clone --recursive https://github.com/dmlc/tvm
```

* Set `set(USE_NNPACK ON)` in config.cmake.
* Set `NNPACK_PATH` to the $(YOUR_NNPACK_INSTALL_PATH)

after configuration use `make` to build TVM

```bash
make
```
