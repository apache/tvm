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
# Installation

TVM FFI is built and tested on Windows, macOS, and various
Linux distributions. You can install tvm-ffi using one of the
methods below

## Quick Start

The easiest way to try it out is to install from PyPI.

```bash
pip install apache-tvm-ffi
```

After installation, you can run the following command to confirm that
the installation was successful

```bash
tvm-ffi-config -h
```

This configuration tool is also useful in various ways to help you build
libraries with tvm-ffi.


## Install From Source

You can also build and install tvm-ffi from source.

### Dependencies

- CMake (>= 3.24.0)
- Git
- A recent C++ compiler supporting C++17, at minimum:
    - GCC 7.1
    - Clang 5.0
    - Apple Clang 9.3
    - Visual Studio 2019 (v16.7)
- Python (>= 3.9)


Developers can clone the source repository from GitHub.

```bash
git clone --recursive https://github.com/apache/tvm tvm
```

```{note}
It's important to use the ``--recursive`` flag when cloning the repository, which will
automatically clone the submodules. If you forget to use this flag, you can manually clone the submodules
by running ``git submodule update --init --recursive`` in the root directory.
```

Then you can install directly in development mode

```bash
cd tvm/ffi
pip install -ve .
```

The additional `-e` flag will install the Python files in `editable` mode,
which allows direct editing of the Python files to be immediately reflected in the package
and is useful for development.

## What to Do Next

Now that you have installed TVM FFI, we recommend reading the [Quick Start](./quick_start.md) tutorial.
