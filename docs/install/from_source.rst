..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _install-from-source:

Install from Source
===================
This page gives instructions on how to build and install the TVM package from source.

.. contents:: Table of Contents
    :local:
    :depth: 2

.. _install-dependencies:

Step 1. Install Dependencies
----------------------------

Apache TVM requires the following dependencies:

- CMake (>= 3.24.0)
- LLVM (recommended >= 15)
- Git
- A recent C++ compiler supporting C++ 17, at the minimum
    - GCC 7.1
    - Clang 5.0
    - Apple Clang 9.3
    - Visual Studio 2019 (v16.7)
- Python (>= 3.8)
- (Optional) Conda (Strongly Recommended)

To easiest way to manage dependency is via conda, which maintains a set of toolchains
including LLVM across platforms. To create the environment of those build dependencies,
one may simply use:

.. code:: bash

    # make sure to start with a fresh environment
    conda env remove -n tvm-build-venv
    # create the conda environment with build dependency
    conda create -n tvm-build-venv -c conda-forge \
        "llvmdev>=15" \
        "cmake>=3.24" \
        git \
        python=3.11
    # enter the build environment
    conda activate tvm-build-venv


Step 2. Get Source from Github
------------------------------
You can also choose to clone the source repo from github.

.. code:: bash

    git clone --recursive https://github.com/apache/tvm tvm

.. note::
    It's important to use the ``--recursive`` flag when cloning the TVM repository, which will
    automatically clone the submodules. If you forget to use this flag, you can manually clone the submodules
    by running ``git submodule update --init --recursive`` in the root directory of the TVM repository.

Step 3. Configure and Build
---------------------------
Create a build directory and run CMake to configure the build. The following example shows how to build

.. code:: bash

    cd tvm
    rm -rf build && mkdir build && cd build
    # Specify the build configuration via CMake options
    cp ../cmake/config.cmake .

We want to specifically tweak the following flags by appending them to the end of the configuration file:

.. code:: bash

    # controls default compilation flags (Candidates: Release, Debug, RelWithDebInfo)
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

    # LLVM is a must dependency for compiler end
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

    # GPU SDKs, turn on if needed
    echo "set(USE_CUDA   OFF)" >> config.cmake
    echo "set(USE_METAL  OFF)" >> config.cmake
    echo "set(USE_VULKAN OFF)" >> config.cmake
    echo "set(USE_OPENCL OFF)" >> config.cmake

    # cuBLAS, cuDNN, cutlass support, turn on if needed
    echo "set(USE_CUBLAS OFF)" >> config.cmake
    echo "set(USE_CUDNN  OFF)" >> config.cmake
    echo "set(USE_CUTLASS OFF)" >> config.cmake


.. note::
    ``HIDE_PRIVATE_SYMBOLS`` is a configuration option that enables the ``-fvisibility=hidden`` flag.
    This flag helps prevent potential symbol conflicts between TVM and PyTorch. These conflicts arise due to
    the frameworks shipping LLVMs of different versions.

    `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ controls default compilation flag:

    - ``Debug`` sets ``-O0 -g``
    - ``RelWithDebInfo`` sets ``-O2 -g -DNDEBUG`` (recommended)
    - ``Release`` sets ``-O3 -DNDEBUG``

Once ``config.cmake`` is edited accordingly, kick off build with the commands below:

.. code-block:: bash

    cmake .. && cmake --build . --parallel $(nproc)

.. note::
    ``nproc`` may not be available on all systems, please replace it with the number of cores on your system

A success build should produce ``libtvm`` and ``libtvm_runtime`` under ``build/`` directory.

Leaving the build environment ``tvm-build-venv``, there are two ways to install the successful build into your environment:

-  Install via environment variable

.. code-block:: bash

    export TVM_HOME=/path-to-tvm
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

- Install via pip local project

.. code-block:: bash

    conda activate your-own-env
    conda install python # make sure python is installed
    export TVM_LIBRARY_PATH=/path-to-tvm/build
    pip install -e /path-to-tvm/python

Step 4. Validate Installation
-----------------------------

Using a compiler infrastructure with multiple language bindings could be error-prone.
Therefore, it is highly recommended to validate Apache TVM installation before use.

**Step 1. Locate TVM Python package.** The following command can help confirm that TVM is properly installed as a python package and provide the location of the TVM python package:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.__file__)"
    /some-path/lib/python3.11/site-packages/tvm/__init__.py

**Step 2. Confirm which TVM library is used.** When maintaining multiple build or installation of TVM, it becomes important to double check if the python package is using the proper ``libtvm`` with the following command:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm._ffi.base._LIB)"
    <CDLL '/some-path/lib/python3.11/site-packages/tvm/libtvm.dylib', handle 95ada510 at 0x1030e4e50>

**Step 3. Reflect TVM build option.** Sometimes when downstream application fails, it could likely be some mistakes with a wrong TVM commit, or wrong build flags. To find it out, the following commands will be helpful:

.. code-block:: bash

    >>> python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
    ... # Omitted less relevant options
    GIT_COMMIT_HASH: 4f6289590252a1cf45a4dc37bce55a25043b8338
    HIDE_PRIVATE_SYMBOLS: ON
    USE_LLVM: llvm-config --link-static
    LLVM_VERSION: 15.0.7
    USE_VULKAN: OFF
    USE_CUDA: OFF
    CUDA_VERSION: NOT-FOUND
    USE_OPENCL: OFF
    USE_METAL: ON
    USE_ROCM: OFF


**Step 4. Check device detection.** Sometimes it could be helpful to understand if TVM could detect your device at all with the following commands:

.. code-block:: bash

    >>> python -c "import tvm; print(tvm.metal().exist)"
    True # or False
    >>> python -c "import tvm; print(tvm.cuda().exist)"
    False # or True
    >>> python -c "import tvm; print(tvm.vulkan().exist)"
    False # or True

Please note that the commands above verify the presence of an actual device on the local machine for the TVM runtime (not the compiler) to execute properly. However, TVM compiler can perform compilation tasks without requiring a physical device. As long as the necessary toolchain, such as NVCC, is available, TVM supports cross-compilation even in the absence of an actual device.


Step 5. Extra Python Dependencies
---------------------------------
Building from source does not ensure the installation of all necessary Python dependencies.
The following commands can be used to install the extra Python dependencies:

* Necessary dependencies:

.. code:: bash

    pip3 install numpy decorator attrs

* If you want to use RPC Tracker

.. code:: bash

    pip3 install tornado

* If you want to use auto-tuning module

.. code:: bash

    pip3 install tornado psutil 'xgboost>=1.1.0' cloudpickle


Advanced Build Configuration
----------------------------

Ccache
~~~~~~
On supported platforms, the `Ccache compiler wrapper <https://ccache.dev/>`_ may be helpful for
reducing TVM's build time, especially when building with `cutlass <https://github.com/NVIDIA/cutlass>`_
or `flashinfer <https://github.com/flashinfer-ai/flashinfer>`_.
There are several ways to enable CCache in TVM builds:

    - Leave ``USE_CCACHE=AUTO`` in ``build/config.cmake``. CCache will be used if it is found.

    - Ccache's Masquerade mode. This is typically enabled during the Ccache installation process.
      To have TVM use Ccache in masquerade, simply specify the appropriate C/C++ compiler
      paths when configuring TVM's build system.  For example:
      ``cmake -DCMAKE_CXX_COMPILER=/usr/lib/ccache/c++ ...``.

    - Ccache as CMake's C++ compiler prefix.  When configuring TVM's build system,
      set the CMake variable ``CMAKE_CXX_COMPILER_LAUNCHER`` to an appropriate value.
      E.g. ``cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...``.


Building on Windows
~~~~~~~~~~~~~~~~~~~
TVM support build via MSVC using cmake. You will need to obtain a visual studio compiler.
The minimum required VS version is **Visual Studio Enterprise 2019** (NOTE: we test
against GitHub Actions' `Windows 2019 Runner <https://github.com/actions/virtual-environments/blob/main/images/win/Windows2019-Readme.md>`_, so see that page for full details.
We recommend following :ref:`install-dependencies` to obtain necessary dependencies and
get an activated tvm-build environment. Then you can run the following command to build

.. code:: bash

    mkdir build
    cd build
    cmake ..
    cd ..

The above command generates the solution file under the build directory.
You can then run the following command to build

.. code:: bash

    cmake --build build --config Release -- /m


Building ROCm support
~~~~~~~~~~~~~~~~~~~~~

Currently, ROCm is supported only on linux, so all the instructions are written with linux in mind.

- Set ``set(USE_ROCM ON)``, set ROCM_PATH to the correct path.
- You need to first install HIP runtime from ROCm. Make sure the installation system has ROCm installed in it.
- Install latest stable version of LLVM (v6.0.1), and LLD, make sure ``ld.lld`` is available via command line.

.. _install-from-source-cpp-tests:

Enable C++ Tests
~~~~~~~~~~~~~~~~
We use `Google Test <https://github.com/google/googletest>`_ to drive the C++
tests in TVM. The easiest way to install GTest is from source.

.. code:: bash

    git clone https://github.com/google/googletest
    cd googletest
    mkdir build
    cd build
    cmake -DBUILD_SHARED_LIBS=ON ..
    make
    sudo make install

After installing GTest, the C++ tests can be built and started with ``./tests/scripts/task_cpp_unittest.sh`` or just built with ``make cpptest``.
