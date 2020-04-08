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
This page gives instructions on how to build and install the TVM package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libtvm.so` for linux, `libtvm.dylib` for macOS and `libtvm.dll` for windows).
2. Setup for the language packages (e.g. Python Package).

To get started, download tvm source code from the `Download Page <https://tvm.apache.org/download>`_.

Developers: Get Source from Github
----------------------------------
You can also choose to clone the source repo from github.
It is important to clone the submodules along, with ``--recursive`` option.

.. code:: bash

    git clone --recursive https://github.com/apache/incubator-tvm tvm

For windows users who use github tools, you can open the git shell, and type the following command.

.. code:: bash

   git submodule init
   git submodule update


.. _build-shared-library:

Build the Shared Library
------------------------

Our goal is to build the shared libraries:

- On Linux the target library are `libtvm.so, libtvm_topi.so`
- On macOS the target library are `libtvm.dylib, libtvm_topi.dylib`
- On Windows the target library are `libtvm.dll, libtvm_topi.dll`


.. code:: bash

    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

The minimal building requirements are

- A recent c++ compiler supporting C++ 14 (g++-5 or higher)
- CMake 3.5 or higher
- We highly recommend to build with LLVM to enable all the features.
- If you want to use CUDA, CUDA toolkit version >= 8.0 is required. If you are upgrading from an older version, make sure you purge the older version and reboot after installation.


We use cmake to build the library.
The configuration of TVM can be modified by `config.cmake`.


- First, check the cmake in your system. If you do not have cmake,
  you can obtain the latest version from `official website <https://cmake.org/download/>`_
- First create a build directory, copy the ``cmake/config.cmake`` to the directory.

  .. code:: bash

      mkdir build
      cp cmake/config.cmake build

- Edit ``build/config.cmake`` to customize the compilation options

  - On macOS, for some versions of Xcode, you need to add ``-lc++abi`` in the LDFLAGS or you'll get link errors.
  - Change ``set(USE_CUDA OFF)`` to ``set(USE_CUDA ON)`` to enable CUDA backend. So do other backends and libraries
    (OpenCL, RCOM, METAL, VULKAN, ...).

- TVM optionally depends on LLVM. LLVM is required for CPU codegen that needs LLVM.

  - LLVM 4.0 or higher is needed for build with LLVM. Note that version of LLVM from default apt may lower than 4.0.
  - Since LLVM takes long time to build from source, you can download pre-built version of LLVM from
    `LLVM Download Page <http://releases.llvm.org/download.html>`_.


    - Unzip to a certain location, modify ``build/config.cmake`` to add ``set(USE_LLVM /path/to/your/llvm/bin/llvm-config)``
    - You can also directly set ``set(USE_LLVM ON)`` and let cmake search for a usable version of LLVM.

  - You can also use `LLVM Nightly Ubuntu Build <https://apt.llvm.org/>`_

    - Note that apt-package append ``llvm-config`` with version number.
      For example, set ``set(LLVM_CONFIG llvm-config-4.0)`` if you installed 4.0 package

- We can then build tvm and related libraries.

  .. code:: bash

      cd build
      cmake ..
      make -j4

  - You can also use Ninja build system instead of Unix Makefiles. It can be faster to build than using Makefiles.

  .. code:: bash

      cd build
      cmake .. -G Ninja
      ninja

If everything goes well, we can go to :ref:`python-package-installation`

Building on Windows
~~~~~~~~~~~~~~~~~~~

TVM support build via MSVC using cmake. The minimum required VS version is **Visual Studio Community 2015 Update 3**.
In order to generate the VS solution file using cmake, make sure you have a recent version of cmake added to your path and then from the TVM directory:

.. code:: bash

  mkdir build
  cd build
  cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ..

This will generate the VS project using the MSVC 14 64 bit generator.
Open the .sln file in the build directory and build with Visual Studio.
In order to build with LLVM in windows, you will need to build LLVM from source.

Building ROCm support
~~~~~~~~~~~~~~~~~~~~~

Currently, ROCm is supported only on linux, so all the instructions are written with linux in mind.

- Set ``set(USE_ROCM ON)``, set ROCM_PATH to the correct path.
- You need to first install HIP runtime from ROCm. Make sure the installation system has ROCm installed in it.
- Install latest stable version of LLVM (v6.0.1), and LLD, make sure ``ld.lld`` is available via command line.

.. _python-package-installation:

Python Package Installation
---------------------------

TVM package
~~~~~~~~~~~

The python package is located at `tvm/python`
There are two ways to install the package:

Method 1
   This method is **recommended for developers** who may change the codes.

   Set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `tvm` on the home directory
   `~`. then we can added the following line in `~/.bashrc`.
   The changes will be immediately reflected once you pull the code and rebuild the project (no need to call ``setup`` again)

   .. code:: bash

       export TVM_HOME=/path/to/tvm
       export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}


Method 2
   Install TVM python bindings by `setup.py`:

   .. code:: bash

       # install tvm package for the current user
       # NOTE: if you installed python via homebrew, --user is not needed during installaiton
       #       it will be automatically installed to your user directory.
       #       providing --user flag may trigger error during installation in such case.
       export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
       cd python; python setup.py install --user; cd ..
       cd topi/python; python setup.py install --user; cd ../..


Python dependencies
~~~~~~~~~~~~~~~~~~~
   * Necessary dependencies:

   .. code:: bash

       pip3 install --user numpy decorator attrs

   * If you want to use RPC Tracker

   .. code:: bash

       pip3 install --user tornado

   * If you want to use auto-tuning module

   .. code:: bash

       pip3 install --user tornado psutil xgboost

   * If you want to build tvm to compile a model, you must use Python 3 and run the following

   .. code:: bash

       sudo apt install antlr4
       pip3 install --user mypy orderedset antlr4-python3-runtime


Install Contrib Libraries
-------------------------

.. toctree::
   :maxdepth: 1

   nnpack


Enable C++ Tests
----------------
We use `Google Test <https://github.com/google/googletest>`_ to drive the C++
tests in TVM. The easiest way to install GTest is from source.

   .. code:: bash

       git clone https://github.com/google/googletest
       cd googletest
       mkdir build
       cd build
       cmake ..
       make
       make install


After installing GTest, the C++ tests can be built and started with ``./tests/scripts/task_cpp_unittest.sh`` or just built with ``make cpptest``.
