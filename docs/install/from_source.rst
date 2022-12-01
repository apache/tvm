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

    git clone --recursive https://github.com/apache/tvm tvm

For windows users who use github tools, you can open the git shell, and type the following command.

.. code:: bash

   git submodule init
   git submodule update


.. _build-shared-library:

Build the Shared Library
------------------------

Our goal is to build the shared libraries:

   - On Linux the target library are `libtvm.so` and `libtvm_runtime.so`
   - On macOS the target library are `libtvm.dylib` and `libtvm_runtime.dylib`
   - On Windows the target library are `libtvm.dll` and `libtvm_runtime.dll`

It is also possible to :ref:`build the runtime <deploy-and-integration>` library only.

The minimal building requirements for the ``TVM`` libraries are:

   - A recent C++ compiler supporting C++ 17, at the minimum
      - GCC 7.1
      - Clang 5.0
      - Apple Clang 9.3
      - Visual Studio 2019 (v16.7)
   - CMake 3.18 or higher
   - We highly recommend to build with LLVM to enable all the features.
   - If you want to use CUDA, CUDA toolkit version >= 8.0 is required. If you are upgrading from an older version, make sure you purge the older version and reboot after installation.
   - On macOS, you may want to install `Homebrew <https://brew.sh>`_ to easily install and manage dependencies.
   - Python is also required. Avoid using Python 3.9.X+ which is not `supported <https://github.com/apache/tvm/issues/8577>`_. 3.7.X+ and 3.8.X+ should be well supported however.

To install the these minimal pre-requisites on Ubuntu/Debian like
linux operating systems, execute (in a terminal):

.. code:: bash

    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev


Note that the version of CMake on apt may not be sufficiently up to date; it may be necessary to install it directly from `Kitware's third-party APT repository <https://apt.kitware.com/>`_.


On Fedora/CentOS and related operating systems use:

.. code:: bash

    sudo dnf update
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y python-devel ncurses-compat-libs zlib-devel cmake libedit-devel libxml2-devel

Use Homebrew to install the required dependencies for macOS running either the Intel or M1 processors. You must follow the post-installation steps specified by
Homebrew to ensure the dependencies are correctly installed and configured:

.. code:: bash

    brew install gcc git cmake
    brew install llvm
    brew install python@3.8

If you are on macOS with an M1 Processor you may need to use conda to manage dependencies while building. Specifically you may need, `Miniforge <https://github.com/conda-forge/miniforge>`_ to ensure that the dependencies obtained using pip are compatible with M1.

.. code:: bash

    brew install miniforge
    conda init
    conda create --name tvm python=3.8
    conda activate tvm

We use cmake to build the library.
The configuration of TVM can be modified by editing `config.cmake` and/or by passing cmake flags to the command line:


- First, check the cmake in your system. If you do not have cmake,
  you can obtain the latest version from `official website <https://cmake.org/download/>`_
- First create a build directory, copy the ``cmake/config.cmake`` to the directory.

  .. code:: bash

      mkdir build
      cp cmake/config.cmake build

- Edit ``build/config.cmake`` to customize the compilation options

  - On macOS, for some versions of Xcode, you need to add ``-lc++abi`` in the LDFLAGS or you'll get link errors.
  - Change ``set(USE_CUDA OFF)`` to ``set(USE_CUDA ON)`` to enable CUDA backend. Do the same for other backends and libraries
    you want to build for (OpenCL, RCOM, METAL, VULKAN, ...).
  - To help with debugging, ensure the embedded graph executor and debugging functions are enabled with ``set(USE_GRAPH_EXECUTOR ON)`` and ``set(USE_PROFILER ON)``
  - To debug with IRs, ``set(USE_RELAY_DEBUG ON)`` and set environment variable `TVM_LOG_DEBUG`.

      .. code:: bash

          export TVM_LOG_DEBUG="ir/transform.cc=1,relay/ir/transform.cc=1"

- TVM requires LLVM for CPU codegen. We highly recommend you to build with the LLVM support on.

  - LLVM 4.0 or higher is needed for build with LLVM. Note that version of LLVM from default apt may lower than 4.0.
  - Since LLVM takes long time to build from source, you can download pre-built version of LLVM from
    `LLVM Download Page <http://releases.llvm.org/download.html>`_.

    - Unzip to a certain location, modify ``build/config.cmake`` to add ``set(USE_LLVM /path/to/your/llvm/bin/llvm-config)``
    - You can also directly set ``set(USE_LLVM ON)`` and let cmake search for a usable version of LLVM.

  - You can also use `LLVM Nightly Ubuntu Build <https://apt.llvm.org/>`_

    - Note that apt-package append ``llvm-config`` with version number.
      For example, set ``set(USE_LLVM llvm-config-10)`` if you installed LLVM 10 package

  - If you are a PyTorch user, it is recommended to set ``(USE_LLVM "/path/to/llvm-config --link-static")`` and ``set(HIDE_PRIVATE_SYMBOLS ON)``
    to avoid potential symbol conflicts between different versions LLVM used by TVM and PyTorch.

  - On supported platforms, the `Ccache compiler wrapper <https://ccache.dev/>`_ may be helpful for
    reducing TVM's build time.  There are several ways to enable CCache in TVM builds:

    - Leave `USE_CCACHE=AUTO` in `build/config.cmake`. CCache will be used if it is found.

    - Ccache's Masquerade mode. This is typically enabled during the Ccache installation process.
      To have TVM use Ccache in masquerade, simply specify the appropriate C/C++ compiler
      paths when configuring TVM's build system.  For example:
      ``cmake -DCMAKE_CXX_COMPILER=/usr/lib/ccache/c++ ...``.

    - Ccache as CMake's C++ compiler prefix.  When configuring TVM's build system,
      set the CMake variable ``CMAKE_CXX_COMPILER_LAUNCHER`` to an appropriate value.
      E.g. ``cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...``.

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

  - There is also a makefile in the top-level tvm directory that can
    automate several of these steps.  It will create the build
    directory, copy the default ``config.cmake`` to the build
    directory, run cmake, then run make.

    The build directory can be specified using the environment
    variable ``TVM_BUILD_PATH``.  If ``TVM_BUILD_PATH`` is unset, the
    makefile assumes that the ``build`` directory inside tvm should be
    used.  Paths specified by ``TVM_BUILD_PATH`` can be either
    absolute paths or paths relative to the base tvm directory.
    ``TVM_BUILD_PATH`` can also be set to a list of space-separated
    paths, in which case all paths listed will be built.

    If an alternate build directory is used, then the environment
    variable ``TVM_LIBRARY_PATH`` should be set at runtime, pointing
    to the location of the compiled ``libtvm.so`` and
    ``libtvm_runtime.so``.  If not set, tvm will look relative to the
    location of the tvm python module.  Unlike ``TVM_BUILD_PATH``,
    this must be an absolute path.

  .. code:: bash

     # Build in the "build" directory
     make

     # Alternate location, "build_debug"
     TVM_BUILD_PATH=build_debug make

     # Build both "build_release" and "build_debug"
     TVM_BUILD_PATH="build_debug build_release" make

     # Use debug build
     TVM_LIBRARY_PATH=~/tvm/build_debug python3

If everything goes well, we can go to :ref:`python-package-installation`

.. _build-with-conda:

Building with a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conda is a very handy way to the necessary obtain dependencies needed for running TVM.
First, follow the `conda's installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
to install miniconda or anaconda if you do not yet have conda in your system. Run the following command in a conda environment:

.. code:: bash

    # Create a conda environment with the dependencies specified by the yaml
    conda env create --file conda/build-environment.yaml
    # Activate the created environment
    conda activate tvm-build

The above command will install all necessary build dependencies such as cmake and LLVM. You can then run the standard build process in the last section.

If you want to use the compiled binary outside the conda environment,
you can set LLVM to static linking mode ``set(USE_LLVM "llvm-config --link-static")``.
In this way, the resulting library won't depend on the dynamic LLVM libraries in the conda environment.

The above instructions show how to use conda to provide the necessary build dependencies to build libtvm.
If you are already using conda as your package manager and wish to directly build and install tvm as a conda package, you can follow the instructions below:

.. code:: bash

   conda build --output-folder=conda/pkg  conda/recipe
   # Run conda/build_cuda.sh to build with cuda enabled
   conda install tvm -c ./conda/pkg

Building on Windows
~~~~~~~~~~~~~~~~~~~
TVM support build via MSVC using cmake. You will need to obtain a visual studio compiler.
The minimum required VS version is **Visual Studio Enterprise 2019** (NOTE: we test
against GitHub Actions' `Windows 2019 Runner <https://github.com/actions/virtual-environments/blob/main/images/win/Windows2019-Readme.md>`_, so see that page for full details.
We recommend following :ref:`build-with-conda` to obtain necessary dependencies and
get an activated tvm-build environment. Then you can run the following command to build

.. code:: bash

    mkdir build
    cd build
    cmake -A x64 -Thost=x64 ..
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

.. _python-package-installation:

Python Package Installation
---------------------------

TVM package
~~~~~~~~~~~

Depending on your development environment, you may want to use a virtual environment and package manager, such
as ``virtualenv`` or ``conda``, to manage your python packages and dependencies.

The python package is located at `tvm/python`
There are two ways to install the package:

Method 1
   This method is **recommended for developers** who may change the codes.

   Set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `tvm` on the directory
   `/path/to/tvm` then we can add the following line in `~/.bashrc`.
   The changes will be immediately reflected once you pull the code and rebuild the project (no need to call ``setup`` again)

   .. code:: bash

       export TVM_HOME=/path/to/tvm
       export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}


Method 2
   Install TVM python bindings by `setup.py`:

   .. code:: bash

       # install tvm package for the current user
       # NOTE: if you installed python via homebrew, --user is not needed during installaiton
       #       it will be automatically installed to your user directory.
       #       providing --user flag may trigger error during installation in such case.
       export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
       cd python; python setup.py install --user; cd ..

Python dependencies
~~~~~~~~~~~~~~~~~~~

Note that the ``--user`` flag is not necessary if you're installing to a managed local environment,
like ``virtualenv``.

   * Necessary dependencies:

   .. code:: bash

       pip3 install --user numpy decorator attrs

   * If you want to use RPC Tracker

   .. code:: bash

       pip3 install --user tornado

   * If you want to use auto-tuning module

   .. code:: bash

       pip3 install --user tornado psutil 'xgboost<1.6.0' cloudpickle

Note on M1 macs, you may have trouble installing xgboost / scipy. scipy and xgboost requires some additional dependencies to be installed,
including openblas and its dependencies. Use the following commands to install scipy and xgboost with the required dependencies and
configuration. A workaround for this is to do the following commands:

    .. code:: bash

        brew install openblas gfortran

        pip install pybind11 cython pythran

        export OPENBLAS=/opt/homebrew/opt/openblas/lib/

        pip install scipy --no-use-pep517

        pip install 'xgboost<1.6.0'

Install Contrib Libraries
-------------------------

.. toctree::
   :maxdepth: 1

   nnpack


.. _install-from-source-cpp-tests:

Enable C++ Tests
----------------
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
