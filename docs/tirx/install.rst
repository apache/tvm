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

Installation
============

There are two pieces:

- the **TIRx compiler** (``tvm.tirx``), which ships inside Apache TVM — this is
  the authoring, IR, lowering, and backend infrastructure. Compiling or running
  for a particular target also requires a TVM build with that target enabled;
- the optional **kernel library** (``tirx-kernels``), a collection of ready-made
  kernels and benchmark infrastructure built with TIRx.

Requirements
------------

- Python ≥ 3.10.
- For the CUDA programming guide and bundled kernels: a CUDA-enabled TVM build,
  an NVIDIA driver, and a compatible CUDA toolkit. The bundled kernels target
  Blackwell (``sm_100a``); TIRx itself also supports other target backends.

Install the TIRx compiler
-------------------------

Install the Apache TVM wheel (the TIRx compiler is the ``tvm.tirx`` module):

.. code-block:: bash

   pip install apache-tvm

The wheel is enough to inspect and author TIRx IR. Some CUDA workflows also
need NVIDIA's Python CUDA bindings, available through ``apache-tvm[cuda]``.
That extra does not enable CUDA in the TVM library and does not install a driver
or toolkit. To compile and run the CUDA examples, use a TVM build configured
with ``TVM_TIRX_USE_CUDA=ON``; see :ref:`install TVM from source <install-from-source>`.

Verify:

.. code-block:: bash

   python -c "import tvm, tvm.tirx; print(tvm.__version__)"

Install the kernel library (optional)
-------------------------------------

Install the latest ``tirx-kernels`` release from PyPI:

.. code-block:: bash

   pip install tirx-kernels

Or install a checkout for development:

.. code-block:: bash

   git clone https://github.com/mlc-ai/tirx-kernels
   cd tirx-kernels
   pip install -e .

Apache TVM and PyTorch are externally managed runtime/compiler dependencies;
installing ``tirx-kernels`` does not install them. Put a TIRx-enabled TVM on
``PYTHONPATH`` and use a CUDA build of PyTorch matching the system. Individual
correctness tests and reference baselines need additional upstream projects.
For a source checkout, install their pinned, mutually compatible revisions with:

.. code-block:: bash

   python scripts/install_reference_dependencies.py

The kernel registry and reference dependency set evolve independently of TVM.
Use ``python -m tirx_kernels.registry --format json`` for the installed kernel
list, and consult the `tirx-kernels README
<https://github.com/mlc-ai/tirx-kernels#readme>`_ and
``reference-dependencies.json`` in that repository for current optional
requirements.

Standalone CMake build
----------------------

The core can be built independently from the repository checkout. Initialize
its bundled FFI dependency, then use native CMake options (CMake 3.26 or newer)::

    git submodule update --init --recursive tirx/3rdparty/tvm-ffi
    cmake -S tirx -B tirx/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DTVM_TIRX_USE_LLVM=ON -DTVM_TIRX_BUILD_TESTS=ON
    cmake --build tirx/build
    ctest --test-dir tirx/build
    cmake --install tirx/build --prefix /path/to/install

Core feature options use the ``TVM_TIRX_USE_`` prefix. SDK selectors such as
``TVM_TIRX_USE_CUDA`` and ``TVM_TIRX_USE_LLVM`` also accept an SDK path or
``llvm-config`` command. Compiler caches, linkers, and toolchain flags use
standard CMake settings. The standalone core does not read ``config.cmake``.
C++ tests are off by default; enabling them supplies GTest in the test subtree.
Installation includes public headers, libraries, and CMake package targets.

For a Python core package build, pass these options through the ``CMAKE_ARGS``
environment variable. Quote values containing spaces inside the variable::

    CMAKE_ARGS='-DTVM_TIRX_USE_LLVM="llvm-config --link-static"' \
        python -m pip install -v ./tirx

The repository's native build script also accepts ``CMAKE_ARGS`` for its TVM
configuration, including parent-only options::

    mkdir -p build
    CMAKE_ARGS='-DTVM_TIRX_USE_LLVM=ON -DUSE_RPC=ON' \
        python tests/scripts/task_build.py --build-dir build

The native build script applies these settings only to TVM configuration;
its separate source-FFI Python package build keeps its own defaults. Direct
``cmake`` invocations use command-line options as shown above, since CMake
itself does not read ``CMAKE_ARGS``.

A parent project can use ``add_subdirectory(path/to/tirx)`` and link to
``tvm_tirx::shared`` or ``tvm_runtime::shared``. This builds the bundled FFI
source targets and leaves tests, packaging, and installation to the parent.
TIRx registers installation rules only when it is the top-level project.
An embedding parent owns the installation rules for the libraries, public
headers, and package files it chooses to include.

Installed consumers
-------------------

An ordinary downstream compiler links TIRx and the separate FFI package
explicitly. Install both Python packages into the environment used by CMake::

    python -m pip install apache-tvm-tirx apache-tvm-ffi

Query each package's CMake directory::

    python -m tvm_tirx.config --cmakedir
    python -m tvm_ffi.config --cmakedir

Pass those directories as ``tvm_tirx_DIR`` and ``tvm_ffi_DIR`` when configuring
the consumer. Its CMakeLists.txt uses both packages directly:

.. code-block:: cmake

    find_package(tvm_tirx CONFIG REQUIRED)
    find_package(tvm_ffi CONFIG REQUIRED)
    add_executable(app main.cc)
    target_link_libraries(app PRIVATE tvm_tirx::shared tvm_ffi::shared)

The public TIRx targets are ``tvm_tirx::header``, ``tvm_tirx::shared``, and
``tvm_runtime::shared``. The package config describes the normal Python build;
it does not reconstruct build options, expose static or backend targets, or
find/link FFI implicitly. Enabled backend libraries are still installed and
loaded by the runtime. Consumers of an internally built static runtime supply
its dependencies, including OpenMP when enabled, themselves.

TIRx installs only its own public headers, libraries, and package helpers;
it does not copy FFI libraries, headers, or CMake files. For a native prefix
installation, use its ``lib/cmake/tvm_tirx`` directory (or the configured
library directory). The handwritten config queries that exact prefix, so a
separately installed TIRx wheel cannot override the selected installation.
``Python_EXECUTABLE`` selects the Python interpreter used for discovery.
