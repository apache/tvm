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

tvm_tirx.backend.cuda
=====================

.. automodule:: tvm_tirx.backend.cuda
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda

Classes
-------

.. autosummary::
   :toctree: generated/

   BackendConfig

Functions
---------

.. autosummary::
   :toctree: generated/

   export_cuda_host
   register_backend
   script_namespaces
   script_namespace

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.backend.cuda.op <backend.cuda.op>`
     - CUDA, PTX, and NVSHMEM TIR intrinsic builders.
   * - :doc:`tvm_tirx.backend.cuda.axis <backend.cuda.axis>`
     - CUDA layout axes.
   * - :doc:`tvm_tirx.backend.cuda.backend_config <backend.cuda.backend_config>`
     - CUDA toolchain selection and native argument lists.
   * - :doc:`tvm_tirx.backend.cuda.codegen <backend.cuda.codegen>`
     - Shared machinery behind every CUDA device-intrinsic codegen.
   * - :doc:`tvm_tirx.backend.cuda.compiler <backend.cuda.compiler>`
     - Compile one CUDA source group using explicit backend settings.
   * - :doc:`tvm_tirx.backend.cuda.cpp <backend.cuda.cpp>`
     - Hand-written CUDA C++ device helpers.
   * - :doc:`tvm_tirx.backend.cuda.iket <backend.cuda.iket>`
     - NVIDIA IKET annotations and run-iket orchestration.
   * - :doc:`tvm_tirx.backend.cuda.lang <backend.cuda.lang>`
     - CUDA-specific TIRx language helpers.
   * - :doc:`tvm_tirx.backend.cuda.launch <backend.cuda.launch>`
     - CUDA launch configuration shared by TVMScript and host launch adapters.
   * - :doc:`tvm_tirx.backend.cuda.layout <backend.cuda.layout>`
     - CUDA-specific memory and instruction layout factories.
   * - :doc:`tvm_tirx.backend.cuda.ptx <backend.cuda.ptx>`
     - Table-driven PTX dialect (T.ptx).
   * - :doc:`tvm_tirx.backend.cuda.script <backend.cuda.script>`
     - CUDA TVMScript namespaces.
   * - :doc:`tvm_tirx.backend.cuda.thread_scope <backend.cuda.thread_scope>`
     - CUDA thread-scope lookup.
   * - :doc:`tvm_tirx.backend.cuda.tile <backend.cuda.tile>`
     - Namespace reference.
   * - :doc:`tvm_tirx.backend.cuda.transform <backend.cuda.transform>`
     - CUDA-specific TIRx transformations.

.. toctree::
   :hidden:
   :maxdepth: 2

   backend.cuda.op
   backend.cuda.axis
   backend.cuda.backend_config
   backend.cuda.codegen
   backend.cuda.compiler
   backend.cuda.cpp
   backend.cuda.iket
   backend.cuda.lang
   backend.cuda.launch
   backend.cuda.layout
   backend.cuda.ptx
   backend.cuda.script
   backend.cuda.thread_scope
   backend.cuda.tile
   backend.cuda.transform
