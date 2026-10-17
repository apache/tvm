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

tvm_tirx.backend
================

.. automodule:: tvm_tirx.backend
   :no-members:

.. currentmodule:: tvm_tirx.backend

Functions
---------

.. autosummary::
   :toctree: generated/

   is_loaded
   load

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.backend.cuda <backend.cuda>`
     - CUDA-owned TIRx modules.
   * - :doc:`tvm_tirx.backend.adreno <backend.adreno>`
     - Adreno-owned backend hooks.
   * - :doc:`tvm_tirx.backend.config <backend.config>`
     - Copy, merge, and serialize explicit per-backend compilation settings.
   * - :doc:`tvm_tirx.backend.hexagon <backend.hexagon>`
     - Hexagon-owned backend hooks.
   * - :doc:`tvm_tirx.backend.loader <backend.loader>`
     - Backend loading and public alias support.
   * - :doc:`tvm_tirx.backend.metal <backend.metal>`
     - Metal-owned TIRx modules.
   * - :doc:`tvm_tirx.backend.opencl <backend.opencl>`
     - OpenCL-owned backend hooks.
   * - :doc:`tvm_tirx.backend.rocm <backend.rocm>`
     - ROCm-owned TIRx modules.
   * - :doc:`tvm_tirx.backend.trn <backend.trn>`
     - Trainium-owned TIRx modules.
   * - :doc:`tvm_tirx.backend.vulkan <backend.vulkan>`
     - Vulkan-owned backend hooks.
   * - :doc:`tvm_tirx.backend.webgpu <backend.webgpu>`
     - WebGPU-owned backend hooks.

.. toctree::
   :hidden:
   :maxdepth: 2

   backend.cuda
   backend.adreno
   backend.config
   backend.hexagon
   backend.loader
   backend.metal
   backend.opencl
   backend.rocm
   backend.trn
   backend.vulkan
   backend.webgpu
