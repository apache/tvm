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

Python API
==========

.. automodule:: tvm_tirx
   :no-members:

.. currentmodule:: tvm_tirx

Namespaces
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.runtime <runtime>`
     - Tensors, devices, data types, and runtime modules.
   * - :doc:`tvm_tirx.ir <ir>`
     - Shared expressions, types, statements, and IR modules.
   * - :doc:`tvm_tirx.ir.prim <ir.prim>`
     - Primitive expressions and scalar operators.
   * - :doc:`tvm_tirx.ir.transform <ir.transform>`
     - Pass infrastructure and instrumentation.
   * - :doc:`tvm_tirx.script <script>`
     - Python syntax registration for IR dialects.
   * - :doc:`tvm_tirx.script.parser <script.parser>`
     - Translate Python scripts into IR builder operations.
   * - :doc:`tvm_tirx.script.ir_builder <script.ir_builder>`
     - Shared IR builder contexts and statements.
   * - :doc:`tvm_tirx.sym <sym>`
     - Symbolic arithmetic, analysis, and simplification.
   * - :doc:`tvm_tirx.target <target>`
     - Compilation targets and device selection.
   * - :doc:`tvm_tirx.tirx <tirx>`
     - Tensor IR construction and compilation.
   * - :doc:`tvm_tirx.tirx.op <tirx.op>`
     - Tensor IR operators and intrinsic calls.
   * - :doc:`tvm_tirx.tirx.analysis <tirx.analysis>`
     - Tensor IR analysis and verification.
   * - :doc:`tvm_tirx.tirx.transform <tirx.transform>`
     - Tensor IR transformation passes.
   * - :doc:`tvm_tirx.tirx.layout <tirx.layout>`
     - Tensor layouts and logical-to-physical index mappings.
   * - :doc:`tvm_tirx.tirx.tile <tirx.tile>`
     - Tile operations, views, and dispatch.
   * - :doc:`tvm_tirx.tirx.script <tirx.script>`
     - Tensor IR syntax and JIT compilation.
   * - :doc:`tvm_tirx.codegen <codegen>`
     - Shared code generation and LLVM helpers.
   * - :doc:`tvm_tirx.backend <backend>`
     - Load and register device backends.
   * - :doc:`tvm_tirx.backend.cuda <backend.cuda>`
     - CUDA backend configuration and registration.
   * - :doc:`tvm_tirx.backend.cuda.op <backend.cuda.op>`
     - CUDA intrinsic builders.
   * - :doc:`tvm_tirx.backend.cuda.script <backend.cuda.script>`
     - CUDA kernel authoring namespace.
   * - :doc:`tvm_tirx.backend.cuda.layout <backend.cuda.layout>`
     - CUDA shared-memory and tensor-memory layouts.
   * - :doc:`tvm_tirx.backend.cuda.lang <backend.cuda.lang>`
     - CUDA pipelines, barriers, storage pools, and scheduling.
   * - :doc:`tvm_tirx.backend.cuda.transform <backend.cuda.transform>`
     - CUDA-specific transformation passes.
   * - :doc:`tvm_tirx.backend.rocm <backend.rocm>`
     - ROCm backend registration.
   * - :doc:`tvm_tirx.backend.metal <backend.metal>`
     - Metal backend and script namespaces.
   * - :doc:`tvm_tirx.backend.metal.op <backend.metal.op>`
     - Metal intrinsic builders.
   * - :doc:`tvm_tirx.backend.opencl <backend.opencl>`
     - OpenCL backend and script namespaces.
   * - :doc:`tvm_tirx.backend.adreno <backend.adreno>`
     - Adreno backend registration.
   * - :doc:`tvm_tirx.backend.hexagon <backend.hexagon>`
     - Hexagon backend and linker integration.
   * - :doc:`tvm_tirx.backend.vulkan <backend.vulkan>`
     - Vulkan backend registration.
   * - :doc:`tvm_tirx.backend.webgpu <backend.webgpu>`
     - WebGPU backend registration.
   * - :doc:`tvm_tirx.backend.trn <backend.trn>`
     - Trainium backend and script namespaces.
   * - :doc:`tvm_tirx.backend.trn.op <backend.trn.op>`
     - Trainium NKI intrinsic wrappers.
   * - :doc:`tvm_tirx.backend.trn.transform <backend.trn.transform>`
     - Trainium-specific lowering passes.
   * - :doc:`tvm_tirx.support <support>`
     - Build tools and system utilities.
   * - :doc:`tvm_tirx.testing <testing>`
     - Test fixtures and numerical checks.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   tvm_tirx.runtime <runtime>
   tvm_tirx.ir <ir>
   tvm_tirx.ir.prim <ir.prim>
   tvm_tirx.ir.transform <ir.transform>
   tvm_tirx.script <script>
   tvm_tirx.script.parser <script.parser>
   tvm_tirx.script.ir_builder <script.ir_builder>
   tvm_tirx.sym <sym>
   tvm_tirx.target <target>
   tvm_tirx.tirx <tirx>
   tvm_tirx.tirx.op <tirx.op>
   tvm_tirx.tirx.analysis <tirx.analysis>
   tvm_tirx.tirx.transform <tirx.transform>
   tvm_tirx.tirx.layout <tirx.layout>
   tvm_tirx.tirx.tile <tirx.tile>
   tvm_tirx.tirx.script <tirx.script>
   tvm_tirx.codegen <codegen>
   tvm_tirx.backend <backend>
   tvm_tirx.backend.cuda <backend.cuda>
   tvm_tirx.backend.cuda.op <backend.cuda.op>
   tvm_tirx.backend.cuda.script <backend.cuda.script>
   tvm_tirx.backend.cuda.layout <backend.cuda.layout>
   tvm_tirx.backend.cuda.lang <backend.cuda.lang>
   tvm_tirx.backend.cuda.transform <backend.cuda.transform>
   tvm_tirx.backend.rocm <backend.rocm>
   tvm_tirx.backend.metal <backend.metal>
   tvm_tirx.backend.metal.op <backend.metal.op>
   tvm_tirx.backend.opencl <backend.opencl>
   tvm_tirx.backend.adreno <backend.adreno>
   tvm_tirx.backend.hexagon <backend.hexagon>
   tvm_tirx.backend.vulkan <backend.vulkan>
   tvm_tirx.backend.webgpu <backend.webgpu>
   tvm_tirx.backend.trn <backend.trn>
   tvm_tirx.backend.trn.op <backend.trn.op>
   tvm_tirx.backend.trn.transform <backend.trn.transform>
   tvm_tirx.support <support>
   tvm_tirx.testing <testing>

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``DataTypeCode``
     - :obj:`tvm_tirx.runtime.DataTypeCode`
   * - ``IRModule``
     - :obj:`tvm_tirx.ir.IRModule`

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``cpu``
     - :obj:`tvm_tirx.runtime.cpu`
   * - ``cuda``
     - :obj:`tvm_tirx.runtime.cuda`
   * - ``opencl``
     - :obj:`tvm_tirx.runtime.opencl`
   * - ``vulkan``
     - :obj:`tvm_tirx.runtime.vulkan`
   * - ``metal``
     - :obj:`tvm_tirx.runtime.metal`
   * - ``vpi``
     - :obj:`tvm_tirx.runtime.vpi`
   * - ``rocm``
     - :obj:`tvm_tirx.runtime.rocm`
   * - ``ext_dev``
     - :obj:`tvm_tirx.runtime.ext_dev`
   * - ``device_from_target``
     - :obj:`tvm_tirx.target.device_from_target`
