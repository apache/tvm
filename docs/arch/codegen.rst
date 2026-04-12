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

.. _codegen-arch:

Code Generation
===============

Code generation is the final stage of the TVM compilation pipeline — it translates TIR
``PrimFunc``\ s into executable code for a target device. This document explains how TIR
functions become native CPU instructions, GPU kernels, or source code strings, covering the
target dispatch mechanism, the two codegen families (LLVM and Source), and the runtime module
system that wraps the generated code.


Where Codegen Fits
------------------

When a user calls ``tvm.compile()``, the compilation proceeds in two phases:

1. **Relax phase**: the Relax pipeline optimizes and fuses the computational graph, then
   ``VMCodeGen`` translates Relax functions into VM bytecode (see :ref:`relax-vm-arch`).
2. **TIR phase**: TIR ``PrimFunc``\ s (the actual compute kernels) are compiled to native code.

The TIR phase is handled internally by ``tirx.build()`` (called from ``relax.build()``).
It performs these steps:

.. code-block:: text

   TIR PrimFuncs (in IRModule)
        │
        ▼  TIR pipeline                   ← lowering passes (flatten buffers, lower intrinsics, etc.)
   TIR PrimFuncs (lowered)
        │
        ▼  split_host_device_mods()        ← separate host and device functions
   Host IRModule + Device IRModule(s)
        │                    │
        ▼                    ▼
   codegen_build()      codegen_build()    ← target-specific code generation
        │                    │
        ▼                    ▼
   Host Module          Device Module(s)
        │                    │
        ▼  import_module()   │
   Host Module ◄─────────────┘             ← device modules imported into host
        │
        ▼  (returned to relax.build for linking with VM bytecode)


Target Dispatch
---------------

The core dispatch logic lives in ``codegen::Build()`` (``src/target/codegen.cc``), which is
called from the Python-side ``codegen_build()`` in ``tirx/build.py``. It selects the correct
backend based on the ``Target`` object:

.. code-block:: cpp

   ffi::Module Build(IRModule mod, Target target) {
     std::string build_f_name = "target.build." + target->kind->name;
     const auto bf = tvm::ffi::Function::GetGlobal(build_f_name);
     return (*bf)(mod, target).cast<ffi::Module>();
   }

Each backend registers its build function via FFI:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - FFI Key
     - Backend
     - Codegen Class
   * - ``target.build.llvm``
     - CPU (x86, ARM, etc.)
     - ``CodeGenCPU`` (→ LLVM IR → machine code)
   * - ``target.build.cuda``
     - NVIDIA GPU
     - ``CodeGenCUDA`` (→ CUDA C → PTX/cubin)
   * - ``target.build.rocm``
     - AMD GPU
     - ``CodeGenAMDGPU`` (→ LLVM IR → AMDGPU ISA)
   * - ``target.build.nvptx``
     - NVIDIA PTX
     - ``CodeGenNVPTX`` (→ LLVM IR → PTX)
   * - ``target.build.metal``
     - Apple GPU
     - ``CodeGenMetal`` (→ Metal Shading Language)
   * - ``target.build.opencl``
     - OpenCL devices
     - ``CodeGenOpenCL`` (→ OpenCL C)
   * - ``target.build.vulkan``
     - Vulkan devices
     - ``CodeGenSPIRV`` (→ SPIR-V binary)
   * - ``target.build.webgpu``
     - WebGPU
     - ``CodeGenWebGPU`` (→ WGSL)
   * - ``target.build.c``
     - C host code
     - ``CodeGenCHost`` (→ C source)


Two Codegen Families
--------------------

TVM has two families of code generators, corresponding to two fundamentally different strategies
for producing executable code:

.. code-block:: text

   LLVM Family                          Source Family
   ──────────                           ─────────────
   TIR → LLVM IR → machine code        TIR → source string → external compiler
   (in-process, JIT or AOT)            (CUDA C, OpenCL C, Metal, WGSL)

LLVM family
~~~~~~~~~~~

``CodeGenLLVM`` (``src/target/llvm/codegen_llvm.h``) translates TIR directly to LLVM IR using
the LLVM C++ API. The generated ``llvm::Module`` is then compiled to native code by LLVM's
backend (x86, ARM, NVPTX, AMDGPU, etc.).

**Inheritance**:

.. code-block:: text

   CodeGenLLVM (base)
   ├── CodeGenCPU       ← x86, ARM (target.build.llvm)
   │   └── CodeGenHexagon
   ├── CodeGenNVPTX     ← NVIDIA PTX via LLVM (target.build.nvptx)
   └── CodeGenAMDGPU    ← AMD GPU via LLVM (target.build.rocm)

``CodeGenLLVM`` inherits from both ``ExprFunctor<llvm::Value*(const PrimExpr&)>`` and
``StmtFunctor<void(const Stmt&)>``. Each TIR node type has a corresponding visitor:

- **Expressions** (``VisitExpr_``) convert TIR expressions to LLVM ``Value``\ s:
  arithmetic ops → LLVM binary instructions, ``BufferLoad`` → load with pointer arithmetic,
  ``Cast`` → LLVM type conversions, ``Call`` → intrinsic or extern function calls.
- **Statements** (``VisitStmt_``) emit LLVM IR side effects:
  ``BufferStore`` → store instructions, ``For`` → loop basic blocks with branches,
  ``IfThenElse`` → conditional branches, ``AllocBuffer`` → stack or heap allocation.

The key methods on ``CodeGenLLVM`` are:

- ``Create(LLVMTarget*)`` — factory that returns a target-specific subclass.
- ``Init(...)`` — set up the LLVM context, module, and builder.
- ``DeclareFunction(gvar, f)`` / ``AddFunction(gvar, f)`` — forward-declare then compile a
  ``PrimFunc`` to LLVM IR.
- ``Finish()`` — return the completed ``llvm::Module``.

Source family
~~~~~~~~~~~~~

``CodeGenC`` (``src/target/source/codegen_c.h``) generates C-like source code as text. Each
target subclass overrides methods to emit target-specific syntax.

**Inheritance**:

.. code-block:: text

   CodeGenC (base)
   ├── CodeGenCUDA      ← CUDA C (target.build.cuda)
   ├── CodeGenOpenCL    ← OpenCL C (target.build.opencl)
   ├── CodeGenMetal     ← Metal Shading Language (target.build.metal)
   ├── CodeGenWebGPU    ← WGSL (target.build.webgpu)
   └── CodeGenCHost     ← C host code (target.build.c)

``CodeGenC`` also uses the visitor pattern (``ExprFunctor`` and ``StmtFunctor``), but outputs to
``std::ostream`` instead of constructing LLVM IR. Subclasses override target-specific methods:

- ``PrintStorageScope(scope, os)`` — emit memory qualifiers (e.g., ``__shared__`` for CUDA,
  ``__local`` for OpenCL).
- ``BindThreadIndex(iv)`` — emit thread index bindings (e.g., ``threadIdx.x``, ``blockIdx.y``).
- ``PrintType(dtype, os)`` — emit target-specific type names (e.g., ``half`` for float16).
- ``PrintVecBinaryOp(...)`` — emit vectorized operations in target syntax.

For CUDA, the build flow (``BuildCUDA`` in ``src/target/opt/build_cuda_on.cc``) is:

1. ``CodeGenCUDA`` generates CUDA C source.
2. An optional post-processing callback (``tvm_callback_cuda_postproc``) transforms the source.
3. A Python callback (``tvm_callback_cuda_compile``) compiles the source to PTX or cubin via
   NVRTC or NVCC.
4. The result is wrapped in a ``CUDAModule``.

Design choice
~~~~~~~~~~~~~

Why two families?

- **LLVM family** produces higher-quality code — LLVM applies its own optimization passes
  (instruction selection, register allocation, vectorization). Best for CPU targets where TVM
  has full control over the compilation.
- **Source family** is more portable — it generates human-readable source that can be compiled
  by vendor toolchains (NVCC, Metal compiler, etc.). This is necessary for GPU targets where
  the vendor compiler handles device-specific optimizations and the runtime compilation model
  (e.g., NVRTC for CUDA, runtime shader compilation for Metal/OpenCL).


Host/Device Split
-----------------

When compiling for GPU targets, TIR functions are split into two categories:

- **Host functions** — run on the CPU. They set up kernel launch parameters (grid/block
  dimensions), allocate memory, and invoke device kernels. Compiled with ``target.build.llvm``
  or ``target.build.c``.
- **Device functions** — the actual compute kernels that run on the GPU. Compiled with the
  target-specific codegen (``target.build.cuda``, etc.).

``split_host_device_mods()`` (``python/tvm/tirx/build.py``) separates functions by their
``target`` attribute: functions whose target kind is ``"llvm"`` or ``"c"`` go to the host
module; all others go to device modules grouped by target.

After compilation, device modules are imported into the host module via ``import_module()``,
forming a module tree. At runtime, the host module dispatches to the imported device module
when a device kernel is called.


Runtime Modules
---------------

Each codegen produces a ``runtime.Module`` — the container that holds the generated code and
exposes it as callable ``PackedFunc``\ s.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Module Type
     - How Code Is Stored
     - How Code Is Executed
   * - ``LLVMModule``
     - LLVM IR (in-memory ``llvm::Module``)
     - JIT-compiled on first call (MCJIT or ORC). Function pointers cached for subsequent calls.
   * - ``CUDAModule``
     - PTX or cubin binary
     - Loaded via CUDA driver API (``cuModuleLoad``). Kernels launched via ``cuLaunchKernel``.
   * - ``CSourceModule``
     - C source string
     - Not directly executable. Used as a build artifact for AOT compilation.
   * - ``DeviceSourceModule``
     - Device source string (OpenCL C, Metal, WGSL)
     - Compiled at runtime by the device driver (e.g., ``clCreateProgramWithSource``).

All module types implement the same interface: ``GetFunction(name)`` returns a ``PackedFunc``
that can be called from Python or C++. The VM and other runtime components use this interface
to invoke compiled kernels without knowing which backend produced them.

The module tree is serializable via ``export_library()``, which packs the host module and all
imported device modules into a single shared library (``.so`` / ``.dll`` / ``.dylib``) or
a tar archive for deployment.


Source Code Map
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Path
     - Contents
   * - ``python/tvm/tirx/build.py``
     - ``tirx.build()``: TIR compilation entry, host/device split, module linking
   * - ``src/target/codegen.cc``
     - ``codegen::Build()``: target dispatch via ``"target.build.<kind>"``
   * - ``src/target/llvm/codegen_llvm.h``
     - ``CodeGenLLVM``: TIR → LLVM IR base class
   * - ``src/target/llvm/codegen_cpu.h``
     - ``CodeGenCPU``: CPU-specific LLVM codegen (x86, ARM)
   * - ``src/target/llvm/codegen_nvptx.cc``
     - ``CodeGenNVPTX``: NVIDIA PTX via LLVM
   * - ``src/target/llvm/codegen_amdgpu.cc``
     - ``CodeGenAMDGPU``: AMD GPU via LLVM
   * - ``src/target/llvm/llvm_module.cc``
     - ``LLVMModuleNode``: runtime module with JIT compilation
   * - ``src/target/source/codegen_c.h``
     - ``CodeGenC``: TIR → C-like source base class
   * - ``src/target/source/codegen_cuda.h``
     - ``CodeGenCUDA``: TIR → CUDA C
   * - ``src/target/source/codegen_opencl.h``
     - ``CodeGenOpenCL``: TIR → OpenCL C
   * - ``src/target/source/codegen_metal.h``
     - ``CodeGenMetal``: TIR → Metal Shading Language
   * - ``src/target/source/codegen_c_host.h``
     - ``CodeGenCHost``: TIR → C host code
   * - ``src/target/opt/build_cuda_on.cc``
     - ``BuildCUDA``: CUDA build flow (codegen → compile → module)
   * - ``src/target/spirv/codegen_spirv.h``
     - ``CodeGenSPIRV``: TIR → SPIR-V for Vulkan
