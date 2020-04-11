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


Frequently Asked Questions
==========================


How to Install
--------------
See :ref:`installation`.


How to add a new Hardware Backend
---------------------------------

- If the hardware backend has LLVM support,
  then we can directly generate the code by setting the correct target triple as in :py:mod:`~tvm.target`.
- If the target hardware is a GPU, try to use the cuda, opencl or vulkan backend.
- If the target hardware is a special accelerator,
  checkout :ref:`vta-index` and :ref:`relay-bring-your-own-codegen`.
- For all of the above cases, You may want to add target specific
  optimization templates using AutoTVM, see :ref:`tutorials-autotvm-sec`.
- Besides using LLVM's vectorization, we can also embed micro-kernels to leverage hardware intrinsics,
  see :ref:`tutorials-tensorize`.


TVM's relation to Other IR/DSL Projects
---------------------------------------
There are usually two levels of abstractions of IR in the deep learning systems.
TensorFlow's XLA and Intel's ngraph both use a computation graph representation.
This representation is high level, and can be helpful to perform generic optimizations
such as memory reuse, layout transformation and automatic differentiation.

TVM adopts a low-level representation, that explicitly express the choice of memory
layout, parallelization pattern, locality and hardware primitives etc.
This level of IR is closer to directly target hardwares.
The low-level IR adopts ideas from existing image processing languages like Halide, darkroom
and loop transformation tools like loopy and polyhedra-based analysis.
We specifically focus on expressing deep learning workloads (e.g. recurrence),
optimization for different hardware backends and embedding with frameworks to provide
end-to-end compilation stack.


TVM's relation to libDNN, cuDNN
-------------------------------
TVM can incorporate these libraries as external calls. One goal of TVM is to be able to
generate high-performing kernels. We will evolve TVM an incremental manner as
we learn from the techniques of manual kernel crafting and add these as primitives in DSL.
See also top for recipes of operators in TVM.


Security
--------
See :ref:`dev-security`
