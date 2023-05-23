# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Introduction
============
**Authors**:
`Jocelyn Shiue <https://github.com/>`_,
`Chris Hoge <https://github.com/hogepodge>`_,
`Lianmin Zheng <https://github.com/merrymercy>`_

Apache TVM is an open source machine learning compiler framework for CPUs,
GPUs, and machine learning accelerators. It aims to enable machine learning
engineers to optimize and run computations efficiently on any hardware backend.
The purpose of this tutorial is to take a guided tour through all of the major
features of TVM by defining and demonstrating key concepts. A new user should
be able to work through the tutorial from start to finish and be able to
operate TVM for automatic model optimization, while having a basic
understanding of the TVM architecture and how it works.

Contents
--------

#. :doc:`Introduction <introduction>`
#. :doc:`Installing TVM <install>`
#. :doc:`Compiling and Optimizing a Model with the Command Line Interface <tvmc_command_line_driver>`
#. :doc:`Compiling and Optimizing a Model with the Python Interface <autotvm_relay_x86>`
#. :doc:`Working with Operators Using Tensor Expression <tensor_expr_get_started>`
#. :doc:`Optimizing Operators with Templates and AutoTVM <autotvm_matmul_x86>`
#. :doc:`Optimizing Operators with Template-free AutoScheduler <auto_scheduler_matmul_x86>`
#. :doc:`Cross Compilation and Remote Procedure Calls (RPC) <cross_compilation_and_rpc>`
#. :doc:`Compiling Deep Learning Models for GPUs <relay_quick_start>`
"""


################################################################################
# An Overview of TVM and Model Optimization
# =========================================
#
# The diagram below illustrates the steps a machine model takes as it is
# transformed with the TVM optimizing compiler framework.
#
# .. image:: https://raw.githubusercontent.com/apache/tvm-site/main/images/tutorial/overview.png
#   :width: 100%
#   :alt: A High Level View of TVM
#
# 1. Import the model from a framework like *Tensorflow*, *PyTorch*, or *Onnx*.
#    The importer layer is where TVM can ingest models from other frameworks, like
#    Tensorflow, PyTorch, or ONNX. The level of support that TVM offers for each
#    frontend varies as we are constantly improving the open source project. If
#    you're having issues importing your model into TVM, you may want to try
#    converting it to ONNX.
#
# 2. Translate to *Relay*, TVM's high-level model language.
#    A model that has been imported into TVM is represented in Relay. Relay is a
#    functional language and intermediate representation (IR) for neural networks.
#    It has support for:
#
#    - Traditional data flow-style representations
#    - Functional-style scoping, let-binding which makes it a fully featured
#      differentiable language
#    - Ability to allow the user to mix the two programming styles
#
#    Relay applies graph-level optimization passes to optimize the model.
#
# 3. Lower to *Tensor Expression* (TE) representation. Lowering is when a
#    higher-level representation is transformed into a lower-level
#    representation. After applying the high-level optimizations, Relay
#    runs FuseOps pass to partition the model into many small subgraphs and lowers
#    the subgraphs to TE representation. Tensor Expression (TE) is a
#    domain-specific language for describing tensor computations.
#    TE also provides several *schedule* primitives to specify low-level loop
#    optimizations, such as tiling, vectorization, parallelization,
#    unrolling, and fusion.
#    To aid in the process of converting Relay representation into TE representation,
#    TVM includes a Tensor Operator Inventory (TOPI) that has pre-defined
#    templates of common tensor operators (e.g., conv2d, transpose).
#
# 4. Search for the best schedule using the auto-tuning module *AutoTVM* or *AutoScheduler*.
#    A schedule specifies the low-level loop optimizations for an operator or
#    subgraph defined in TE. Auto-tuning modules search for the best schedule
#    and compare them with cost models and on-device measurements.
#    There are two auto-tuning modules in TVM.
#
#    - **AutoTVM**: A template-based auto-tuning module. It runs search algorithms
#      to find the best values for the tunable knobs in a user-defined template.
#      For common operators, their templates are already provided in TOPI.
#    - **AutoScheduler (a.k.a. Ansor)**: A template-free auto-tuning module.
#      It does not require pre-defined schedule templates. Instead, it generates
#      the search space automatically by analyzing the computation definition.
#      It then searches for the best schedule in the generated search space.
#
# 5. Choose the optimal configurations for model compilation. After tuning, the
#    auto-tuning module generates tuning records in JSON format. This step
#    picks the best schedule for each subgraph.
#
# 6. Lower to Tensor Intermediate Representation (TIR), TVM's low-level
#    intermediate representation. After selecting the optimal configurations
#    based on the tuning step, each TE subgraph is lowered to TIR and be
#    optimized by low-level optimization passes. Next, the optimized TIR is
#    lowered to the target compiler of the hardware platform.
#    This is the final code generation phase to produce an optimized model
#    that can be deployed into production. TVM supports several different
#    compiler backends including:
#
#    - LLVM, which can target arbitrary microprocessor architecture including
#      standard x86 and ARM processors, AMDGPU and NVPTX code generation, and any
#      other platform supported by LLVM.
#    - Specialized compilers, such as NVCC, NVIDIA's compiler.
#    - Embedded and specialized targets, which are implemented through TVM's
#      Bring Your Own Codegen (BYOC) framework.
#
# 7. Compile down to machine code. At the end of this process, the
#    compiler-specific generated code can be lowered to machine code.
#
#    TVM can compile models down to a linkable object module, which can then be
#    run with a lightweight TVM runtime that provides C APIs to dynamically
#    load the model, and entry points for other languages such as Python and
#    Rust. TVM can also build a bundled deployment in which the runtime is
#    combined with the model in a single package.
#
# The remainder of the tutorial will cover these aspects of TVM in more detail.
