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
`Chris Hoge <https://github.com/hogepodge>`_

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
#. :doc:`Compiling and Optimizing a Model with TVMC <tvmc_command_line_driver>`
#. :doc:`Compiling and Optimizing a Model with the Python AutoScheduler <auto_tuning_with_python>`
#. :doc:`Working with Operators Using Tensor Expressions <tensor_expr_get_started>`
#. :doc:`Optimizing Operators with Templates and AutoTVM <autotvm_matmul>`
#. :doc:`Optimizing Operators with AutoScheduling <tune_matmul_x86>`
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
# .. image:: /_static/img/tvm.png
#   :width: 100%
#   :alt: A High Level View of TVM
#
# 1. Import the model from a framework like *Tensorflow*, *Pytorch*, or *Onnx*.
#    The importer layer is where TVM can ingest models from other frameworks, like
#    ONNX, Tensorflow, or PyTorch. The level of support that TVM offers for each
#    frontend varies as we are constantly improving the open source project. If
#    you're having issues importing your model into TVM, you may want to try
#    converting it to ONNX.
#
# 2. Translate to *Relay,* TVM's high level model language.
#    A model that has been imported into TVM is represented in Relay. Relay is a
#    functional language and intermediate representation (IR) for neural networks.
#    It has support for:
#
#    - Traditional data flow-style representations
#    - Functional-style scoping, let-binding which makes it a fully featured
#      differentiable language
#    - Ability to allow the user to mix the two programming styles
#
#    Relay (or more detailedly, its fusion pass) is in charge of splitting the
#    neural network into small subgraphs, each of which is a task.
#
# 3. Lower to *TE*, tensor expressions that define the *computational
#    operations* of the neural network.
#    Upon completing the import and high level optimizations, the next step is
#    to decide how to implement the Relay representation to a hardware target.
#    Relay (or more specifically, its fusion pass) is in charge of splitting the
#    neural network into small subgraphs, each of which is a task. Here
#    lowering means going lowering into TE tasks. The first step is to lower
#    each task within the Relay model into a tensor expression. The tensor
#    expressions describe the operations, aka functions, contained within a
#    neural network. Once transformed into TE, further optimizations for the
#    specific hardware target can be made. Work is underway to replace TE with
#    a new representation, Tensor Intermediate Representation (TIR), that
#    includes TE as a subset of TIR.
#
# 4. Search for an optimized schedule using *AutoTVM* or *AutoScheduler*.
#    Tuning is the process of searching for a schedule (an ordered
#    notation) for the neural network to be compiled. There are couple of
#    optimization options available, each requiring varying levels of user
#    interaction. Both of these methods can draw from the TVM Operator
#    Inventory (TOPI). TOPI includes pre-defined templates of common machine
#    learning operations. The optimization options include:
#
#    - **AutoTVM**: The user specifies a search template for the schedule of a TE task,
#      or TE subraph. AutoTVM directs the search of the parameter space defined by the
#      template to produce an optimized configuration. AutoTVM requires users to
#      define manually templates for each operator as part of the TOPI.
#    - **Ansor/AutoSchedule**: Using a TVM Operator Inventory (TOPI) of operations,
#      Ansor can automatically search an optimization space with much less
#      intervention and guidance from the end user. Ansor depends on TE templates to
#      guide the search.
#
# 5. Determing optimal schedule. After tuning, a schedule is determined to
#    optimize on. Regardless if it is AutoTVM or AutoSchedule, schedule records in
#    JSON format are produced. Afterwards, the best schedule found is chosen to
#    determine how to optimize each layer of the neural network.
#
# 6. Lower to hardware specific compiler.  TVM tuning operates by computing
#    performance metrics for different operator configurations on the target
#    hardware, then choosing the best configuration in the final code generation
#    phase. This code generation is meant to produce an optimized model that can
#    be deployed into production. TVM supports a number of different compiler
#    backends including:
#
#    - LLVM, which can target arbitrary microprocessor architecture including
#      standard x86 and ARM processors, AMDGPU and NVPTX code generation, and any
#      other platform supported by LLVM.
#    - Source-to-source compilation, such as with NVCC, NVIDIA's compiler.
#    - Embedded and specialized targets, which are implemented through TVM's
#      Bring Your Own Codegen (BYOC) framework.
#
#    TVM can compile models down to a linkable object module, which can then be
#    run with a lightweight TVM runtime that provides C APIs to dynamically
#    load the model, and entry points for other languages such as Python and
#    Rust. TVM can also build a bundled deployment in which the runtime is
#    combined with the model in a single package.
#
# 7. Compile down to machine code. At the end of this process, the
#    compiler-specific generated code can be lowered to machine code.
#
# The remainder of the tutorial will cover these aspects of TVM in more detail.
