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
# .. image:: https://raw.githubusercontent.com/hogepodge/web-data/c339ebbbae41f3762873147c1e920a53a08963dd/images/getting_started/overview.png
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
# 2. Translate to *Relay*, TVM's high level model language.
#    A model that has been imported into TVM is represented in Relay. Relay is a
#    functional language and intermediate representation (IR) for neural networks.
#    It has support for:
#
#    - Traditional data flow-style representations
#    - Functional-style scoping, let-binding which makes it a fully featured
#      differentiable language
#    - Ability to allow the user to mix the two programming styles
#
#    Relay applies several high-level optimization to the model, after which
#    is runs the Relay Fusion Pass. To aid in the process of converting to
#    Relay, TVM includes a Tensor Operator Inventory (TOPI) that has pre-defined
#    templates of common computations.
#
# 3. Lower to *Tensor Expression* (TE) representation. Lowering is when a
#    higher-level representation is transformed into a lower-level
#    representation. In Relay Fusion Pass, the model is lowered from the
#    higher-level Relay representation into a smaller set of subgraphs, where
#    each node is a task. A task is a collection of computation templates,
#    expressed in TE, where there parameters of the template can control how
#    the computation is carried out on hardware. The specific ordering of compuation,
#    defined by parameters to the TE template, is called a schedule.
#
# 4. Search for optimized schedule using *AutoTVM* or *AutoScheduler* for each
#    task through tuning. Tuning is the process of searching the TE parameter
#    space for a schedule that is optimized for target hardware. There are
#    couple of optimization options available, each requiring varying levels of
#    user interaction. The optimization options include:
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
# 5. Choose the optimal configuration for the model. After tuning, an optimal schedule
#    for each task is chosen. Regardless if it is AutoTVM or AutoSchedule,
#    schedule records in JSON format are produced that are referred to by this step
#    to build an optimized model.
#
# 6. Lower to a hardware specific compiler. After selecting an optimized configuration
#    based on the tuning step, the model is then lowered to a representation
#    expected by the target compiler for the hardware platform. This is the
#    final code generation phase with the intention of producing an optimized
#    model that can be deployed into production. TVM supports a number of
#    different compiler backends including:
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
