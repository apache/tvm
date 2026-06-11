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
.. _tutorial-bring-your-own-codegen:

Bring Your Own Codegen: NPU Backend Example
===========================================

This tutorial shows how to integrate a custom hardware backend with TVM's
BYOC framework, using the bundled example NPU backend (CPU emulation, no
real hardware required) as the worked example.  You will see the key
concepts needed to offload operations to a custom accelerator: pattern
registration, graph partitioning, codegen, and runtime dispatch.

NPUs are purpose-built accelerators designed around a fixed set of operations
common in neural network inference, such as matrix multiplication, convolution,
and activation functions.
The example backend's runtime is a *stub*: it logs the dispatch decisions an
NPU would make (memory tier, execution engine, fusion) but performs no real
computation, so output buffers are uninitialized.  Assertions in this tutorial
therefore check shapes, not values.  When you replace the runtime with your
hardware SDK calls, the same flow produces real results.

**Prerequisites**: Build TVM with ``USE_EXAMPLE_NPU_CODEGEN=ON`` and
``USE_EXAMPLE_NPU_RUNTIME=ON``.
"""

######################################################################
# Overview of the BYOC Flow
# -------------------------
#
# The BYOC framework lets you plug a custom backend into TVM's compilation
# pipeline in four steps:
#
# 1. **Register patterns** - describe which sequences of Relax ops the
#    backend can handle.
# 2. **Partition the graph** - group matched ops into composite functions.
# 3. **Run codegen** - lower composite functions to backend-specific
#    representation (JSON graph for the example NPU).
# 4. **Execute** - the runtime dispatches composite functions to the
#    registered backend runtime.

######################################################################
# Step 1: Import the backend to register its patterns
# ---------------------------------------------------
#
# Importing the module is enough to register all supported patterns with
# TVM's pattern registry.

import numpy as np

import tvm
import tvm.relax.backend.contrib.example_npu  # registers patterns
from tvm import relax
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen
from tvm.script import relax as R

has_example_npu_codegen = tvm.get_global_func("relax.ext.example_npu", True)
has_example_npu_runtime = tvm.get_global_func("runtime.ExampleNPUJSONRuntimeCreate", True)
has_example_npu = has_example_npu_codegen and has_example_npu_runtime

target = tvm.target.Target("llvm")

patterns = get_patterns_with_prefix("example_npu")
print("Registered patterns:", [p.name for p in patterns])

######################################################################
# Step 2: Define a model
# ----------------------
#
# We use a simple MatMul + ReLU module to illustrate the flow.


@tvm.script.ir_module
class MatmulReLU:
    @R.function
    def main(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 8), "float32"),
    ) -> R.Tensor((2, 8), "float32"):
        with R.dataflow():
            y = relax.op.matmul(x, w)
            z = relax.op.nn.relu(y)
            R.output(z)
        return z


######################################################################
# Step 3: Partition the graph
# ---------------------------
#
# ``FuseOpsByPattern`` groups ops that match a registered pattern into
# composite functions, controlled by two flags:
#
# - ``bind_constants=False`` keeps weights as function arguments instead
#   of baking them in, so the host stays in charge of parameter
#   ownership.
# - ``annotate_codegen=True`` tags each composite with its backend name
#   (``example_npu``); without this tag, ``RunCodegen`` has no way to
#   route the composite to a backend.
#
# ``MergeCompositeFunctions`` then consolidates adjacent composites
# that target the same backend so each group becomes a single external
# call.  Note that consolidation depends on the patterns themselves: an
# ``op_a + op_b`` chain only collapses into one composite if a fused
# pattern (e.g. ``matmul_relu_fused``) was registered for it; otherwise
# each op stays as its own composite even when both target the same
# backend.

mod = MatmulReLU
mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod = MergeCompositeFunctions()(mod)
print("After partitioning:")
print(mod)

######################################################################
# Step 4: Run codegen
# -------------------
#
# ``RunCodegen`` lowers each annotated composite function to the backend's
# serialization format.  For the example NPU this produces a JSON graph
# that the C++ runtime can execute.
#
# Steps 4 and 5 require TVM to be built with ``USE_EXAMPLE_NPU_CODEGEN=ON``
# and ``USE_EXAMPLE_NPU_RUNTIME=ON``.

if has_example_npu:
    mod = RunCodegen()(mod)
    print("After codegen:")
    print(mod)

    ######################################################################
    # Step 5: Build and run
    # ---------------------
    #
    # Build the module for the host target, create a virtual machine, and
    # execute the compiled function.

    np.random.seed(0)
    x_np = np.random.randn(2, 4).astype("float32")
    w_np = np.random.randn(4, 8).astype("float32")

    with tvm.transform.PassContext(opt_level=3):
        built = relax.build(mod, target)

    vm = relax.VirtualMachine(built, tvm.cpu())
    result = vm["main"](tvm.runtime.tensor(x_np, tvm.cpu()), tvm.runtime.tensor(w_np, tvm.cpu()))

    assert result.numpy().shape == (2, 8)
    print("Execution completed. Output shape:", result.numpy().shape)

######################################################################
# Step 6: Conv2D + ReLU
# ---------------------
#
# The same flow applies to convolution workloads.  Because the fused
# ``conv2d + relu`` pattern is registered after the standalone
# ``conv2d`` pattern in ``patterns.py`` (later entries have higher
# priority), both ops are offloaded as a single composite function.


@tvm.script.ir_module
class Conv2dReLU:
    @R.function
    def main(
        x: R.Tensor((1, 3, 32, 32), "float32"),
        w: R.Tensor((16, 3, 3, 3), "float32"),
    ) -> R.Tensor((1, 16, 30, 30), "float32"):
        with R.dataflow():
            y = relax.op.nn.conv2d(x, w)
            z = relax.op.nn.relu(y)
            R.output(z)
        return z


if has_example_npu:
    mod2 = Conv2dReLU
    mod2 = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod2)
    mod2 = MergeCompositeFunctions()(mod2)
    mod2 = RunCodegen()(mod2)

    with tvm.transform.PassContext(opt_level=3):
        built2 = relax.build(mod2, target)

    x2_np = np.random.randn(1, 3, 32, 32).astype("float32")
    w2_np = np.random.randn(16, 3, 3, 3).astype("float32")

    vm2 = relax.VirtualMachine(built2, tvm.cpu())
    result2 = vm2["main"](
        tvm.runtime.tensor(x2_np, tvm.cpu()), tvm.runtime.tensor(w2_np, tvm.cpu())
    )
    assert result2.numpy().shape == (1, 16, 30, 30)
    print("Conv2dReLU output shape:", result2.numpy().shape)

######################################################################
# Next steps
# ----------
#
# To build a real NPU backend using this example as a starting point:
#
# - Replace ``example_npu_runtime.cc`` with your hardware SDK calls.
# - Extend ``patterns.py`` with the ops your hardware supports.
# - Add a C++ codegen under ``src/relax/backend/contrib/`` if your
#   hardware requires a non-JSON serialization format.
# - Add your cmake module under ``cmake/modules/contrib/`` following
#   the pattern in ``cmake/modules/contrib/ExampleNPU.cmake``.
