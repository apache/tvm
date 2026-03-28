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
# ruff: noqa: E402

"""
.. _mix_python_and_tvm:

Mix Python/PyTorch with TVM Using BasePyModule
===============================================
In a typical TVM workflow, you write an ``IRModule``, compile it, and load the compiled artifact
into a ``VirtualMachine`` to run. This means **you cannot test or debug anything until the entire
module compiles successfully**. If a single op is unsupported, the whole pipeline is blocked.

``BasePyModule`` solves this by letting Python functions, TIR kernels, and Relax functions coexist
in one module. TIR and Relax functions are JIT-compiled, Python functions run as-is, and tensors
move between TVM and PyTorch via zero-copy DLPack. This enables:

- **Incremental development**: get a model running with Python fallbacks first, then replace them
  with TVM ops one by one.
- **Debugging at any stage**: convert Relax functions back to PyTorch to verify numerical
  correctness after applying optimization passes.
- **Hybrid execution**: let the compiled VM call back into Python for ops that are hard to
  express in TIR or Relax.

This tutorial walks through a concrete example: building a small model where Python, TIR, and
Relax functions work together, then using the converter and ``call_py_func`` to debug and extend it.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Preparation
# -----------

import os

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

import tvm
from tvm import relax
from tvm.relax.base_py_module import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

IS_IN_CI = os.getenv("CI", "").lower() == "true"
HAS_TORCH = torch is not None
RUN_EXAMPLE = HAS_TORCH and not IS_IN_CI


######################################################################
# Step 1: Your First Hybrid Module
# ----------------------------------
# The core idea: decorate a class with ``@I.ir_module``, inherit from ``BasePyModule``, and use
# three decorators for three kinds of functions:
#
# - ``@T.prim_func`` — low-level TIR kernel (compiled)
# - ``@R.function`` — high-level Relax graph (compiled)
# - ``@I.pyfunc`` — plain Python (runs as-is, can call PyTorch)
#
# On instantiation, TIR and Relax functions are JIT-compiled. Python functions stay in Python.
# ``call_tir`` bridges them: it converts PyTorch tensors to TVM via DLPack, allocates the output,
# calls the compiled kernel, and converts back.

if RUN_EXAMPLE:

    @I.ir_module
    class MyFirstModule(BasePyModule):
        @T.prim_func
        def add_tir(
            A: T.Buffer((4,), "float32"),
            B: T.Buffer((4,), "float32"),
            C: T.Buffer((4,), "float32"),
        ):
            for i in range(4):
                C[i] = A[i] + B[i]

        @I.pyfunc
        def forward(self, x, y):
            """Takes PyTorch tensors, calls TIR, returns PyTorch tensors."""
            x_tvm = self._convert_pytorch_to_tvm(x)
            y_tvm = self._convert_pytorch_to_tvm(y)
            result = self.call_tir(
                self.add_tir, [x_tvm, y_tvm], out_sinfo=R.Tensor((4,), "float32")
            )
            return self._convert_tvm_to_pytorch(result)

    # TIR functions are JIT-compiled here
    mod = MyFirstModule(device=tvm.cpu(0))

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    result = mod.forward(x, y)

    print("forward(x, y) =", result)
    assert torch.allclose(result, x + y)

    # show() prints the TVMScript representation, including Python functions as ExternFunc
    mod.show()


######################################################################
# Step 2: A Realistic Pipeline — Python, TIR, and Relax Together
# -----------------------------------------------------------------
# Real models are not just one op. Here we build a mini inference pipeline:
#
# 1. **Python** preprocesses the input (normalization — easy in PyTorch, verbose in TIR)
# 2. **TIR** runs a hand-written matmul kernel
# 3. **Python** applies softmax via PyTorch (a temporary fallback)
#
# The key point: you do not need every op to be a TIR kernel to get the module running.
# Write what you can in TIR, fall back to Python for the rest, and iterate.

if RUN_EXAMPLE:

    @I.ir_module
    class InferenceModule(BasePyModule):
        @T.prim_func
        def matmul_tir(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            n = T.int32()
            A = T.match_buffer(var_A, (n, 8), "float32")
            B = T.match_buffer(var_B, (8, 4), "float32")
            C = T.match_buffer(var_C, (n, 4), "float32")
            for i, j, k in T.grid(n, 4, 8):
                with T.sblock("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @I.pyfunc
        def preprocess(self, x):
            """Normalize input — trivial in PyTorch, annoying in TIR."""
            return (x - x.mean()) / (x.std() + 1e-5)

        @I.pyfunc
        def forward(self, x, weights):
            # Step 1: Python preprocessing
            x_norm = self.preprocess(x)

            # Step 2: TIR matmul
            x_tvm = self._convert_pytorch_to_tvm(x_norm)
            w_tvm = self._convert_pytorch_to_tvm(weights)
            out = self.call_tir(
                self.matmul_tir,
                [x_tvm, w_tvm],
                out_sinfo=R.Tensor((x.shape[0], 4), "float32"),
            )
            logits = self._convert_tvm_to_pytorch(out)

            # Step 3: Python softmax (fallback — could be replaced with TIR later)
            return F.softmax(logits, dim=-1)

    mod = InferenceModule(device=tvm.cpu(0))

    batch = torch.randn(2, 8)
    weights = torch.randn(8, 4)
    probs = mod.forward(batch, weights)

    print("Input shape:", batch.shape)
    print("Output probs:", probs)
    print("Probs sum per row:", probs.sum(dim=-1))  # should be ~1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


######################################################################
# Step 3: Dynamic Function Registration
# ----------------------------------------
# Sometimes you want to add a Python function after the module is created — for example, to
# swap in a different activation function or to register a custom op at runtime. Use
# ``add_python_function`` for this.

if RUN_EXAMPLE:
    mod.add_python_function("gelu", lambda x: F.gelu(x))

    x = torch.randn(4)
    result = mod.gelu(x)
    print("Dynamically registered gelu:", result)
    assert torch.allclose(result, F.gelu(x))


######################################################################
# Step 4: Relax-to-Python Converter for Debugging
# --------------------------------------------------
# After importing a model or applying passes, you end up with Relax IR. How do you know the IR
# is numerically correct? The ``RelaxToPyFuncConverter`` translates Relax functions into equivalent
# PyTorch code so you can compare outputs directly.
#
# This is especially useful after running optimization passes: convert the optimized Relax
# function back to PyTorch and compare against the original model's output.

if RUN_EXAMPLE:
    from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter

    @I.ir_module
    class RelaxModel:
        @T.prim_func
        def bias_add_tir(var_x: T.handle, var_b: T.handle, var_out: T.handle):
            x = T.match_buffer(var_x, (2, 4), "float32")
            b = T.match_buffer(var_b, (4,), "float32")
            out = T.match_buffer(var_out, (2, 4), "float32")
            for i, j in T.grid(2, 4):
                out[i, j] = x[i, j] + b[j]

        @R.function
        def main(
            x: R.Tensor((2, 4), "float32"),
            w: R.Tensor((4, 4), "float32"),
            b: R.Tensor((4,), "float32"),
        ) -> R.Tensor((2, 4), "float32"):
            # matmul + bias + relu — a typical dense layer
            h = R.matmul(x, w)
            cls = RelaxModel
            h_bias = R.call_tir(
                cls.bias_add_tir, (h, b), out_sinfo=R.Tensor((2, 4), "float32")
            )
            return R.nn.relu(h_bias)

    # Convert "main" to a Python/PyTorch function
    converter = RelaxToPyFuncConverter(RelaxModel)
    converted = converter.convert(["main"])

    # Run through the converted Python function
    x = torch.randn(2, 4)
    w = torch.randn(4, 4)
    b = torch.randn(4)

    py_result = converted.pyfuncs["main"](x, w, b)

    # Compare with manual PyTorch computation
    expected = F.relu(x @ w + b)

    print("Converted Python result:", py_result)
    print("PyTorch expected:       ", expected)
    assert torch.allclose(py_result, expected, atol=1e-5)


######################################################################
# Step 5: R.call_py_func — Python Callbacks in Compiled IR
# -----------------------------------------------------------
# What if you want the compiled Relax VM (not just Python-side code) to call a Python function?
# ``R.call_py_func`` embeds a Python callback directly in Relax IR. The VM compiles and
# optimizes everything else, but calls back into Python for the specified op.
#
# Use case: your model has one custom op that is complex to implement in TIR. Compile
# everything else for performance, and let that one op run in Python.

if RUN_EXAMPLE:

    @I.ir_module
    class HybridVMModule(BasePyModule):
        @I.pyfunc
        def silu(self, x):
            """SiLU activation — not yet a native Relax op, so we use Python."""
            return torch.sigmoid(x) * x

        @I.pyfunc
        def layer_norm(self, x):
            """LayerNorm — another Python fallback."""
            return F.layer_norm(x, x.shape[-1:])

        @R.function
        def main(x: R.Tensor((4, 8), "float32")) -> R.Tensor((4, 8), "float32"):
            h = R.call_py_func(
                "layer_norm", (x,), out_sinfo=R.Tensor((4, 8), "float32")
            )
            out = R.call_py_func(
                "silu", (h,), out_sinfo=R.Tensor((4, 8), "float32")
            )
            return out

    mod = HybridVMModule(device=tvm.cpu(0))

    x = torch.randn(4, 8)

    # call_py_func is callable from Python too
    result = mod.call_py_func("layer_norm", [x])
    result = mod.call_py_func("silu", [result])

    expected = torch.sigmoid(F.layer_norm(x, x.shape[-1:])) * F.layer_norm(
        x, x.shape[-1:]
    )
    print("call_py_func result:", result)
    assert torch.allclose(torch.tensor(result.numpy()), expected, atol=1e-5)


######################################################################
# Step 6: Symbolic Shapes — Dynamic Batch Sizes
# ------------------------------------------------
# Real models have dynamic shapes (e.g., variable batch size). TIR and Relax functions can
# declare symbolic dimensions. ``BasePyModule`` automatically infers concrete shapes from the
# input tensors at call time.

if RUN_EXAMPLE:

    @I.ir_module
    class DynamicModule(BasePyModule):
        @T.prim_func
        def scale_tir(var_x: T.handle, var_out: T.handle):
            n = T.int64()
            x = T.match_buffer(var_x, (n,), "float32")
            out = T.match_buffer(var_out, (n,), "float32")
            for i in T.serial(n):
                out[i] = x[i] * T.float32(2.0)

        @R.function
        def add_relax(
            x: R.Tensor(("n",), "float32"), y: R.Tensor(("n",), "float32")
        ) -> R.Tensor(("n",), "float32"):
            return R.add(x, y)

    mod = DynamicModule(device=tvm.cpu(0), target="llvm")

    # Works with length 5
    a5 = torch.randn(5)
    b5 = torch.randn(5)
    out5 = mod.add_relax(a5, b5)
    print("add_relax(len=5):", out5)

    # Same module, now length 10 — no recompilation needed
    a10 = torch.randn(10)
    b10 = torch.randn(10)
    out10 = mod.add_relax(a10, b10)
    print("add_relax(len=10):", out10)

    # call_tir with symbolic output shape
    n = T.int64()
    x7 = torch.randn(7)
    scaled = mod.call_tir("scale_tir", [x7], relax.TensorStructInfo((n,), "float32"))
    print("scale_tir(len=7):", scaled)
    assert torch.allclose(
        torch.tensor(scaled.numpy()), x7 * 2.0, atol=1e-5
    )


######################################################################
# Summary
# -------
# Here is what each step demonstrated and which PRs implement it:
#
# +--------+----------------------------------------+---------------------+
# | Step   | What you learned                       | Key PRs             |
# +========+========================================+=====================+
# | 1      | ``@I.pyfunc`` + ``call_tir`` basics,   | #18229, #18331      |
# |        | DLPack conversion, ``show()``          | #18253              |
# +--------+----------------------------------------+---------------------+
# | 2      | Realistic pipeline: Python preprocess  | #18229              |
# |        | → TIR kernel → Python fallback         |                     |
# +--------+----------------------------------------+---------------------+
# | 3      | ``add_python_function`` for runtime     | #18229              |
# |        | registration                           |                     |
# +--------+----------------------------------------+---------------------+
# | 4      | ``RelaxToPyFuncConverter``: verify      | #18269, #18301      |
# |        | Relax IR numerically against PyTorch   |                     |
# +--------+----------------------------------------+---------------------+
# | 5      | ``R.call_py_func``: Python callbacks   | #18313, #18326      |
# |        | inside compiled Relax VM               |                     |
# +--------+----------------------------------------+---------------------+
# | 6      | Symbolic shapes for dynamic inputs     | #18288              |
# +--------+----------------------------------------+---------------------+
#
# The workflow in practice:
#
# 1. Import a model → some ops unsupported → use ``@I.pyfunc`` as Python fallbacks
# 2. Get it running end-to-end with ``BasePyModule``
# 3. Use ``RelaxToPyFuncConverter`` to verify correctness after optimization passes
# 4. Gradually replace Python fallbacks with TIR/Relax implementations
# 5. Use ``R.call_py_func`` for ops that must stay in Python even after compilation
