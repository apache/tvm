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
This tutorial shows how to mix Python functions, TIR kernels, and Relax graph-level functions
in a single ``IRModule`` using the ``BasePyModule`` system. The key benefits are:

- **Debug without compiling**: Run IRModules directly in Python, calling TIR and Relax functions
  through JIT compilation while keeping Python functions as-is.
- **PyTorch interop**: Use PyTorch operators as fallbacks for ops TVM does not yet support,
  with zero-copy DLPack tensor conversion.
- **Relax-to-Python conversion**: Automatically translate compiled Relax functions into equivalent
  PyTorch code for numerical verification at any compilation stage.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Preparation
# -----------
# We import the necessary modules. ``BasePyModule`` is the base class that enables Python function
# integration with TVM's IRModule. The ``I``, ``T``, ``R`` namespaces provide TVMScript decorators
# for IR modules, TIR functions, and Relax functions respectively.

import os

try:
    import torch
except ImportError:
    torch = None

import tvm
from tvm.relax.base_py_module import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

IS_IN_CI = os.getenv("CI", "").lower() == "true"
HAS_TORCH = torch is not None
RUN_EXAMPLE = HAS_TORCH and not IS_IN_CI


######################################################################
# Part 1: BasePyModule Basics
# ----------------------------
# A ``BasePyModule`` wraps an ``IRModule`` and provides:
#
# - Automatic JIT compilation of TIR and Relax functions
# - DLPack-based zero-copy conversion between PyTorch tensors and TVM NDArrays
# - A unified interface where Python, TIR, and Relax functions coexist
#
# Let us start with a simple example: a module that contains one TIR function (element-wise add)
# and one Python function that orchestrates the computation using PyTorch tensors.

if RUN_EXAMPLE:

    @I.ir_module
    class MyModule(BasePyModule):
        @I.pyfunc
        def forward(self, x, y):
            """Python function: receives PyTorch tensors, calls TIR, returns PyTorch tensors."""
            # Convert PyTorch tensors to TVM NDArrays (zero-copy via DLPack)
            x_tvm = self._convert_pytorch_to_tvm(x)
            y_tvm = self._convert_pytorch_to_tvm(y)

            # Call the TIR function below
            result = self.call_tir(
                self.add_tir, [x_tvm, y_tvm], out_sinfo=R.Tensor((4,), "float32")
            )

            # Convert back to PyTorch (zero-copy via DLPack)
            return self._convert_tvm_to_pytorch(result)

        @T.prim_func
        def add_tir(
            A: T.Buffer((4,), "float32"),
            B: T.Buffer((4,), "float32"),
            C: T.Buffer((4,), "float32"),
        ):
            for i in range(4):
                C[i] = A[i] + B[i]

    # Instantiate the module on CPU. TIR functions are JIT-compiled at this point.
    mod = MyModule(device=tvm.cpu(0))

    # Call the Python function with PyTorch tensors
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    result = mod.forward(x, y)

    print("Input x:", x)
    print("Input y:", y)
    print("Result (x + y via TIR):", result)
    assert torch.allclose(result, x + y)

    # BasePyModule also supports pretty-printing via show(), including Python functions
    print("\n=== Module TVMScript ===")
    mod.show()


######################################################################
# How it Works
# ~~~~~~~~~~~~
# When the class is decorated with ``@I.ir_module`` and inherits from ``BasePyModule``:
#
# 1. Methods decorated with ``@T.prim_func`` and ``@R.function`` are parsed into TIR/Relax IR
#    and stored in the underlying ``IRModule``.
# 2. Methods decorated with ``@I.pyfunc`` are registered as Python functions.
# 3. On instantiation (``MyModule(device=...)``), TIR functions are compiled via ``tvm.compile``
#    and Relax functions are loaded into a ``VirtualMachine``. Python functions remain as-is.
# 4. ``call_tir`` handles DLPack conversion, output allocation, and calling the compiled kernel.
#


######################################################################
# Part 2: Mixing TIR, Relax, and Python
# ----------------------------------------
# A single module can contain all three kinds of functions. This is useful when some operations
# are best expressed as low-level TIR kernels, others as high-level Relax graphs, and some
# require Python-level logic (e.g., dynamic control flow, calling external libraries).

if RUN_EXAMPLE:

    @I.ir_module
    class HybridModule(BasePyModule):
        @I.pyfunc
        def preprocess(self, x):
            """Use PyTorch for preprocessing — e.g., normalization."""
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-5)

        @I.pyfunc
        def run_pipeline(self, x):
            """Orchestrate: Python preprocessing -> TIR computation -> result."""
            # Step 1: Python-based preprocessing
            normalized = self.preprocess(x)

            # Step 2: Convert and run TIR kernel
            tvm_input = self._convert_pytorch_to_tvm(normalized)
            tvm_result = self.call_tir(
                self.scale_tir, [tvm_input], out_sinfo=R.Tensor((4,), "float32")
            )
            return self._convert_tvm_to_pytorch(tvm_result)

        @T.prim_func
        def scale_tir(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
            for i in range(4):
                B[i] = A[i] * T.float32(2.0)

    mod = HybridModule(device=tvm.cpu(0))

    x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    result = mod.run_pipeline(x)
    print("Pipeline result:", result)


######################################################################
# Part 3: Adding Python Functions Dynamically
# ---------------------------------------------
# You can also register Python functions after module creation using ``add_python_function``.
# This is useful for attaching PyTorch-based fallback operators or custom post-processing.

if RUN_EXAMPLE:
    py_mod = BasePyModule(tvm.IRModule({}), device=tvm.cpu(0))

    # Dynamically add a Python function
    def my_activation(x):
        """Custom activation using PyTorch."""
        return torch.relu(x) + torch.tanh(x)

    py_mod.add_python_function("my_activation", my_activation)

    # Now we can call it as a method
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = py_mod.my_activation(x)
    print("Custom activation:", result)
    expected = torch.relu(x) + torch.tanh(x)
    assert torch.allclose(result, expected)


######################################################################
# Part 4: Relax-to-Python Function Converter
# --------------------------------------------
# One powerful feature is the ability to automatically convert Relax functions into equivalent
# PyTorch code. This is useful for:
#
# - Numerically verifying Relax IR against PyTorch after applying optimization passes
# - Debugging: inspect what a Relax function actually computes by running it in Python
# - Prototyping: test Relax graph transformations without a full compilation cycle
#
# The ``RelaxToPyFuncConverter`` maps 300+ Relax operators to their PyTorch equivalents.

if RUN_EXAMPLE:
    from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter

    @I.ir_module
    class RelaxModel:
        @T.prim_func
        def custom_add(var_x: T.handle, var_y: T.handle, var_out: T.handle):
            x = T.match_buffer(var_x, (5,), "float32")
            y = T.match_buffer(var_y, (5,), "float32")
            out = T.match_buffer(var_out, (5,), "float32")
            for i in range(5):
                out[i] = x[i] + y[i]

        @R.function
        def main(
            x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")
        ) -> R.Tensor((5,), "float32"):
            # Mix of Relax ops and TIR calls
            added = R.add(x, y)
            activated = R.nn.relu(added)
            cls = RelaxModel
            result = R.call_tir(cls.custom_add, (activated, y), out_sinfo=R.Tensor((5,), "float32"))
            return result

    # Convert the Relax function "main" to an equivalent Python/PyTorch function
    converter = RelaxToPyFuncConverter(RelaxModel)
    converted_mod = converter.convert(["main"])

    # The converted function lives in ir_mod.pyfuncs and accepts PyTorch tensors directly
    x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    y = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

    py_result = converted_mod.pyfuncs["main"](x, y)

    # Manually compute the expected result for verification
    step1 = torch.add(x, y)  # [1.5, -1.5, 3.5, -3.5, 5.5]
    step2 = torch.relu(step1)  # [1.5, 0.0, 3.5, 0.0, 5.5]
    expected = step2 + y  # [2.0, 0.5, 4.0, 0.5, 6.0]

    print("Relax function converted to Python:")
    print("  Input x:", x)
    print("  Input y:", y)
    print("  Python result:", py_result)
    print("  Expected:     ", expected)
    assert torch.allclose(py_result, expected)


######################################################################
# Part 5: Using R.call_py_func in Relax IR
# ------------------------------------------
# ``R.call_py_func`` lets you embed Python function calls directly inside Relax IR. This means
# the compiled Relax VM can call back into Python at runtime. This is the bridge for ops that
# TVM cannot compile natively — the rest of the graph is compiled and optimized, while specific
# ops fall back to Python/PyTorch.
#
# .. note::
#    ``R.call_py_func`` adds runtime overhead due to the Python-TVM boundary crossing.
#    Use it for prototyping or for ops that are not performance-critical.
#
# Here is an example using ``call_py_func`` inside a Relax function:
#
# .. code-block:: python
#
#    @I.ir_module
#    class CallPyFuncModule(BasePyModule):
#        @I.pyfunc
#        def my_custom_op(self, x):
#            """Python fallback for a custom op."""
#            return torch.sigmoid(x) * x  # SiLU / Swish activation
#
#        @R.function
#        def main(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
#            # Call the Python function from within Relax IR
#            result = R.call_py_func(
#                "my_custom_op", (x,), out_sinfo=R.Tensor((4,), "float32")
#            )
#            return result
#
#    mod = CallPyFuncModule(device=tvm.cpu(0))
#    x = torch.tensor([1.0, -1.0, 2.0, -2.0])
#    result = mod.main(x)
#
# The VM executes the compiled Relax bytecode, and when it hits ``call_py_func``, it looks up
# the registered Python function by name and calls it with DLPack-converted tensors.


######################################################################
# Summary
# -------
# This tutorial covered the Relax Python Module system:
#
# - **BasePyModule**: A base class that unifies Python, TIR, and Relax functions in one module,
#   with JIT compilation and DLPack-based tensor conversion.
# - **@I.pyfunc**: Decorator to mark Python functions inside an ``@I.ir_module`` class.
# - **Dynamic registration**: ``add_python_function()`` to attach Python functions after creation.
# - **RelaxToPyFuncConverter**: Automatically converts Relax functions to PyTorch for debugging
#   and numerical verification.
# - **R.call_py_func**: Embeds Python function calls in Relax IR, enabling fallback to PyTorch
#   for unsupported ops while keeping the rest of the graph compiled.
#
# Together, these features make TVM a hybrid execution framework where you can freely mix
# compiled TVM code and Python/PyTorch, enabling faster iteration during development
# and gradual migration of ops from Python fallbacks to optimized TVM kernels.
