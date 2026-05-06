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
# pylint: disable=invalid-name
"""
ONNX Backend Tests
===================
Systematically verify the Relax ONNX importer using the official ONNX
Backend Test Suite (node-level tests only).  Each test loads a small
ONNX model with protobuf reference inputs/outputs and checks that the
Relax-imported model produces numerically correct results.

Only ``onnx.backend.test.data.node`` tests are registered here; real,
simple, and PyTorch model tests are out of scope for importer-level
semantic verification.

"""

import numpy as np
import onnx
import onnx.backend.test
from onnx.backend.base import Backend, BackendRep

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------


class TVMRelaxBackendRep(BackendRep):
    """Compiled Relax VM representation for running an ONNX model."""

    def __init__(self, mod, params, func_param_names, graph_input_names):
        super().__init__()
        self._params = params
        self._func_param_names = func_param_names
        self._graph_input_names = graph_input_names

        with tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(mod, target="llvm")
        self._vm = relax.VirtualMachine(ex, tvm.cpu())

    def run(self, inputs, **kwargs):
        # Map positional inputs to names.  The runner loads one .pb per
        # non-initializer input, aligned with model.graph.input order.
        input_map = {}
        for i, arr in enumerate(inputs):
            if i < len(self._graph_input_names):
                input_map[self._graph_input_names[i]] = arr

        # Build the argument list matching the Relax function's param order:
        # user inputs first, then weight params from self._params.
        input_list = []
        for name in self._func_param_names:
            if name in input_map:
                input_list.append(input_map[name])
        if self._params and "main" in self._params:
            input_list += self._params["main"]

        self._vm.set_input("main", *input_list)
        self._vm.invoke_stateful("main")
        output = self._vm.get_outputs("main")

        if isinstance(output, (tvm.runtime.Tensor, np.ndarray)):
            return (output.numpy() if hasattr(output, "numpy") else output,)
        if isinstance(output, (tuple, list)):
            return tuple(
                o.numpy() if hasattr(o, "numpy") else np.array(o) for o in output
            )
        return (np.array(output),)


class TVMRelaxBackend(Backend):
    """ONNX backend that imports models through Relax's ONNX frontend."""

    @classmethod
    def is_compatible(cls, model, device="CPU", **kwargs):
        return True

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        opset = None
        for opset_import in model.opset_import:
            if opset_import.domain in ("", "ai.onnx"):
                opset = opset_import.version
                break

        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
        tvm_model = relax.transform.LegalizeOps()(tvm_model)
        tvm_model, params = relax.frontend.detach_params(tvm_model)

        func = tvm_model["main"]
        func_param_names = [p.name_hint for p in func.params]
        graph_input_names = [inp.name for inp in model.graph.input]

        return TVMRelaxBackendRep(
            tvm_model, params, func_param_names, graph_input_names
        )

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device == "CPU"


# ---------------------------------------------------------------------------
# Test registration
# ---------------------------------------------------------------------------

backend_test = onnx.backend.test.BackendTest(TVMRelaxBackend, __name__)

# Operators where ALL ONNX node tests pass on the Relax importer.
# Each prefix covers the base test and all its variants
# (e.g. test_add, test_add_bcast, test_add_uint8).
#
# Operators not listed here have known importer gaps or have not yet been
# validated against the ONNX Backend Test Suite.  They can be added
# incrementally as the importer improves.
_INCLUDE_OPS = [
    "abs", "acos", "acosh", "add", "and", "argmax", "argmin",
    "averagepool", "bitshift",
    "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor",
    "ceil", "clip", "compress", "concat",
    "conv", "cos", "cosh",
    "depthtospace", "div",
    "einsum", "erf", "exp",
    "flatten", "floor",
    "gathernd", "gemm",
    "globalaveragepool", "globalmaxpool", "greater", "greater_equal",
    "hardmax", "hardswish",
    "isnan",
    "less", "less_equal", "lrn",
    "matmul", "matmulinteger", "mean", "min", "mod", "mul", "neg",
    "nonzero", "not",
    "or",
    "reciprocal",
    "round",
    "scatternd",
    "sigmoid", "sign",
    "sin", "sinh", "size", "slice",
    "spacetodepth",
    "sqrt", "squeeze", "sub", "sum",
    "tan", "tanh", "tile", "transpose",
    "unique", "unsqueeze",
    "where", "xor",
]

for _op in _INCLUDE_OPS:
    backend_test.include(rf"^test_{_op}(?:_.*)?(?:_cpu|_cuda)$")

globals().update(backend_test.test_cases)
