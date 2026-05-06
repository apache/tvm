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
ONNX Backend Tests for Relax ONNX Frontend
===========================================

Uses the official ONNX Backend Test Suite to systematically verify the
Relax ONNX importer against the ONNX specification.

Phase 1 (PoC): simple element-wise operators only.
"""

import numpy as np
import onnx
import onnx.backend.test
from onnx.backend.base import Backend, BackendRep

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


class TVMRelaxBackendRep(BackendRep):
    """Compiled Relax VM representation for running ONNX models."""

    def __init__(self, mod, params, func_param_names, graph_input_names):
        super().__init__()
        self._mod = mod
        self._params = params
        self._func_param_names = func_param_names
        self._graph_input_names = graph_input_names

        with tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(mod, target="llvm")
        self._vm = relax.VirtualMachine(ex, tvm.cpu())

    def run(self, inputs, **kwargs):
        # Build a name -> array mapping from the positional inputs list.
        # The runner loads inputs matching model.graph.input order, but only
        # for the number of .pb files found (non-initializer inputs).
        input_map = {}
        for i, arr in enumerate(inputs):
            if i < len(self._graph_input_names):
                input_map[self._graph_input_names[i]] = arr

        # Build input_list matching the Relax function's param order.
        # User inputs come first (by func_param_names[:num_input]),
        # then weight params from self._params.
        input_list = []
        for name in self._func_param_names:
            if name in input_map:
                input_list.append(input_map[name])

        if self._params and "main" in self._params:
            input_list += self._params["main"]

        self._vm.set_input("main", *input_list)
        self._vm.invoke_stateful("main")
        output = self._vm.get_outputs("main")

        # Normalize output to tuple of numpy arrays.
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
        # Extract opset version.
        opset = None
        for opset_import in model.opset_import:
            if opset_import.domain in ("", "ai.onnx"):
                opset = opset_import.version
                break

        # Import ONNX model into Relax.
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
        tvm_model = relax.transform.LegalizeOps()(tvm_model)
        tvm_model, params = relax.frontend.detach_params(tvm_model)

        # Collect function parameter names (user inputs) and graph input names.
        func = tvm_model["main"]
        func_param_names = [p.name_hint for p in func.params]

        # Graph input names from the ONNX model (all inputs including initializers).
        graph_input_names = [inp.name for inp in model.graph.input]

        return TVMRelaxBackendRep(
            tvm_model, params, func_param_names, graph_input_names
        )

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device == "CPU"


# Register the backend test suite.
backend_test = onnx.backend.test.BackendTest(TVMRelaxBackend, __name__)

# Include node tests for all operators supported by the Relax ONNX frontend.
# The runner appends _cpu/_cuda to test names, so we match with device suffix.
# 116 operators have corresponding ONNX BackendTest node tests.
_INCLUDE_OPS = [
    "abs", "acos", "acosh", "add", "and", "argmax", "argmin", "asin",
    "asinh", "atan", "atanh", "attention", "averagepool", "bitshift",
    "cast", "ceil", "clip", "compress", "concat", "constant",
    "constantofshape", "conv", "convtranspose", "cos", "cosh", "cumsum",
    "depthtospace", "dequantizelinear", "div", "dynamicquantizelinear",
    "einsum", "elu", "equal", "erf", "exp", "expand", "eyelike",
    "flatten", "floor", "gather", "gathernd", "gelu", "gemm",
    "globalaveragepool", "globalmaxpool", "greater", "gridsample",
    "hardmax", "hardsigmoid", "hardswish", "identity", "isinf", "isnan",
    "leakyrelu", "less", "log", "logsoftmax", "lppool", "lrn", "matmul",
    "matmulinteger", "max", "maxpool", "maxunpool", "mean", "min", "mish",
    "mod", "mul", "neg", "nonmaxsuppression", "nonzero", "not", "onehot",
    "optional", "or", "pow", "prelu", "quantizelinear", "range",
    "reciprocal", "relu", "reshape", "resize", "roialign", "round",
    "scatter", "scatternd", "selu", "shape", "shrink", "sigmoid", "sign",
    "sin", "sinh", "size", "slice", "softmax", "softplus", "softsign",
    "spacetodepth", "split", "sqrt", "squeeze", "sub", "sum", "tan",
    "tanh", "thresholdedrelu", "tile", "transpose", "unique", "unsqueeze",
    "upsample", "where", "xor",
]

for _op in _INCLUDE_OPS:
    backend_test.include(rf"test_{_op}.*(?:_cpu|_cuda)$")

# Known failures — xfail by category.
# Use (?:_.*)? to optionally match variant suffixes before _cpu/_cuda,
# avoiding greedy .* that would consume the device suffix.
# Trig function precision issues.
backend_test.xfail(r"test_asin(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_asinh(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_atan(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_atanh(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_mish(?:_.*)?(?:_cpu|_cuda)$")
# Output format mismatches.
backend_test.xfail(r"test_shape(?:_.*)?(?:_cpu|_cuda)$")
# Dynamic split not supported.
backend_test.xfail(r"test_split_variable_parts(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_split_zero_size_splits(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_split_to_sequence(?:_.*)?(?:_cpu|_cuda)$")
# All cast/castlike tests (exotic dtypes).
backend_test.xfail(r"test_cast(?:_e8m0)?_.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_castlike.+(?:_cpu|_cuda)$")
# Quantize/Dequantize edge cases.
backend_test.xfail(r"test_dequantizelinear_.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_quantizelinear_.+(?:_cpu|_cuda)$")
# Attention (complex op).
backend_test.xfail(r"test_attention.+(?:_cpu|_cuda)$")
# Resize (many interpolation edge cases).
backend_test.xfail(r"test_resize.+(?:_cpu|_cuda)$")
# Reshape edge cases.
backend_test.xfail(r"test_reshape_.+(?:_cpu|_cuda)$")
# cumsum edge cases.
backend_test.xfail(r"test_cumsum.+(?:_cpu|_cuda)$")
# Constant/ConstantOfShape edge cases.
backend_test.xfail(r"test_constant_pad(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_constantofshape(?:_.*)?(?:_cpu|_cuda)$")
# ConvInteger / ConvTranspose edge cases.
backend_test.xfail(r"test_convinteger.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_convtranspose_dilations(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_convtranspose_output_shape(?:_.*)?(?:_cpu|_cuda)$")
# Pow with mixed types.
backend_test.xfail(r"test_pow_types.+(?:_cpu|_cuda)$")
# Expanded versions of ops.
backend_test.xfail(r"test_elu_.+expanded.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_hardsigmoid_.+expanded.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_leakyrelu.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_relu_expanded.+(?:_cpu|_cuda)$")
# Ops that fail on base tests.
backend_test.xfail(r"test_selu(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_thresholdedrelu(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_shrink(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_softplus(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_softsign(?:_.*)?(?:_cpu|_cuda)$")
# String comparison.
backend_test.xfail(r"test_equal_string.+(?:_cpu|_cuda)$")
# Various individual failures.
backend_test.xfail(r"test_expand_.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_eyelike.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_gather_elements_negative.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_gather_negative.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_gelu.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_gridsample_volumetric.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_identity_opt(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_identity_sequence(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_isinf_negative(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_isinf_positive(?:_.*)?(?:_cpu|_cuda)$")
backend_test.xfail(r"test_lppool.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_log_softmax.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_maxpool_with_argmax.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_maxunpool.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_nonmaxsuppression.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_onehot.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_optional.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_prelu.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_range_.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_roialign_mode_max.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_scatter_elements_with_reduction.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_scatter_elements_with_duplicate.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_softmax_functional.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_softmax_lastdim.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_upsample.+(?:_cpu|_cuda)$")
backend_test.xfail(r"test_squeezenet.+(?:_cpu|_cuda)$")

globals().update(backend_test.test_cases)
