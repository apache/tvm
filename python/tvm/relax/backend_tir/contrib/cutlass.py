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
# pylint: disable=invalid-name,comparison-with-callable,unused-variable,missing-function-docstring
"""codegen for cutlass"""
import operator
from functools import reduce
from typing import List, Dict, Any

from tvm.contrib.cutlass.build import _get_cutlass_path, _get_cutlass_compile_options
from tvm.contrib.nvcc import get_target_compute_version
from tvm.contrib.cutlass.library import LayoutType, ConvKind
from tvm.contrib.cutlass.gen_tensor_op import instantiate_template
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler
from tvm.contrib.cutlass.gen_conv2d import CutlassConv2DProfiler
from ..pattern import (
    MatchResult,
    matmul_rrr_fp16,
    bias_row_2d_fp16,
    bias_row_1d_fp16,
    batch_bias_row_2d_fp16,
    batch_bias_row_1d_fp16,
    relu_fp16,
    erf_3d_fp32,
    batch_matmul_rrr_2d_fp16,
    batch_matmul_rrr_3d_fp16,
    conv2d_nhwc_fp16,
    padding_2d_nhwc_fp16,
    copy_4d_fp16,
    bias_add_nhwc_2d_fp16,
    bias_add_nhwc_1d_fp16,
    elem_add_4d_fp16,
    elem_mul_3d_fp16,
    scalar_add_3d_fp16,
    scalar_mul_3d_fp16,
    cast_3d_fp16,
    cast_3d_fp32,
)

#### helper functions ####
# list representing the anchor ops
# in the future more layouts/dtypes can be supported
MATMUL_LIST = [matmul_rrr_fp16]
MATMUL_BIAS_LIST = [bias_row_2d_fp16, bias_row_1d_fp16]
BATCH_MATMUL_LIST = [batch_matmul_rrr_2d_fp16, batch_matmul_rrr_3d_fp16]
BATCH_MATMUL_BIAS_LIST = [batch_bias_row_2d_fp16, batch_bias_row_1d_fp16]
CONV2D_LIST = [conv2d_nhwc_fp16]

# attributes for anchor ops used in code generation
OP_PATTERN_ATTR_LIST = {
    matmul_rrr_fp16: {
        "arg0_dtype": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float16",
    },
    batch_matmul_rrr_2d_fp16: {
        "arg0_dtype": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float16",
    },
    batch_matmul_rrr_3d_fp16: {
        "arg0_dtype": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float16",
    },
    conv2d_nhwc_fp16: {
        "arg0_dtype": "float16",
        "arg1_dtype": "float16",
        "ret_dtype": "float16",
        # in the future we can add layout here
    },
}


def _get_cutlass_code(attr):
    pattern = attr["op_type"]
    if pattern.startswith("cutlass.matmul"):
        return cutlass_codegen_gemm(attr)
    elif pattern.startswith("cutlass.conv2d"):
        return cutlass_codegen_conv2d(attr)
    else:
        raise ValueError("op not supported")


def _final_code(code, headers, func_args):
    res = ""
    res += "#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>\n"
    res += "#include <tvm/runtime/c_runtime_api.h>\n"
    res += "#include <tvm/runtime/packed_func.h>\n"
    res += "#include <dlpack/dlpack.h>\n"
    res += "#include <cuda_fp16.h>\n"
    res += "#include <cutlass/cutlass.h>\n"
    res += "#include <cutlass/coord.h>\n"
    res += "#include <cutlass/tensor_ref.h>\n"
    res += "#include <cutlass/util/host_tensor.h>\n"

    for header in headers:
        res += "#include <" + header + ">\n"
    res += "namespace {\n"
    res += "using namespace tvm;\n"
    res += "using namespace tvm::runtime;\n"
    res += "void _cutlass_kernel("
    for arg in func_args:
        res += "NDArray " + arg + ", "
    res += "NDArray out0) {"
    res += code
    res += "}\n"
    res += "}  // namespace\n"
    res += "TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _cutlass_kernel);\n"
    return res


#### cutlass patterns ####
def matmul_bias_relu(match_results, attr, get_code=True):
    if len(match_results) < 3:
        return None
    attr = matmul_bias(match_results[:2], attr, get_code=False)
    if attr is None or match_results[2].pattern != relu_fp16:
        return None
    m_bias, n_bias = match_results[1].symbol_values
    m_relu, n_relu = match_results[2].symbol_values
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    A_relu, B_relu = match_results[2].matched_buffers
    if m_bias == m_relu and n_bias == n_relu and C_bias == A_relu:
        attr["op_type"] = "cutlass.matmul_bias_relu"
        return [_get_cutlass_code(attr=attr), 3, attr["args"]] if get_code else attr
    return None


def matmul_bias(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    attr = matmul(match_results[:1], attr, get_code=False)
    if attr is None or match_results[1].pattern not in MATMUL_BIAS_LIST:
        return None
    m_matmul, n_matmul, k_matmul = match_results[0].symbol_values
    m_bias, n_bias = match_results[1].symbol_values
    A_matmul, B_matmul, C_matmul = match_results[0].matched_buffers
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    if m_matmul == m_bias and n_matmul == n_bias and C_matmul == A_bias:
        attr["op_type"] = "cutlass.matmul_bias"
        attr["bias_arg_idx"] = 2
        attr["args"].append(B_bias)
        return [_get_cutlass_code(attr=attr), 2, attr["args"]] if get_code else attr
    return None


def matmul(match_results, attr, get_code=True):
    if len(match_results) < 1:
        return None
    if match_results[0].pattern in MATMUL_LIST:
        # matmul
        attr["op_type"] = "cutlass.matmul"
        return [_get_cutlass_code(attr=attr), 1, attr["args"]] if get_code else attr
    return None


def batch_matmul_bias_gelu(match_results, attr, get_code=True):
    if len(match_results) < 9:
        return None
    attr = batch_matmul_bias(match_results[:2], attr, get_code=False)  # batch_matmul, batch_bias
    if (
        attr is None
        or match_results[2].pattern != scalar_mul_3d_fp16
        or match_results[3].pattern != cast_3d_fp32
        or match_results[4].pattern != erf_3d_fp32
        or match_results[5].pattern != cast_3d_fp16
        or match_results[6].pattern != scalar_mul_3d_fp16
        or match_results[7].pattern != scalar_add_3d_fp16
        or match_results[8].pattern != elem_mul_3d_fp16
    ):
        return None

    def shape_match_3d(shape1, shape2):
        if len(shape1) < 3 or len(shape2) < 3:
            return False
        return shape1[0] == shape2[0] and shape1[1] == shape2[1] and shape1[2] == shape2[2]

    for i in range(1, 8):
        if not shape_match_3d(match_results[i].symbol_values, match_results[i + 1].symbol_values):
            return None

    if not (
        match_results[1].matched_buffers[-1] == match_results[2].matched_buffers[0]
        and match_results[2].matched_buffers[-1] == match_results[3].matched_buffers[0]
        and match_results[3].matched_buffers[-1] == match_results[4].matched_buffers[0]
        and match_results[4].matched_buffers[-1] == match_results[5].matched_buffers[0]
        and match_results[5].matched_buffers[-1] == match_results[6].matched_buffers[0]
        and match_results[6].matched_buffers[-1] == match_results[7].matched_buffers[0]
        and match_results[1].matched_buffers[-1] == match_results[8].matched_buffers[0]
        and match_results[7].matched_buffers[-1] == match_results[8].matched_buffers[1]
    ):
        return None

    if (
        abs(float(match_results[2].symbol_values[-1] - 0.5**0.5)) > 1e-5
        or abs(float(match_results[6].symbol_values[-1] - 0.5)) > 1e-5
        or abs(float(match_results[7].symbol_values[-1] - 0.5)) > 1e-5
    ):
        return None

    attr["op_type"] = "cutlass.matmul_bias_gelu"
    return [_get_cutlass_code(attr=attr), 9, attr["args"]] if get_code else attr


def batch_matmul_bias_residual_mul(match_results, attr, get_code=True):
    if len(match_results) < 3:
        return None
    attr = batch_matmul_bias(match_results[:2], attr, get_code=False)  # batch_matmul, batch_bias
    if attr is None or match_results[2].pattern != elem_mul_3d_fp16:
        return None
    (
        b_bias,
        m_bias,
        n_bias,
    ) = match_results[1].symbol_values
    (
        b_mul,
        m_mul,
        n_mul,
    ) = match_results[2].symbol_values
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    A_mul, B_mul, C_mul = match_results[2].matched_buffers
    if b_bias == b_mul and m_bias == m_mul and n_bias == n_mul and C_bias == A_mul:
        attr["op_type"] = "cutlass.matmul_bias_residual_multiply"
        attr["residual_arg_idx"] = 3
        return [_get_cutlass_code(attr=attr), 3, attr["args"]] if get_code else attr
    return None


def batch_matmul_bias(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    attr = batch_matmul(match_results[:1], attr, get_code=False)
    if attr is None or match_results[1].pattern not in BATCH_MATMUL_BIAS_LIST:
        return None
    (
        b_matmul,
        m_matmul,
        n_matmul,
        k_matmul,
    ) = match_results[0].symbol_values
    (
        b_bias,
        m_bias,
        n_bias,
    ) = match_results[1].symbol_values
    A_matmul, B_matmul, C_matmul = match_results[0].matched_buffers
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    if b_matmul == b_bias and m_matmul == m_bias and n_matmul == n_bias and C_matmul == A_bias:
        attr["op_type"] = "cutlass.matmul_bias"
        attr["bias_arg_idx"] = 2
        attr["args"].append(B_bias)
        return [_get_cutlass_code(attr=attr), 2, attr["args"]] if get_code else attr
    return None


def batch_matmul(match_results, attr, get_code=True):
    if len(match_results) < 1:
        return None
    if match_results[0].pattern in BATCH_MATMUL_LIST:
        attr["op_type"] = "cutlass.matmul"
        return [_get_cutlass_code(attr=attr), 1, attr["args"]] if get_code else attr
    return None


def conv2d_bias_residual_add(match_results, attr, get_code=True):
    if len(match_results) < 4:
        return None
    attr = conv2d_bias(match_results[:3], attr, get_code=False)
    if attr is None or match_results[3].pattern != elem_add_4d_fp16:
        return None
    N_bias, H_bias, W_bias, C_bias = match_results[2].symbol_values
    in1_bias, in2_bias, out_bias = match_results[2].matched_buffers
    N_add, H_add, W_add, C_add = match_results[3].symbol_values
    in1_add, in2_add, out_add = match_results[3].matched_buffers
    if (
        N_bias == N_add
        and H_bias == H_add
        and W_bias == W_add
        and C_bias == C_add
        and out_bias in [in1_add, in2_add]
    ):
        attr["op_type"] = "cutlass.conv2d_bias_residual_add"
        attr["residual_arg_idx"] = 3
        attr["args"].append(in2_add if out_bias == in1_add else in1_add)
        return [_get_cutlass_code(attr=attr), 4, attr["args"]] if get_code else attr
    return None


def conv2d_bias(match_results, attr, get_code=True):
    if len(match_results) < 3:
        return None
    attr = conv2d(match_results[:2], attr, get_code=False)
    if attr is None or (
        match_results[2].pattern not in [bias_add_nhwc_2d_fp16, bias_add_nhwc_1d_fp16]
    ):
        return None
    (N_conv, pH_conv, pW_conv, H_conv, W_conv, C_conv, O_conv,) = match_results[
        1
    ].symbol_values[:7]
    A_pad_conv, B_conv, out_conv = match_results[1].matched_buffers
    N_bias, H_bias, W_bias, C_bias = match_results[2].symbol_values
    A_bias, B_bias, out_bias = match_results[2].matched_buffers
    if (
        N_bias == N_conv
        and H_bias == H_conv
        and W_bias == W_conv
        and C_bias == O_conv
        and out_conv == A_bias
    ):
        attr["op_type"] = "cutlass.conv2d_bias"
        attr["bias_arg_idx"] = 2
        attr["args"].append(B_bias)
        return [_get_cutlass_code(attr=attr), 3, attr["args"]] if get_code else attr
    return None


def conv2d(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    if (
        match_results[0].pattern in [padding_2d_nhwc_fp16, copy_4d_fp16]
        and match_results[1].pattern == conv2d_nhwc_fp16
    ):
        if match_results[0].pattern == padding_2d_nhwc_fp16:
            (
                N_pad,
                H_pad,
                W_pad,
                C_pad,
                pH_pad,
                pW_pad,
                lH_pad,
                lW_pad,
                rH_pad,
                rW_pad,
            ) = match_results[0].symbol_values
        else:
            (
                N_pad,
                H_pad,
                W_pad,
                C_pad,
            ) = match_results[0].symbol_values
            pH_pad = rH_pad = H_pad
            pW_pad = rW_pad = W_pad
            lH_pad = lW_pad = 0
        (
            N_conv,
            pH_conv,
            pW_conv,
            H_conv,
            W_conv,
            C_conv,
            O_conv,
            KH_conv,
            KW_conv,
            stride_h_conv,
            stride_w_conv,
            dilation_h_conv,
            dilation_w_conv,
        ) = match_results[1].symbol_values
        A, A_pad = match_results[0].matched_buffers
        A_pad_conv, B_conv, out_conv = match_results[1].matched_buffers
        if (
            N_pad == N_conv
            and pH_pad == pH_conv
            and pW_pad == pW_conv
            and C_pad == C_conv
            and A_pad == A_pad_conv
        ):
            if (
                lH_pad == pH_pad - rH_pad
                and lW_pad == pW_pad - rW_pad
                and lH_pad + H_pad == rH_pad
                and lW_pad + W_pad == rW_pad
            ):
                padding = (lH_pad, lW_pad)
                strides = (stride_h_conv, stride_w_conv)
                dilation = (dilation_h_conv, dilation_w_conv)
                attr["padding"] = padding
                attr["strides"] = strides
                attr["dilation"] = dilation
                attr["op_type"] = "cutlass.conv2d"
                return [_get_cutlass_code(attr=attr), 2, attr["args"]] if get_code else attr
    return None


### cutlass codegen functions ###
def compile_options(target, threads=-1, use_fast_math=False):
    compute_version = int("".join(get_target_compute_version(target).split(".")))
    kwargs = _get_cutlass_compile_options(compute_version, threads, use_fast_math)
    kwargs["options"].remove("-c")
    return kwargs


def cutlass_fcodegen(sm=80, bin_dir="./bin"):
    gemm_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), bin_dir)
    conv2d_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), bin_dir)

    def cutlass_codegen_with_match_results(match_results: List[MatchResult]):
        """generate cutlass code with match results"""
        nonlocal gemm_profiler
        nonlocal conv2d_profiler

        assert len(match_results) > 0

        # add shape into attr
        if match_results[0].pattern in MATMUL_LIST:
            A_matmul, B_matmul, C_matmul = match_results[0].matched_buffers
            attr: Dict[Any, Any] = OP_PATTERN_ATTR_LIST[match_results[0].pattern]
            attr["args"] = [A_matmul, B_matmul]
            attr["arg0_shape"] = A_matmul.shape
            attr["arg1_shape"] = B_matmul.shape
            attr["ret_shape"] = C_matmul.shape
            attr["lhs_arg_idx"] = 0
            attr["rhs_arg_idx"] = 1
        elif match_results[0].pattern in BATCH_MATMUL_LIST:
            A_matmul, B_matmul, C_matmul = match_results[0].matched_buffers
            attr = OP_PATTERN_ATTR_LIST[match_results[0].pattern]
            attr["args"] = [A_matmul, B_matmul]
            attr["arg0_shape"] = A_matmul.shape
            attr["arg1_shape"] = B_matmul.shape
            attr["ret_shape"] = C_matmul.shape
            attr["lhs_arg_idx"] = 0
            attr["rhs_arg_idx"] = 1
        elif len(match_results) >= 1 and match_results[1].pattern in CONV2D_LIST:
            A_input = match_results[0].matched_buffers[0]
            A_conv2d, B_conv2d, C_conv2d = match_results[1].matched_buffers
            attr = OP_PATTERN_ATTR_LIST[match_results[1].pattern]
            attr["args"] = [A_input, B_conv2d]
            attr["arg0_shape"] = A_input.shape
            attr["arg1_shape"] = B_conv2d.shape
            attr["ret_shape"] = C_conv2d.shape
            attr["lhs_arg_idx"] = 0
            attr["rhs_arg_idx"] = 1
        else:
            return ["", 0]

        # add profiler into attr
        attr["gemm_profiler"] = gemm_profiler
        attr["conv2d_profiler"] = conv2d_profiler

        cutlass_patterns = [
            # 9
            batch_matmul_bias_gelu,
            # 4
            conv2d_bias_residual_add,
            # 3
            batch_matmul_bias_residual_mul,
            matmul_bias_relu,
            conv2d_bias,
            # 2
            matmul_bias,
            batch_matmul_bias,
            conv2d,
            # 1
            matmul,
            batch_matmul,
        ]
        for pattern in cutlass_patterns:
            res = pattern(match_results, attr)
            if res is not None:
                return res

        return ["", 0]

    return cutlass_codegen_with_match_results


def cutlass_codegen_gemm(attrs):
    """cutlass codegen for gemm"""
    gemm_profiler = attrs["gemm_profiler"]
    op_type = attrs["op_type"]
    lhs_shape = attrs["arg0_shape"]
    rhs_shape = attrs["arg1_shape"]
    MM = lhs_shape[-2]
    KK = lhs_shape[-1]
    if "transposed" in op_type:
        NN = rhs_shape[-2]
        ldb = "K"
        layout_b = LayoutType.ColumnMajor
    else:
        NN = rhs_shape[-1]
        ldb = "N"
        layout_b = LayoutType.RowMajor

    lhs_batches = reduce(operator.mul, lhs_shape[:-2], 1)
    rhs_batches = reduce(operator.mul, rhs_shape[:-2], 1)
    if lhs_batches == 1 and rhs_batches == 1:
        # Regular matmul
        is_batched = False
        batch_attrs = {}
    else:
        is_batched = True
        batch_attrs = {
            # If both lhs_batches and rhs_batches are greater than 1,
            # they must be equal. This is checked by is_shape_valid_for_cutlass_matmul.
            "batch": lhs_batches if rhs_batches == 1 else rhs_batches,
            "batch_stride_A": 0 if lhs_batches == 1 else MM * KK,
            "batch_stride_B": 0 if rhs_batches == 1 else KK * NN,
            "batch_stride_C": MM * NN,
        }
    op_name, op_def, _ = gemm_profiler.profile(
        op_type,
        MM,
        NN,
        KK,
        attrs["ret_dtype"],
        attrs["arg0_dtype"],
        attrs["arg1_dtype"],
        False,
        batched=is_batched,
        find_first_valid=False,
        use_multiprocessing=True,
        layout_b=layout_b,
    )
    attrs["cutlass_op_name"] = op_name
    attrs["cutlass_op_def"] = op_def
    attrs["lda"] = "K"
    attrs["ldb"] = ldb
    attrs["ldc"] = "N"
    attrs.update(batch_attrs)
    del attrs["gemm_profiler"]
    del attrs["conv2d_profiler"]

    nargs = 2
    if "bias_arg_idx" in attrs:
        nargs += 1
    if "residual_arg_idx" in attrs:
        nargs += 1
    func_args = ["inp" + str(i) for i in range(nargs)]

    # A temporary solution to handle batch matmul residual cases
    # TODO(@bohan): remove this after initialize_template supports bmm residual
    if op_type in [
        "cutlass.matmul_bias_residual_multiply",
    ]:

        def _convert_dtype_str(dtype):
            if isinstance(dtype, list):
                arr = []
                for t in dtype:
                    arr.append(_convert_dtype_str(t))
                return arr
            elif isinstance(dtype, str):
                if dtype == "float16":
                    return "cutlass::half_t"
                elif dtype == "float32":
                    return "float"
            raise ValueError("dtype not supported")

        typea, typeb, typec = _convert_dtype_str(
            [attrs["arg0_dtype"], attrs["arg1_dtype"], attrs["ret_dtype"]]
        )

        text = f"""
#define CUTLASS_ENABLE_CUBLAS 1
#define CUTLASS_NAMESPACE cutlass
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
#define NDEBUG
#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
namespace {{
using namespace tvm;
using namespace tvm::runtime;
void _BHGEMM(NDArray A, NDArray B, NDArray Bias, NDArray D, NDArray C) {{
    // A: [Batch, M, K], B: [1, K, N]/[K, N], Bias: [1, N]/[N], D: [Batch, M, N], C: [Batch, M, N]
    CHECK_EQ(A->ndim, 3);
    int bdim = B->ndim;
    int bias_dim = Bias->ndim;
    CHECK_EQ(C->ndim, 3);
    CHECK_EQ(A->shape[2], B->shape[bdim - 2]);
    CHECK_EQ(Bias->shape[bias_dim - 1], B->shape[bdim - 1]);
    CHECK_EQ(D->ndim, 3);
    CHECK_EQ(D->shape[0], A->shape[0]);
    CHECK_EQ(D->shape[1], A->shape[1]);
    CHECK_EQ(D->shape[2], B->shape[bdim - 1]);
    CHECK_EQ(C->shape[0], A->shape[0]);
    CHECK_EQ(C->shape[1], A->shape[1]);
    CHECK_EQ(C->shape[2], B->shape[bdim - 1]);
    int64_t M = A->shape[0] * A->shape[1];
    int64_t N = B->shape[bdim - 1];
    int64_t K = A->shape[2];
    int64_t input_a_batch_stride = M * K;
    int64_t input_a_stride = K;
    int64_t input_a_offset = 0; // default to 0
    int64_t input_b_batch_stride = K * N;
    int64_t input_b_stride = N;
    int64_t input_b_offset = 0; // default to 0
    int64_t output_stride = N;
    int64_t output_offset = 0;
    int64_t a_size = 1;
    a_size *= A->shape[0];
    a_size *= A->shape[1];
    a_size *= A->shape[2];

    int64_t b_size = 1;
    b_size *= B->shape[bias_dim - 2];
    b_size *= B->shape[bias_dim - 1];

    int64_t c_size = 1;
    c_size *= C->shape[0];
    c_size *= C->shape[1];
    c_size *= C->shape[2];

    // Define the GEMM operation
    {op_def}
    using kernel = Operation_{op_name};
    using ElementComputeEpilogue = typename kernel::ElementAccumulator;
    typename kernel::Arguments arguments({{
        cutlass::gemm::GemmUniversalMode::kGemm, // GemmUniversalMode mode
        {{M, N, K}}, // GemmCoord problem_size
        1, // int batch_count
        {{ElementComputeEpilogue(1), ElementComputeEpilogue(1)}}, // typename EpilogueOutputOp::Params epilogue
        ({typea}*)(A->data) + input_a_offset, // void const * ptr_A
        ({typeb}*)(B->data) + input_b_offset, // void const * ptr_B
        ({typec}*)(D->data), // void const * ptr_C1
        ({typec}*)(C->data) + output_offset, // void * ptr_D
        ({typea}*)(Bias->data), // void * ptr_Vector
        nullptr, // void * ptr_Tensor
        input_a_batch_stride, // int64_t batch_stride_A
        input_b_batch_stride, // int64_t batch_stride_B
        0, // int64_t batch_stride_C1
        0, // int64_t batch_stride_D
        0, // int64_t batch_stride_Vector
        0, // int64_t batch_stride_Tensor
        input_a_stride, // typename LayoutA::Stride::Index lda
        input_b_stride, // typename LayoutB::Stride::Index ldb
        N, // typename LayoutC::Stride::Index ldc1
        output_stride, // typename LayoutC::Stride::Index ldd
        0, // typename LayoutC::Stride::Index ldr
        0, // typename LayoutC::Stride::Index ldt
    }});
    kernel gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.can_implement(arguments);
    CHECK(status == cutlass::Status::kSuccess);
    status = gemm_op.initialize(arguments, workspace.get());
    CHECK(status == cutlass::Status::kSuccess);
    status = gemm_op();
    CHECK(status == cutlass::Status::kSuccess);
    return;
}}
}}  // namespace
TVM_DLL_EXPORT_TYPED_FUNC({{global_symbol}}, _BHGEMM);
      """
        return text

    code = instantiate_template(op_type, attrs, func_args)
    return _final_code(code.code, code.headers, func_args)


def cutlass_codegen_conv2d(attrs):
    """cutlass codegen for conv2d"""
    # cutlass backend only supports nhwc for now
    conv2d_profiler = attrs["conv2d_profiler"]
    op_type = attrs["op_type"]
    conv_kind = ConvKind.Fprop
    op_name, op_def, _ = conv2d_profiler.profile(
        op_type=attrs["op_type"],
        d_shape=attrs["arg0_shape"],
        w_shape=attrs["arg1_shape"],
        padding=attrs["padding"],
        stride=attrs["strides"],
        dilation=attrs["dilation"],
        out_dtype=attrs["ret_dtype"],
        data_dtype=attrs["arg0_dtype"],
        weight_dtype=attrs["arg1_dtype"],
        use_3xtf32=False,
        conv_kind=conv_kind,
        split_k_slices=[1],
        profile_all_alignments=True,
        find_first_valid=False,
        use_multiprocessing=True,
    )
    attrs["cutlass_op_def"] = op_def
    attrs["cutlass_op_name"] = op_name
    del attrs["gemm_profiler"]
    del attrs["conv2d_profiler"]

    nargs = 2
    if "bias_arg_idx" in attrs:
        nargs += 1
    if "residual_arg_idx" in attrs:
        nargs += 1
    func_args = ["inp" + str(i) for i in range(nargs)]
    code = instantiate_template(op_type, attrs, func_args)
    return _final_code(code.code, code.headers, func_args)
