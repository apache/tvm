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
"""Quantized Neural Network (QNN) Operators"""
import tvm
from tvm import te, tir, topi

SQNN_FP32 = 0
SQNN_INT8 = 1
SQNN_UINT8 = 2
SQNN_INT32 = 3

SQNN_DTYPE_TO_CODE = {
    "float32": SQNN_FP32,
    "int8": SQNN_INT8,
    "uint8": SQNN_UINT8,
    "int32": SQNN_INT32,
}

SQNN_CODE_TO_DTYPE = {v: k for k, v in SQNN_DTYPE_TO_CODE.items()}


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def simulated_quantize(data, out_dtype, output_scale=None, output_zero_point=None, axis=-1):
    """Simulated QNN quantize operator that mimics QNN outputs in floating point. The benefit
    of this operator over true QNN quantize is that this operator allows dynamic datatype
    selection and can operate on both per-channel and scalar scales and zero points while
    QNN quantize requires both of these to be fixed at compile time.

    Parameters
    ----------
    data: tvm.te.Tensor
        An N-D input tensor to the operator.

    out_dtype: tvm.te.Tensor
        A 1-D variable that indicates which datatype to simulate quantization with. Use
        SQNN_DTYPE_TO_CODE to convert a dtype string into the corresponding variable
        value.

    output_scale: tvm.te.Tensor, optional
        A 1-D tensor representing the scale to use when quantizing to integer datatypes.
        When it contains more than a single value, N must match the number of channels in data.

    output_zero_point: tvm.te.Tensor, optional
        A 1-D tensor representing the zero point to use when quantizing to integer datatypes.
        When it contains more than a single value, N must match the number of channels in data.

    axis: int, optional
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    """

    # Since all simulated outputs are in float32, we can just return the input tensor for fp32.
    def _compute_fp32(value, *indices):
        return value[indices]

    # Simulate quantization for arbitrary integer datatypes. The computation for all datatypes is:
    # Q_output = clip((round(input_tensor/output_scale) + output_zero_point),
    #                 out_dtype::min,
    #                 out_dtype::max)
    def _compute_intn(dtype, value, *indices):
        assert output_scale is not None and output_zero_point is not None
        const_min = tvm.tir.min_value(dtype)
        const_max = tvm.tir.max_value(dtype)
        # Use indexmod to handle both scalar and per-channel QNN parameters.
        scale_idx = tir.indexmod(indices[axis], topi.shape(output_scale)[0])
        zp_idx = tir.indexmod(indices[axis], topi.shape(output_zero_point)[0])
        return te.max(
            te.min(
                te.round(value[indices] / output_scale[scale_idx]) + output_zero_point[zp_idx],
                const_max,
            ),
            const_min,
        )

    # Use an if chain to dynamically return the proper quantization based on the input datatype.
    # This allows the op to compile once but apply different quantization approaches
    # using a variable datatype input.
    def _dispatch_sim_qnn(value):
        fp32_value = te.compute(data.shape, lambda *indices: _compute_fp32(value, *indices))
        int8_value = te.compute(
            data.shape,
            lambda *indices: tir.if_then_else(
                out_dtype[0] == SQNN_DTYPE_TO_CODE['int8'],
                _compute_intn('int8', value, *indices),
                fp32_value[indices],
            ),
        )
        uint8_value = te.compute(
            data.shape,
            lambda *indices: tir.if_then_else(
                out_dtype[0] == SQNN_DTYPE_TO_CODE['uint8'],
                _compute_intn('uint8', value, *indices),
                int8_value[indices],
            ),
        )
        int32_value = te.compute(
            data.shape,
            lambda *indices: tir.if_then_else(
                out_dtype[0] == SQNN_DTYPE_TO_CODE['int32'],
                _compute_intn('int32', value, *indices),
                uint8_value[indices],
            ),
        )

        return int32_value

    return te.compute(data.shape, lambda *indices: _dispatch_sim_qnn(data)[indices])
