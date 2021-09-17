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
"""Helper module to build relay operations for testing"""

from pathlib import Path
import numpy as np
import math

import tvm
from tvm import relay
from tvm.relay.op.contrib import get_pattern_table
from tvm.relay import qnn
from tvm.relay.backend.contrib.ethosu.util import get_range_for_dtype_str


class TensorType:
    """A data structure to capture tensor parameters"""

    def __init__(self):
        self.shape = None
        self.dtype = None
        self.zp = None
        self.sc = None
        self.layout = None

    def get_dim_size(self, dim):
        for idx, char in enumerate(self.layout):
            if dim == char:
                return self.shape[idx]
        return None

    def get_dim_index(self, dim):
        for idx, char in enumerate(self.layout):
            if dim == char:
                return idx
        return None


class QnnConv2DParams:
    """A data structure to capture relay.qnn.op.conv2D parameters"""

    def __init__(self, dtype):
        self.ifm = TensorType()
        self.ofm = TensorType()
        self.kernel = TensorType()

        # default values
        self.ifm.dtype = dtype
        self.ifm.layout = "NHWC"
        ifm_min, ifm_max = get_range_for_dtype_str(self.ifm.dtype)
        self.ifm.zp = relay.const(np.random.randint(ifm_min, ifm_max), "int32")
        self.ifm.sc = relay.const(np.random.random() * 2, "float32")
        self.kernel.dtype = dtype
        self.kernel.layout = "HWIO"
        kernel_min, kernel_max = get_range_for_dtype_str(self.kernel.dtype)
        self.kernel.zp = relay.const(np.random.randint(kernel_min, kernel_max), "int32")
        self.kernel.sc = relay.const(np.random.random() * 2, "float32")
        self.ofm.layout = "NHWC"
        self.ofm.dtype = dtype
        ofm_min, ofm_max = get_range_for_dtype_str(self.ofm.dtype)
        self.ofm.zp = relay.const(np.random.randint(ofm_min, ofm_max), "int32")
        self.ofm.sc = relay.const(np.random.random() * 2, "float32")
        self.dilation = (1, 1)

        self.strides = None
        self.pad = None
        self.activation = "NONE"
        self.clip_min = 0
        self.clip_max = 0

    def update_output_qnn_params(
        self, input_dtype="uint8", kernel_dtype="uint8", output_dtype="uint8"
    ):
        _, dtype_max = get_range_for_dtype_str(input_dtype)
        input_max = self.ifm.sc.data.asnumpy() * (dtype_max - self.ifm.zp.data.asnumpy())
        input_min = -self.ifm.sc.data.asnumpy() * self.ifm.zp.data.asnumpy()
        _, dtype_max = get_range_for_dtype_str(kernel_dtype)
        kernel_max = np.max(
            self.kernel.sc.data.asnumpy() * (dtype_max - self.kernel.zp.data.asnumpy())
        )
        kernel_min = np.min(-self.kernel.sc.data.asnumpy() * self.kernel.zp.data.asnumpy())
        kernel_h = self.kernel.get_dim_size("H")
        kernel_w = self.kernel.get_dim_size("W")
        channels = self.kernel.get_dim_size("I")
        output_limits = [
            kernel_max * kernel_h * kernel_w * channels * input_max,
            kernel_min * kernel_h * kernel_w * channels * input_max,
            kernel_min * kernel_h * kernel_w * channels * input_min,
            kernel_max * kernel_h * kernel_w * channels * input_min,
        ]
        output_max = max(output_limits)
        output_min = min(output_limits)
        dtype_min, dtype_max = get_range_for_dtype_str(input_dtype)
        self.ofm.sc = relay.const((output_max - output_min) / (dtype_max - dtype_min), "float32")
        self.ofm.zp = relay.const(-int(output_min / self.ofm.sc.data.asnumpy()), "int32")


class PoolingParams:
    """A data structure to capture relay.op.max_pool2d /
    relay.op.avg_pool2d parameters
    """

    def __init__(self, dtype):
        self.type = None
        self.size = None
        self.strides = None
        self.pad = None
        self.layout = None
        self.ifm = TensorType()
        self.ofm = TensorType()

        # default values
        self.ifm.dtype = dtype
        self.ifm.layout = "NHWC"
        self.ifm.zp = relay.const(np.random.randint(0, 255), "int32")
        self.ifm.sc = relay.const(np.random.random() * 2, "float32")
        self.ofm.zp = relay.const(np.random.randint(0, 255), "int32")
        self.ofm.sc = relay.const(np.random.random() * 2, "float32")
        self.ofm.dtype = dtype
        self.dilation = (1, 1)


class AddParams:
    """A data structure to capture relay.qnn.op.add parameters"""

    def __init__(self, dtype):
        self.ifm0 = TensorType()
        self.ifm1 = TensorType()
        self.ofm = TensorType()

        # default values
        self.ifm0.dtype = dtype
        self.ifm0.zp = relay.const(np.random.randint(0, 255), "int32")
        self.ifm0.sc = relay.const(np.random.random() * 2, "float32")
        self.ifm1.dtype = dtype
        self.ifm1.zp = relay.const(np.random.randint(0, 255), "int32")
        self.ifm1.sc = relay.const(np.random.random() * 2, "float32")
        self.update_output_qnn_params()
        self.ofm.dtype = dtype

    def update_output_qnn_params(self):
        ti = np.iinfo(self.ifm0.dtype)
        dtype_min, dtype_max = int(ti.min), int(ti.max)
        input1_max = self.ifm0.sc.data.asnumpy() * (dtype_max - self.ifm0.zp.data.asnumpy())
        input1_min = (dtype_min - self.ifm0.sc.data.asnumpy()) * self.ifm0.zp.data.asnumpy()
        input2_max = self.ifm1.sc.data.asnumpy() * (dtype_max - self.ifm1.zp.data.asnumpy())
        input2_min = (dtype_min - self.ifm1.sc.data.asnumpy()) * self.ifm1.zp.data.asnumpy()
        output_max = input1_max + input2_max
        output_min = input1_min + input2_min
        self.ofm.sc = relay.const((output_max - output_min) / dtype_max, "float32")
        self.ofm.zp = relay.const(
            (dtype_min - int(output_min / self.ofm.sc.data.asnumpy())), "int32"
        )


def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding"""

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def create_qnn_conv2d(qnn_conv2d_params, ifm_expr):
    """Create a relay.Expr of relay.qnn.conv2D given the parameters"""
    v_params = list()
    params = {
        "kernel_size": [
            qnn_conv2d_params.kernel.get_dim_size("H"),
            qnn_conv2d_params.kernel.get_dim_size("W"),
        ],
        "strides": [qnn_conv2d_params.strides[0], qnn_conv2d_params.strides[1]],
        "dilation": [qnn_conv2d_params.dilation[0], qnn_conv2d_params.dilation[1]],
        "padding": [0, 0, 0, 0],
        "data_layout": qnn_conv2d_params.ifm.layout,
    }
    dilated_kernel_h = (
        qnn_conv2d_params.dilation[0] * (qnn_conv2d_params.kernel.get_dim_size("H") - 1) + 1
    )
    dilated_kernel_w = (
        qnn_conv2d_params.dilation[1] * (qnn_conv2d_params.kernel.get_dim_size("W") - 1) + 1
    )
    if qnn_conv2d_params.pad == "SAME":
        pad_top, pad_bottom = get_pad_value(
            qnn_conv2d_params.ifm.get_dim_size("H"), dilated_kernel_h, qnn_conv2d_params.strides[0]
        )
        pad_left, pad_right = get_pad_value(
            qnn_conv2d_params.ifm.get_dim_size("W"), dilated_kernel_w, qnn_conv2d_params.strides[1]
        )
        do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
        if do_pad:
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
    qnn_conv2d_params.pad = params["padding"]
    params["input_zero_point"] = qnn_conv2d_params.ifm.zp
    params["kernel_zero_point"] = qnn_conv2d_params.kernel.zp
    params["out_dtype"] = "int32"
    params["input_scale"] = qnn_conv2d_params.ifm.sc
    params["kernel_scale"] = qnn_conv2d_params.kernel.sc
    params["channels"] = int(qnn_conv2d_params.kernel.get_dim_size("O"))
    params["kernel_layout"] = qnn_conv2d_params.kernel.layout
    k_shape = qnn_conv2d_params.kernel.shape
    k_dtype = qnn_conv2d_params.kernel.dtype
    w = tvm.nd.array(
        np.random.randint(
            np.iinfo(k_dtype).min, high=np.iinfo(k_dtype).max, size=k_shape, dtype=k_dtype
        )
    )
    weight_expr = relay.const(w, k_dtype)
    v_params.append(w)
    qnn_conv2d_expr = qnn.op.conv2d(ifm_expr, weight_expr, **params)
    b = tvm.nd.array(
        np.random.randint(
            0, high=10, size=(qnn_conv2d_params.kernel.get_dim_size("O")), dtype="int32"
        )
    )
    v_params.append(b)
    bias_expr = relay.const(b, "int32")
    bias = relay.nn.bias_add(
        qnn_conv2d_expr, bias_expr, axis=qnn_conv2d_params.ifm.get_dim_index("C")
    )
    bias_scale = relay.const(
        qnn_conv2d_params.ifm.sc.data.asnumpy() * qnn_conv2d_params.kernel.sc.data.asnumpy(),
        "float32",
    )
    req_expr = relay.qnn.op.requantize(
        bias,
        bias_scale,  # input zero scale
        relay.const(0, "int32"),  # input zero point
        qnn_conv2d_params.ofm.sc,  # output zero scale
        qnn_conv2d_params.ofm.zp,  # output zero point
        out_dtype=qnn_conv2d_params.ofm.dtype,
    )
    if qnn_conv2d_params.activation != "NONE":
        assert qnn_conv2d_params.activation == "CLIP"
        clip_expr = relay.clip(req_expr, qnn_conv2d_params.clip_min, qnn_conv2d_params.clip_max)
        return clip_expr, v_params

    return req_expr, v_params


def create_pool2d(pooling_params, ifm_expr):
    """Create a relay pooling operation"""
    assert pooling_params.ifm.layout == "NHWC"
    params = {
        "pool_size": (pooling_params.size[0], pooling_params.size[1]),
        "strides": (pooling_params.strides[0], pooling_params.strides[1]),
        "padding": [0, 0],
        "layout": "NHWC",
    }
    if pooling_params.pad == "SAME":
        pad_top, pad_bottom = get_pad_value(
            pooling_params.ifm.shape[1], pooling_params.size[0], pooling_params.strides[0]
        )
        pad_left, pad_right = get_pad_value(
            pooling_params.ifm.shape[2], pooling_params.size[1], pooling_params.strides[1]
        )
        params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
    if pooling_params.type == "MAX":
        out = relay.op.nn.max_pool2d(ifm_expr, **params)
    else:
        assert pooling_params.type == "AVG"
        out = relay.op.cast(ifm_expr, dtype="int32")
        out = relay.op.nn.avg_pool2d(out, **params)
        out = relay.op.cast(out, dtype=pooling_params.ofm.dtype)
    return out


def create_qnn_add(ifm0_expr, ifm1_expr, add_params):
    add = relay.qnn.op.add(
        lhs=ifm0_expr,
        rhs=ifm1_expr,
        lhs_scale=add_params.ifm0.sc,
        lhs_zero_point=add_params.ifm0.zp,
        rhs_scale=add_params.ifm1.sc,
        rhs_zero_point=add_params.ifm1.zp,
        output_scale=add_params.ofm.sc,
        output_zero_point=add_params.ofm.zp,
    )
    return add
