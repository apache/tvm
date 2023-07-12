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
# pylint: disable=invalid-name, unused-argument, len-as-condition
"""QNN operator feature registration"""

import numpy as np

from tvm import topi

from .. import strategy
from ...op.op import register_compute
from ...op.op import register_injective_schedule
from ...op.op import (
    OpPattern,
    register_alter_op_layout,
    register_legalize,
    register_pattern,
    register_strategy,
)


@register_compute("qnn.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, output_type):
    assert len(inputs) == 4
    return [
        topi.nn.simulated_quantize(
            inputs[0], inputs[1], inputs[2], inputs[3], axis=attrs.get_int("axis")
        )
    ]


register_injective_schedule("qnn.simulated_quantize")
register_pattern("qnn.simulated_quantize", OpPattern.ELEMWISE)


@register_compute("qnn.simulated_dequantize")
def simulated_dequantize_compute(attrs, inputs, output_type):
    assert len(inputs) == 4
    return [
        topi.nn.simulated_dequantize(
            inputs[0], inputs[1], inputs[2], inputs[3], axis=attrs.get_int("axis")
        )
    ]


register_injective_schedule("qnn.simulated_dequantize")
register_pattern("qnn.simulated_dequantize", OpPattern.ELEMWISE)

# qnn.quantize
register_strategy("qnn.quantize", strategy.qnn_quantize_strategy)
register_pattern("qnn.quantize", OpPattern.ELEMWISE)

# qnn.dequantize
register_strategy("qnn.dequantize", strategy.qnn_dequantize_strategy)
register_pattern("qnn.dequantize", OpPattern.ELEMWISE)

# qnn.requantize
register_strategy("qnn.requantize", strategy.qnn_requantize_strategy)
register_pattern("qnn.requantize", OpPattern.ELEMWISE)

# qnn.add
register_strategy("qnn.add", strategy.qnn_add_strategy)

# qnn.subtract
register_strategy("qnn.subtract", strategy.qnn_subtract_strategy)

# qnn.mul
register_strategy("qnn.mul", strategy.qnn_mul_strategy)

# qnn.tanh
register_strategy("qnn.tanh", strategy.qnn_tanh_strategy)
register_pattern("qnn.tanh", OpPattern.ELEMWISE)

# qnn.concatenate
register_strategy("qnn.concatenate", strategy.qnn_concatenate_strategy)
register_pattern("qnn.concatenate", OpPattern.INJECTIVE)

# qnn.conv2d
register_strategy("qnn.conv2d", strategy.qnn_conv2d_strategy)


@register_legalize("clip")
def legalize_clip(attrs, inputs, tinfos):
    """Removes clip operators with bounds matching the defaults for their dtype.

    This is already done after alter_op by TVM's simplification passes, but certain QNN operator
    implementations (like Cortex-M) need it to be done earlier in legalization.
    """

    if (
        hasattr(inputs[0], "op")
        and hasattr(inputs[0].op, "name")
        and inputs[0].op.name == "qnn.requantize"
    ):
        dtype_info = np.iinfo(tinfos[0].dtype)
        if dtype_info.min == attrs.a_min and dtype_info.max == attrs.a_max:
            return inputs[0]

    return None


@register_legalize("nn.bias_add")
def legalize_bias_add(attrs, inputs, tinfos):
    """Legalize a bias add operator.

    May be used to "fold in" unused channels from quantized convolution operators. This should
    be done before layout rewrites occur to minimize the amount of "extra" overhead operators
    like "cast" and "layout_transform".
    """
    return topi.nn.bias_add_legalize(attrs, inputs, tinfos)


@register_alter_op_layout("qnn.conv2d")
def alter_op_layout_qnn_conv2d(attrs, inputs, tinfos, out_type):
    """Alter the layout of a qnn conv2d op.

    May be used to alter the current QNN Conv2D op, but can also be used to alter previous ops to
    better match the current op. For example, Arm Cortex-M uses this to set the out_layout of
    previous ops to the input layout preferred by future layouts.
    """
    return topi.nn.qnn_conv2d_alter_layout(attrs, inputs, tinfos, out_type)


@register_alter_op_layout("add")
def alter_op_layout_add(attrs, inputs, tinfos, out_type):
    """Alter the layout of a add op.

    Useful for fusing the bias constant with an input zero point constant in a previous quantized
    op. Only used when previous op is a quantized op, which is why it lives in topi.nn.qnn.
    """
    return topi.nn.add_alter_layout(attrs, inputs, tinfos, out_type)


@register_alter_op_layout("qnn.requantize")
def alter_op_layout_qnn_requantize(attrs, inputs, tinfos, out_type):
    """Alter the layout of a requantization op."""
    return topi.nn.qnn_requantize_alter_layout(attrs, inputs, tinfos, out_type)


# qnn.dense
register_strategy("qnn.dense", strategy.qnn_dense_strategy)


@register_alter_op_layout("qnn.dense")
def alter_op_layout_qnn_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of qnn.dense"""
    return topi.nn.qnn_dense_alter_layout(attrs, inputs, tinfos, out_type)


# qnn.contrib_dense_pack
register_strategy("qnn.contrib_dense_pack", strategy.qnn_dense_pack_strategy)

# qnn.batch_matmul
register_strategy("qnn.batch_matmul", strategy.qnn_batch_matmul_strategy)
register_pattern("qnn.batch_matmul", OpPattern.OUT_ELEMWISE_FUSABLE)

# qnn.avg_pool2d
register_strategy("qnn.avg_pool2d", strategy.qnn_avg_pool2d_strategy)
