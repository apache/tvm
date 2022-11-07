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

from tvm import topi

from .. import strategy
from ...op.op import register_compute
from ...op.op import register_injective_schedule
from ...op.op import register_strategy, register_pattern, OpPattern


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
register_pattern("qnn.add", OpPattern.BROADCAST)

# qnn.concatenate
register_strategy("qnn.concatenate", strategy.qnn_concatenate_strategy)
register_pattern("qnn.concatenate", OpPattern.INJECTIVE)

# qnn.conv2d
register_strategy("qnn.conv2d", strategy.qnn_conv2d_strategy)
register_pattern("qnn.conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)

# qnn.dense
register_strategy("qnn.dense", strategy.qnn_dense_strategy)
register_pattern("qnn.dense", OpPattern.OUT_ELEMWISE_FUSABLE)

# qnn.batch_matmul
register_strategy("qnn.batch_matmul", strategy.qnn_batch_matmul_strategy)
register_pattern("qnn.batch_matmul", OpPattern.OUT_ELEMWISE_FUSABLE)
