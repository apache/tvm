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
# pylint: disable=invalid-name, unused-argument
"""Arm(R) Ethos(TM) -N NPU supported operators."""
from enum import Enum
import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_constant
from ... import qnn as _qnn
from .register import register_pattern_table
from . import _ethosn as support


class Available(Enum):
    UNAVAILABLE = 0
    SW_ONLY = 1
    SW_AND_HW = 2

    def __bool__(self):
        return self != Available.UNAVAILABLE


def ethosn_available():
    """Return whether Ethos-N software and hardware support is available"""
    if not tvm.get_global_func("relay.ethos-n.query", True):
        print("skip because Ethos-N module is not available")
        return Available.UNAVAILABLE
    hw = tvm.get_global_func("relay.ethos-n.query")()
    return Available.SW_AND_HW if hw else Available.SW_ONLY


@register_pattern_table("ethos-n")
def pattern_table():
    """Get the Ethos-N compiler pattern table."""

    def qnn_conv_pattern():
        pattern = is_op("nn.pad")(wildcard()) | wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("nn.bias_add")(pattern, is_constant())
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_fc_pattern():
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("nn.bias_add")(pattern, is_constant())
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_avg_pool2d_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def qnn_sigmoid_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("sigmoid")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def check_conv2d(extract):
        """Check if a conv2d is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return support.conv2d(extract)

    def check_fc(extract):
        """Check if a fully connected is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return support.fc(extract)

    def check_avg_pool2d(extract):
        """Check if a avg pool2d is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return support.avg_pool2d(extract)

    def check_sigmoid(extract):
        """Check if a sigmoid is supported by Ethos-N."""
        if not ethosn_available():
            return False

        if extract.attrs.out_dtype != "uint8":
            return False

        return support.sigmoid(extract)

    return [
        ("ethos-n.qnn_conv2d", qnn_conv_pattern(), check_conv2d),
        ("ethos-n.qnn_avg_pool2d", qnn_avg_pool2d_pattern(), check_avg_pool2d),
        ("ethos-n.qnn_sigmoid", qnn_sigmoid_pattern(), check_sigmoid),
        ("ethos-n.qnn_fc", qnn_fc_pattern(), check_fc),
    ]


def _is_ethosn_composite(node):
    if isinstance(node, tvm.relay.expr.Call) and isinstance(node.op, tvm.relay.Function):
        if "Composite" in node.op.attrs:
            comp_name = node.op.attrs["Composite"]
            return comp_name.split(".")[0] == "ethos-n"

    return False


@tvm.ir.register_op_attr("nn.max_pool2d", "target.ethos-n")
def max_pool2d(attrs, args):
    """Check if a max pool2d is supported by Ethos-N."""
    if not ethosn_available():
        return False

    pool = tvm.relay.nn.max_pool2d(*args, **attrs)
    return support.max_pool2d(pool)


@tvm.ir.register_op_attr("reshape", "target.ethos-n")
def reshape(attrs, args):
    """Check if a reshape is supported by Ethos-N."""
    if not ethosn_available():
        return False

    if not _is_ethosn_composite(args[0]):
        return False

    rs = tvm.relay.op.reshape(*args, attrs["newshape"])
    return support.reshape(rs)


@tvm.ir.register_op_attr("qnn.add", "target.ethos-n")
def qnn_add(attrs, args):
    """Check if an addition is supported by Ethos-N."""
    if not ethosn_available():
        return False

    add = _qnn.op.add(*args)
    return support.addition(add)


@tvm.ir.register_op_attr("qnn.concatenate", "target.ethos-n")
def qnn_concatenate(attrs, args):
    """Check if a concatenate is supported by Ethos-N."""
    if not ethosn_available():
        return False

    conc = _qnn.op.concatenate(*args, **attrs)
    if not support.concatenate(conc):
        return False

    # Support library has some unenforced restrictions on qnn params
    min_range = 1e9
    max_range = -1e9
    qnn_params = []
    for i in range(len(args[1].fields)):
        scale = args[1].fields[i].data.asnumpy()
        zero_point = args[2].fields[i].data.asnumpy()
        min_range = min(-1 * zero_point * scale, min_range)
        max_range = max((255 - zero_point) * scale, max_range)
        qnn_params.append((scale, zero_point))

    scale = (max_range - min_range) / 255
    zero_point = int(-min_range / scale)
    if (scale, zero_point) in qnn_params:
        return True

    return False


@tvm.ir.register_op_attr("split", "target.ethos-n")
def split(attrs, args):
    """Check if a split is supported by Ethos-N."""
    if not ethosn_available():
        return False

    if isinstance(attrs["indices_or_sections"], tvm.tir.IntImm):
        sp = tvm.relay.split(
            *args, indices_or_sections=attrs["indices_or_sections"].value, axis=attrs["axis"]
        )
    else:
        sp = tvm.relay.split(
            *args, indices_or_sections=attrs["indices_or_sections"], axis=attrs["axis"]
        )
    if not support.split(sp.astuple()):
        return False

    return True


@tvm.ir.register_op_attr("nn.depth_to_space", "target.ethos-n")
def depth_to_space(attrs, args):
    """Check if a depth_to_space is supported by Ethos-N."""
    if not ethosn_available():
        return False

    depth = tvm.relay.nn.depth_to_space(*args, **attrs)
    if not support.depth_to_space(depth):
        return False

    return True


@tvm.ir.register_op_attr("clip", "target.ethos-n")
def clip(attrs, args):
    """Check if a clip is supported by Ethos-N."""
    if not ethosn_available():
        return False

    c = tvm.relay.clip(*args, **attrs)
    if not support.relu(c):
        return False

    return True
