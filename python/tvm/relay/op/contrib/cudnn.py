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
# pylint: disable=unused-argument
"""cuDNN Relay integration."""
from typing import Callable, List, Tuple

import tvm
import tvm.ir
from tvm import relay
from tvm import te
from tvm.relay import transform
from tvm.contrib import cudnn

from ...dataflow_pattern import is_op, wildcard
from .te_target import lower_composite, relay_to_runtime
from .register import register_pattern_table


tvm._ffi.register_func("relay.ext.cudnn", relay_to_runtime(tvm.target.cuda()))


def partition_for_cudnn(mod: tvm.IRModule) -> tvm.IRModule:
    """Partition the graph to offload for cuDNN.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.

    Returns
    -------
    tvm.IRModule
        The partitioned module.
    """

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("cudnn"),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("cudnn")
def pattern_table() -> List[Tuple[str, relay.Pattern, Callable[[relay.Call], bool]]]:
    """Get the cuDNN pattern table."""

    def softmax_pattern() -> relay.Pattern:
        """Create pattern for softmax."""
        return is_op("nn.softmax")(wildcard())

    def log_softmax_pattern() -> relay.Pattern:
        """Create pattern for log_softmax."""
        return is_op("nn.log_softmax")(wildcard())

    def conv2d_pattern() -> relay.Pattern:
        """Create pattern for conv2d."""
        return is_op("nn.conv2d")(wildcard(), wildcard())

    def conv2d_bias_act_pattern() -> relay.Pattern:
        """Create pattern for fused conv2d+bias+activation."""
        conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
        bias = is_op("nn.bias_add")(conv2d, wildcard())
        return bias.optional(is_op("nn.relu"))

    def check_softmax(matched: relay.Call) -> bool:
        """Check if softmax is supported by cuDNN."""
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        return True

    def check_log_softmax(matched: relay.Call) -> bool:
        """Check if log_softmax is supported by cuDNN."""
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        if len(matched.args[0].checked_type.shape) != 2:
            return False

        if matched.attrs["axis"] not in (1, -1):
            return False

        return True

    def check_conv2d(matched: relay.Call) -> bool:
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        if matched.attrs["data_layout"] != "NCHW" or matched.attrs["kernel_layout"] != "OIHW":
            return False

        padding = matched.attrs["padding"]
        if padding[0] != padding[2] or padding[1] != padding[3]:
            return False

        return True

    def check_conv2d_bias_act(matched: relay.Call) -> bool:
        return True

    return [
        ("cudnn.softmax", softmax_pattern(), check_softmax),
        ("cudnn.log_softmax", log_softmax_pattern(), check_log_softmax),
        ("cudnn.conv2d_bias_act", conv2d_bias_act_pattern(), check_conv2d_bias_act),
        ("cudnn.conv2d", conv2d_pattern(), check_conv2d),
    ]


@lower_composite("cudnn.softmax")
def _lower_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a softmax using cuDNN."""
    return cudnn.softmax(inputs[0], axis=op.attrs["axis"])


@lower_composite("cudnn.log_softmax")
def _lower_log_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a log_softmax using cuDNN."""
    return cudnn.log_softmax(inputs[0], axis=op.attrs["axis"])


@lower_composite("cudnn.conv2d_bias_act")
def _lower_conv2d_bias_act(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a fused conv2d+bias+activation using cuDNN."""
    conv_dtype = op.checked_type.dtype
    if op.op.name == "nn.relu":
        activation_mode = 1  # Relu
        conv2d = op.args[0].args[0]
    else:
        activation_mode = 5  # Identity
        conv2d = op.args[0]

    conv_mode = 1
    tensor_format = 0
    algo = 1
    pad = conv2d.attrs["padding"]
    strides = conv2d.attrs["strides"]
    dilation = conv2d.attrs["dilation"]
    groups = conv2d.attrs["groups"]

    oshape = cudnn.conv_output_shape(
        tensor_format,
        pad,
        strides,
        dilation,
        inputs[0].shape,
        inputs[1].shape,
        inputs[0].dtype,
        conv_dtype,
        groups,
    )

    return te.extern(
        oshape,
        inputs,
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.conv2d+bias+act.forward",
            conv_mode,
            tensor_format,
            algo,
            pad[0],
            pad[1],
            strides[0],
            strides[1],
            dilation[0],
            dilation[1],
            activation_mode,
            0,
            ins[0],
            ins[1],
            ins[2],
            outs[0],
            conv_dtype,
            groups,
        ),
        name="y",
    )


@lower_composite("cudnn.conv2d")
def _lower_conv2d(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a conv2d using cuDNN."""
    return cudnn.conv_forward(
        inputs[0],
        inputs[1],
        pad=op.attrs["padding"],
        stride=op.attrs["strides"],
        dilation=op.attrs["dilation"],
        conv_mode=1,
        tensor_format=0,
        algo=1,
        conv_dtype=op.checked_type.dtype,
        groups=op.attrs["groups"],
    )
