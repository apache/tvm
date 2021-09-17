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
# pylint: disable=line-too-long,unused-argument
"""Default behavior for ops in mixed_precision pass. Import this file to use."""
from typing import List

from tvm.relay.op import register_mixed_precision_conversion

# MIXED_PRECISION_ALWAYS ops should always be done in lower precision due to the speed and memory
# savings. MIXED_PRECISION_FOLLOW ops can be done in lower precision but don't have speedups to
# justify a cast. MIXED_PRECISION_NEVER colored ops should not be done in lower precision due to
# numerical reasons.
MIXED_PRECISION_ALWAYS = 0
MIXED_PRECISION_FOLLOW = 1
MIXED_PRECISION_NEVER = 2

# Default lists inspired from TF's classifications:
# github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
# They have a bias toward Nvidia Tensor Cores so modify lists per your hardware choice.
DEFAULT_ALWAYS_LIST = [
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
    "nn.batch_matmul",
]
DEFAULT_FOLLOW_LIST = [
    # These ops add new data or change shape
    "nn.pad",
    "nn.batch_flatten",
    "concatenate",
    "zeros",
    "split",
    "squeeze",
    "transpose",
    "expand_dims",
    "reshape",
    "dyn.reshape",
    "broadcast_to_like",
    "dyn.broadcast_to",
    "strided_slice",
    "dyn.strided_slice",
    "take",
    "argwhere",
    "where",
    "tile",
    "dyn.tile",
    "scatter",
    "full",
    "dyn.full",
    # Comparison
    "less",
    "greater",
    "less_equal",
    "greater_equal",
    # By definition copy and cast will depend on inputs for output.
    "copy",
    "cast",
    "cast_like",
    # Simple arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "nn.bias_add",
    "nn.batch_norm",
    "sqrt",
    "shape_of",
    # Simple activations
    "max",
    "min",
    "maximum",
    "minimum",
    "nn.relu",
    "nn.leaky_relu",
    "nn.prelu",
    "nn.dropout",
    # Complicated activations which saturate in a narrow range
    "sigmoid",
    "tanh",
    # Pooling operations
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    # "nn.global_max_pool1d", # does not exist yet
    "nn.global_max_pool2d",
    # "nn.global_max_pool3d", # does not exist yet
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
]
DEFAULT_NEVER_LIST = [
    # In general if |f(x)| >> |x| for expected inputs then put the op here.
    "exp",
    "power",
    "nn.cross_entropy",
    "nn.cross_entropy_with_logits",
    "nn.softmax",
    "nn.l2_normalize",
    # Error function doesn't seem to be able to be lowered into fp16 version in llvm.
    # Move to follow list when it does.
    "erf",
    # Do not allow arange arguments (begin/end) to be fp16. "end" can be a big fp32 number
    # not representable in fp16.
    "arange",
    # Ops that could involve a large summation are not allowed in fp16.
    "nn.global_avg_pool2d",
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
    "sum",
    "mean",
]


# Returns a decorator which registers for every given op, the function under FTVMMixedPrecisionConversionType
def register_func_to_op_list(list_ops: List):
    def decorator(func):
        for op_name in list_ops:
            register_mixed_precision_conversion(op_name, func=func)

    return decorator


def get_generic_out_dtypes(call_node: "relay.Call", mixed_precision_type: str) -> List[str]:
    """A function which returns output dtypes in a way which works for most ops.

    Parameters
    ---------
    call_node: relay.Call
        The call node containing the op.
    mixed_precision_type: str
        The target type to run the operation in.
    Returns
    -------
    output_dtypes : [str, str]
        A list of two strings. The first represents the datatype used for accumulation
        in the operation. The second represents the actual output datatype.
    """
    # Assume support accumulation dtypes <---> has out_dtype attr.
    # This is because there is no better way right now to tell which ops support accumulating
    # at different data types.
    # Some discussion here about making this better is here:
    # https://discuss.tvm.apache.org/t/rfc-relay-fp32-fp16-model-support/9994/4?u=andrewzhaoluo
    if hasattr(call_node.attrs, "out_dtype"):
        # TODO (AndrewZhaoLuo): evaluate consistent support for mixed_type accumulators
        # return ["float32", mixed_precision_type]
        return [mixed_precision_type, mixed_precision_type]

    # [accumulation_dtype, output_dtype] for the operations
    return [mixed_precision_type, mixed_precision_type]


# Functions for FTVMMixedPrecisionConversionType which
# Take in CallNodes and a DType and returns a conversion type,
# an accumulation dtype, and an output_dtype.
@register_func_to_op_list(list_ops=DEFAULT_ALWAYS_LIST)
def generic_always_op(call_node: "relay.Call", mixed_precision_type: str) -> List:
    return [MIXED_PRECISION_ALWAYS] + get_generic_out_dtypes(call_node, mixed_precision_type)


@register_func_to_op_list(list_ops=DEFAULT_FOLLOW_LIST)
def generic_follow_op(call_node: "relay.Call", mixed_precision_type: str) -> List:
    return [MIXED_PRECISION_FOLLOW] + get_generic_out_dtypes(call_node, mixed_precision_type)


@register_func_to_op_list(list_ops=DEFAULT_NEVER_LIST)
def generic_never_op(call_node: "relay.Call", mixed_precision_type: str) -> List:
    return [MIXED_PRECISION_NEVER] + get_generic_out_dtypes(call_node, mixed_precision_type)
