from typing import List, Tuple

import tvm
from tvm.relay import transform
from tvm.relay.dataflow_pattern import (
    is_op,
    wildcard,
    DFPattern,
)
from tvm.relay.op.contrib.register import register_pattern_table


def partition_for_nnapi(mod: tvm.IRModule) -> tvm.IRModule:
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("nnapi"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(bind_constants=False),
        ]
    )

    mod = seq(mod)
    return mod


def elementwise_binary_patterns(pattern_name: str, op_name: str) -> Tuple[str, DFPattern]:
    input0 = wildcard()
    input1 = wildcard()

    pattern = is_op(op_name)(input0, input1)

    return (pattern_name, pattern)


def unary_patterns(pattern_name: str, op_name: str) -> Tuple[str, DFPattern]:
    input0 = wildcard()
    pattern = is_op(op_name)(input0)

    return (pattern_name, pattern)


def dense_pattern():
    input0 = wildcard()
    input1 = wildcard()

    pattern = is_op("nn.dense")(input0, input1).optional(
        lambda x: is_op("nn.bias_add")(x, wildcard())
    )

    return ("nnapi.fully_connected", pattern)


def conv2d_pattern():
    input0 = wildcard()
    input1 = wildcard()

    pattern = is_op("nn.conv2d")(input0, input1).optional(
        lambda x: is_op("nn.bias_add")(x, wildcard())
    )
    return ("nnapi.conv2d", pattern)

def max_pool2d_pattern():
    input0 = wildcard()

    pattern = is_op("nn.max_pool2d")(input0)
    return ("nnapi.max_pool_2d", pattern)

@register_pattern_table("nnapi")
def pattern_table() -> List[Tuple[str, DFPattern]]:
    return [
        elementwise_binary_patterns("nnapi.add", "add"),
        elementwise_binary_patterns("nnapi.mul", "multiply"),
        elementwise_binary_patterns("nnapi.div", "divide"),
        elementwise_binary_patterns("nnapi.sub", "subtract"),
        elementwise_binary_patterns("nnapi.equal", "equal"),
        elementwise_binary_patterns("nnapi.greater", "greater"),
        elementwise_binary_patterns("nnapi.greater_equal", "greater_equal"),
        elementwise_binary_patterns("nnapi.less", "less"),
        elementwise_binary_patterns("nnapi.less_equal", "less_equal"),
        elementwise_binary_patterns("nnapi.not_equal", "not_equal"),
        unary_patterns("nnapi.floor", "floor"),
        unary_patterns("nnapi.relu", "nn.relu"),
        unary_patterns("nnapi.logistic", "sigmoid"),
        unary_patterns("nnapi.tanh", "tanh"),
        unary_patterns("nnapi.abs", "abs"),
        unary_patterns("nnapi.exp", "exp"),
        unary_patterns("nnapi.log", "log"),
        unary_patterns("nnapi.neg", "negative"),
        dense_pattern(),
        conv2d_pattern(),
        max_pool2d_pattern(),
    ]
