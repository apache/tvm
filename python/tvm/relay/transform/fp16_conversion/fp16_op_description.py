from typing import *

from tvm import relay
from tvm.relay.transform.fp16_conversion import graph_colors

FP16OutDtype = NamedTuple("FP16OutDtype", [("accumulation_dtype", str), ("output_dtype", str)])


class DefaultFP16TypeDefinition:
    # These fp16 operations accumulate their results in a 32 bit buffer
    DEFAULT_FP32_ACCUMULATION_LIST = [
        "nn.conv1d",
        "nn.conv2d",
        "nn.conv3d",
        "nn.conv1d_transpose",
        "nn.conv2d_transpose",
        "nn.conv3d_transpose",
        "nn.dense",
        "nn.avg_pool1d",
        "nn.avg_pool2d",
        "nn.avg_pool3d",
        "nn.adaptive_avg_pool1d",
        "nn.adaptive_avg_pool2d",
        "nn.adaptive_avg_pool3d",
    ]

    # These fp16 operations return fp32 results. If an operation has
    # an fp32 accumulator but is not in this list, it is assumed the accumulator
    # is quantized to 16 bits before being used in other operations.
    DEFAULT_FP32_OUTPUT_LIST = []

    SUPPORTED_OPS = {
        
    }

    def __init__(
        self,
        fp32_accumulation_ops: List[str] = DEFAULT_FP32_ACCUMULATION_LIST,
        fp32_output_ops: List[str] = DEFAULT_FP32_OUTPUT_LIST,
    ):
        self.fp32_accumulation_ops = set(graph_colors.create_op_list(fp32_accumulation_ops))
        self.fp32_output_ops = set(graph_colors.create_op_list(fp32_output_ops))

    def __call__(self, call_node: relay.Call) -> FP16OutDtype:
        accumulation_dtype = "float32" if call_node.op in self.fp32_accumulation_ops else "float16"
        output_dtype = "float32" if call_node.op in self.fp32_output_ops else "float16"
        return FP16OutDtype(accumulation_dtype, output_dtype)
