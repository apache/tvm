import enum
from typing import *

import tvm
from tvm import relay


def create_op_list(op_list: List[str]) -> List[tvm.ir.Op]:
    return [relay.op.get(op_name) for op_name in op_list]


class ConversionCategory(enum.Enum):
    """
    Green: will cast to fp16 version of the op which takes in fp16 inputs
    Gray: may cast after doing analysis
    Red: will not cast to fp16 version
    """

    GREEN = "Green"
    GRAY = "Gray"
    RED = "Red"


class DefaultColorer:
    # Default lists inspired from TF's classifications:
    # https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
    # They might have a bias toward NVidia's Tensor Cores so be aware and modify lists per your hardware choice.

    # These should always be done in fp16 if possible
    DEFAULT_GREEN_LIST = {
        "nn.conv1d",
        "nn.conv2d",
        "nn.conv3d",
        "nn.conv1d_transpose",
        "nn.conv2d_transpose",
        "nn.conv3d_transpose",
        "nn.dense",
    }

    # These can be done in fp16 or fp32 with no point in casting between
    DEFAULT_GRAY_LIST = {
        # These ops add new data or change shape
        "nn.pad",
        "nn.batch_flatten",
        # Simple arithmetic
        "add",
        "nn.bias_add",
        "nn.batch_norm",
        # Simple activations
        "nn.relu",
        "nn.leaky_relu",
        "nn.prelu",
        "nn.dropout",
        # Pooling operations
        "nn.max_pool1d",
        "nn.max_pool2d",
        "nn.max_pool3d",
        "nn.avg_pool1d",
        "nn.avg_pool2d",
        "nn.avg_pool3d",
        ## "nn.global_max_pool1d", # does not exist
        "nn.global_max_pool2d",
        ## "nn.global_max_pool3d", # does not exist
        ## "nn.global_avg_pool1d", # does not exist
        "nn.global_avg_pool2d",
        ## "nn.global_avg_pool3d", # does not exist
        "nn.adaptive_max_pool1d",
        "nn.adaptive_max_pool2d",
        "nn.adaptive_max_pool3d",
        "nn.adaptive_avg_pool1d",
        "nn.adaptive_avg_pool2d",
        "nn.adaptive_avg_pool3d",
    }

    # These should always be done in fp32
    DEFAULT_RED_LIST = {
        # Activations with exponents or division
        "nn.cross_entropy",
        "nn.cross_entropy_with_logits",
        "nn.softmax",
        # Other
        "nn.l2_normalize",
    }

    def __init__(
        self,
        green_list: List[str] = DEFAULT_GREEN_LIST,
        gray_list: List[str] = DEFAULT_GRAY_LIST,
        red_list: List[str] = DEFAULT_RED_LIST,
    ):
        # Convert each list to entry
        green_list = create_op_list(green_list)
        gray_list = create_op_list(gray_list)
        red_list = create_op_list(red_list)

        # Create lookup table mapping relay op -> color in grpah
        self.lookup_table = {}
        for op_list, val in [
            (green_list, ConversionCategory.GREEN),
            (gray_list, ConversionCategory.GRAY),
            (red_list, ConversionCategory.RED),
        ]:
            for op in op_list:
                self.lookup_table[op] = val

    def __call__(self, call_node: relay.Call, ignore_missing: bool = False) -> ConversionCategory:
        if call_node.op not in self.lookup_table:
            if ignore_missing:
                return ConversionCategory.RED
            else:
                raise ValueError(f"Unknown op {call_node.op}")

        return self.lookup_table[call_node.op]
