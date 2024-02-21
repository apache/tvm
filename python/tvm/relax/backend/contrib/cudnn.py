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

"""Pattern table for cuDNN backend"""
from tvm.relax import transform
from tvm.relax.transform import PatternCheckContext

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_conv2d_pattern
from ..utils import has_leaking_intermediate_variables


def _is_supported_dtype(lhs_dtype, rhs_dtype):
    """Check if dtypes in the given workload are supported by cuDNN BYOC."""
    return (lhs_dtype == "float16" and rhs_dtype == "float16") or (
        lhs_dtype == "float32" and rhs_dtype == "float32"
    )


def _is_supported_format(data_layout, kernel_layout):
    """Check if layouts in the given workload are supported by cuDNN BYOC."""
    return (data_layout == "NHWC" and kernel_layout == "OHWI") or (
        data_layout == "NCHW" and kernel_layout == "OIHW"
    )


def _check_conv2d(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    # Retrieve the annotated expression from context
    conv2d_call = context.annotated_expr["root"]
    input_expr = context.annotated_expr["input"]
    weight_expr = context.annotated_expr["weight"]

    # Check if the data types of input and weights are supported by cuDNN BYOC
    input_dtype = input_expr.struct_info.dtype
    weight_dtype = weight_expr.struct_info.dtype
    if not _is_supported_dtype(input_dtype, weight_dtype):
        return False

    input_layout = conv2d_call.attrs.data_layout
    weight_layout = conv2d_call.attrs.kernel_layout
    if not _is_supported_format(input_layout, weight_layout):
        return False

    return True


register_patterns(
    [
        (
            "cudnn.conv2d.nhwc_ohwi",
            *make_conv2d_pattern(
                with_bias=False,
            ),
            _check_conv2d,
        ),
        (
            "cudnn.conv2d.nhwc_ohwi_bias",
            *make_conv2d_pattern(
                with_bias=True,
            ),
            _check_conv2d,
        ),
        (
            "cudnn.conv2d.nhwc_ohwi_bias_relu",
            *make_conv2d_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
            _check_conv2d,
        ),
    ]
)


def partition_for_cudnn(mod):
    """
    Partition the input module into cuDNN-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the cuDNN backend.
    """

    patterns = get_patterns_with_prefix("cudnn")
    return transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
