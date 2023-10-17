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
# pylint: disable=invalid-name
"""Pattern table for ComposableKernel backend"""

import tvm

from tvm.relax import transform
from tvm.relax.transform import PatternCheckContext
import tvm.contrib.composable_kernel
from tvm.relax.dpl import rewrite_call
from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_matmul_pattern


def _check_matmul(context: PatternCheckContext) -> bool:
    """Check if the given matmul workload can be offloaded to ComposableKernel."""
    return True


def _get_activation_from_name(pattern_name):
    if "_relu" in pattern_name:
        return "relax.nn.relu"
    elif "_gelu_tanh" in pattern_name:
        return "relax.nn.gelu_tanh"
    elif "_gelu" in pattern_name:
        return "relax.nn.gelu"
    elif "_silu" in pattern_name:
        return "relax.nn.silu"
    else:
        return None


def matmul_patterns():
    """
    Returns a list of all matmul patterns in ComposableKernel BYOC backend.
    """

    def _matmul_pattern(pattern_name):
        transposed_rhs = "_transposed" in pattern_name
        with_bias = "_bias" in pattern_name
        activation = _get_activation_from_name(pattern_name)

        return (
            pattern_name,
            *make_matmul_pattern(
                transposed_rhs=transposed_rhs,
                with_bias=with_bias,
                activation=activation,
            ),
            _check_matmul,
        )

    return [
        _matmul_pattern("composable_kernel.matmul"),
        # _matmul_pattern("composable_kernel.matmul_bias"),
        # _matmul_pattern("composable_kernel.matmul_bias_relu"),
        # _matmul_pattern("composable_kernel.matmul_bias_gelu"),
        _matmul_pattern("composable_kernel.matmul_transposed"),
        # _matmul_pattern("composable_kernel.matmul_transposed_bias"),
        # _matmul_pattern("composable_kernel.matmul_transposed_bias_relu"),
        # _matmul_pattern("composable_kernel.matmul_transposed_bias_gelu"),
    ]


register_patterns(
    [
        *matmul_patterns(),
    ]
)


def partition_for_composable_kernel(mod, annotate_codegen=True):
    """
    Partition the input module into ComposableKernel-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    annotate_codegen: bool
        Whether to wrap each created composite function with another function, whose
        body consists only of a call to the composite function. See the doc of FuseOpsByPattern
        for more detail.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        compiled by the ComposableKernel backend.
    """
    patterns = get_patterns_with_prefix("composable_kernel")
    return tvm.transform.Sequential(
        [
            transform.FuseOpsByPattern(
                patterns, bind_constants=False, annotate_codegen=annotate_codegen
            ),
            transform.AllocateWorkspace(),
        ]
    )(mod)
