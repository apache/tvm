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
from typing import Callable, List, Tuple, Dict, Optional

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


def partition_for_cudnn(
    mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
) -> tvm.IRModule:
    """Partition the graph to offload for cuDNN.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        Constant input parameters.

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

    def check_softmax(matched: relay.Call) -> bool:
        """Check if softmax is supported by cuDNN."""
        if matched.args[0].checked_type.dtype not in ["float64", "float32", "float16"]:
            return False

        return True

    return [
        ("cudnn.softmax", softmax_pattern(), check_softmax),
    ]


@lower_composite("cudnn.softmax")
def _lower_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a softmax using cuDNN."""
    return cudnn.softmax(inputs[0], axis=op.attrs["axis"])
