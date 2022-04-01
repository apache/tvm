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
from .register import register_pattern_table


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


_LowerFunc = Callable[[relay.Call, List[te.Tensor]], te.Tensor]
_LOWER_MAP: Dict[str, _LowerFunc] = {}


def _lower_composite(comp_name: str) -> Callable[[_LowerFunc], _LowerFunc]:
    """Register a lowering function for a given composite function name."""

    def _register(f: _LowerFunc) -> _LowerFunc:
        _LOWER_MAP[comp_name] = f
        return f

    return _register


@tvm._ffi.register_func("relay.ext.cudnn")
def relay_to_runtime(partition: relay.Function) -> tvm.runtime.Module:
    """Compile cuDNN Relay functions to a runtime module."""
    assert isinstance(partition, relay.Function)
    assert isinstance(partition.body, relay.Call)
    assert isinstance(partition.body.op, relay.Function)

    global_name = str(partition.attrs.global_symbol)
    target = tvm.target.cuda()
    comp_func = partition.body.op
    comp_name = comp_func.attrs["Composite"]
    assert comp_name in _LOWER_MAP
    assert isinstance(comp_func.body, relay.Call)

    op = comp_func.body
    inputs = []
    for i, param in enumerate(comp_func.params):
        inputs.append(
            te.placeholder(
                param.checked_type.shape,
                name=f"input_{i}",
                dtype=param.checked_type.dtype,
            )
        )

    output = _LOWER_MAP[comp_name](op, inputs)
    prim_func = te.create_prim_func(inputs + [output])
    return tvm.build(prim_func, target=target, name=global_name)


@_lower_composite("cudnn.softmax")
def _lower_softmax(op: relay.Call, inputs: List[te.Tensor]) -> te.Tensor:
    """Lower a softmax using cuDNN."""
    return cudnn.softmax(inputs[0], axis=op.attrs["axis"])
