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
"""cuBLAS Relay integration."""
from typing import Callable, List, Tuple, Dict, Optional

import tvm
import tvm.ir
from tvm import relay
from tvm import te
from tvm.relay import transform
from tvm.contrib import cublas

from ...dataflow_pattern import is_op, wildcard
from .register import register_pattern_table


def partition_for_cublas(
    mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
) -> tvm.IRModule:
    """Partition the graph to offload for cuBLAS.

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
            transform.AnnotateTarget("cublas"),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("cublas")
def pattern_table() -> List[Tuple[str, relay.Pattern, Callable]]:
    """Get the cuBLAS pattern table."""

    def matmul_pattern() -> relay.Pattern:
        """Create pattern for matrix multiply."""
        return is_op("nn.matmul")(wildcard(), wildcard())

    def check_matmul(matched: relay.Call) -> bool:
        """Check if matmul is supported by cuBLAS."""
        # Units not supported
        if matched.attrs["units"] != None:
            return False
        # Input data types can't be mixed
        if matched.args[0].checked_type.dtype != matched.args[1].checked_type.dtype:
            return False
        in_dtype = matched.args[0].checked_type.dtype
        out_dtype = matched.checked_type.dtype
        # Only the following data type combinations are supported
        if (in_dtype, out_dtype) not in [
            ("float32", "float32"),
            ("float16", "float16"),
            ("float16", "float32"),
            ("int8", "int32"),
            ("float64", "float64"),
            ("int8", "float32"),
        ]:
            return False
        # If inputs are int8, input column strides must be a multiple of 4
        if in_dtype == "int8":
            if (
                matched.args[0].checked_type.shape[1] % 4 != 0
                or matched.args[1].checked_type.shape[1] % 4 != 0
            ):
                return False

        return True

    return [
        ("cublas.matmul", matmul_pattern(), check_matmul),
    ]


_LOWER_MAP = {}


def lower_composite(comp_name: str) -> Callable:
    """Register a lowering function for a given composite function name."""

    def _register(f: Callable) -> Callable:
        _LOWER_MAP[comp_name] = f
        return f

    return _register


@lower_composite("cublas.matmul")
def lower_matmul(
    comp_func: relay.Function, target: tvm.target.Target, global_name: str
) -> tvm.runtime.Module:
    """Lower a matmul using cuBLAS."""
    op = comp_func.body
    A = te.placeholder(
        comp_func.params[0].checked_type.shape,
        name="A",
        dtype=comp_func.params[0].checked_type.dtype,
    )
    B = te.placeholder(
        comp_func.params[1].checked_type.shape,
        name="B",
        dtype=comp_func.params[1].checked_type.dtype,
    )
    C = cublas.matmul(
        A,
        B,
        transa=op.attrs["transpose_a"],
        transb=op.attrs["transpose_b"],
        dtype=comp_func.body.checked_type.dtype,
    )
    s = te.create_schedule(C.op)
    return tvm.build(s, [A, B, C], target=target, name=global_name)


@tvm._ffi.register_func("relay.ext.cublas")
def relay_to_runtime(partition: relay.Function) -> tvm.runtime.Module:
    """Compile cuBLAS Relay functions to a runtime module."""
    assert isinstance(partition, relay.Function)
    assert isinstance(partition.body, relay.Call)
    assert isinstance(partition.body.op, relay.Function)

    global_name = str(partition.attrs.global_symbol)
    target = tvm.target.cuda()
    comp_func = partition.body.op
    comp_name = comp_func.attrs["Composite"]
    assert comp_name in _LOWER_MAP

    return _LOWER_MAP[comp_name](comp_func, target, global_name)
