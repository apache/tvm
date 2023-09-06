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
"""Default legalization function for ccl operators."""

from tvm import tir
from ...block_builder import BlockBuilder
from ...expr import Call, Expr, ShapeExpr
from ...op import call_pure_packed, call_dps_packed
from ...struct_info import TensorStructInfo
from .common import register_legalize


@register_legalize("relax.ccl.allreduce")
def _allreduce(_bb: BlockBuilder, call: Call) -> Expr:
    op_type_str = call.attrs.op_type
    op_type_map = {
        "sum": 0,
        "prod": 1,
        "min": 2,
        "max": 3,
        "avg": 4,
    }
    if op_type_str not in op_type_map:
        raise ValueError(
            f"Unsupported reduction operation: {op_type_str}. "
            f"Supported operations are {op_type_map.keys()}."
        )
    return call_pure_packed(
        "runtime.disco.allreduce",
        call.args[0],
        ShapeExpr([op_type_map[op_type_str]]),
        sinfo_args=call.args[0].struct_info,
    )


@register_legalize("relax.ccl.broadcast_from_worker0")
def _broadcast_from_worker0(_bb: BlockBuilder, call: Call) -> Expr:
    return call_pure_packed(
        "runtime.disco.broadcast_from_worker0",
        call.args[0],
        sinfo_args=call.args[0].struct_info,
    )


@register_legalize("relax.ccl.scatter_from_worker0")
def _scatter_from_worker0(_bb: BlockBuilder, call: Call) -> Expr:
    output_shape = []
    for i, shape_value in enumerate(call.args[0].struct_info.shape.values):
        if i == 0:
            output_shape.append(tir.div(shape_value, call.attrs.num_workers))
        else:
            output_shape.append(shape_value)
    return call_pure_packed(
        "runtime.disco.scatter_from_worker0",
        call.args[0],
        sinfo_args=TensorStructInfo(
            shape=output_shape,
            dtype=call.args[0].struct_info.dtype,
            vdevice=call.args[0].struct_info.vdevice,
        ),
    )
