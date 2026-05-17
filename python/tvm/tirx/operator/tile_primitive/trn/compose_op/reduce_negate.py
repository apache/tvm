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

"""Implementation of ReduceNegate dispatch."""

from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.ops import ReduceNegate

from ..reduction.utils import reduction_trn
from .utils import optype_table


def reduce_negate_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Generate a TRN schedule for reduce negate operations."""
    op = TilePrimitiveCall.downcast(op)
    assert isinstance(op, ReduceNegate), f"invalid operator downcast: {op}"
    return reduction_trn(op, optype_table[op.reduce_op], sctx, negate=True)


@register_dispatch(
    "reduce_negate",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.scope_kind == "kernel",
                f"unsupported exec_scope {sctx.scope_kind}",
            ),
        )
    ],
)
def reduce_negate_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return reduce_negate_trn(op, sctx)
