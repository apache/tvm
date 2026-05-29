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

"""Register every elementwise op x 3 schedules.

Loops over ``ALL_OPS`` once; no per-arity buckets, no per-op code.
"""

from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch

from ._common import match_all_scope
from .schedule_collective_reg import emit_tile_local, validate_tile_local
from .schedule_collective_smem import emit_shared, validate_shared
from .schedule_thread import emit_per_thread, validate_per_thread
from .schema import ALL_OPS, OpSpec


def _register_per_thread(spec: OpSpec) -> None:
    @register_dispatch(
        spec.name,
        "cuda",
        variant="per_thread",
        priority=10,
        when=[
            predicate("storage_scope", match_all_scope, expected_scope=["local"]),
            predicate("per_thread_valid", validate_per_thread(spec)),
        ],
    )
    def _dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _spec=spec) -> PrimFunc:
        return emit_per_thread(op, _spec, sctx)


def _register_tile_local(spec: OpSpec) -> None:
    @register_dispatch(
        spec.name,
        "cuda",
        variant="tile_local",
        priority=10,
        when=[
            predicate("storage_scope", match_all_scope, expected_scope=["local"]),
            predicate("tile_local_valid", validate_tile_local(spec)),
        ],
    )
    def _dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _spec=spec) -> PrimFunc:
        return emit_tile_local(op, _spec, sctx)


def _register_shared(spec: OpSpec) -> None:
    @register_dispatch(
        spec.name,
        "cuda",
        variant="shared_distributed",
        priority=10,
        when=[
            predicate("storage_scope", match_all_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_shared(spec)),
        ],
    )
    def _dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _spec=spec) -> PrimFunc:
        return emit_shared(op, _spec, sctx)


for _spec in ALL_OPS.values():
    _register_per_thread(_spec)
    _register_tile_local(_spec)
    _register_shared(_spec)


__all__: list[str] = []
