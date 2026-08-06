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

"""Register each op in ``ALL_OPS`` for both dispatch variants (``reg``, ``smem``).

Mirrors copy PR-640's two-variant model: scope-pair drives the dispatch
selection, the underlying algorithm (induced vs synthesized) follows.
"""

from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch

from .ops import ALL_OPS
from .reg import emit_reg, is_reg_ewise
from .smem import emit_smem, is_smem_ewise


def _register_reg(spec) -> None:
    @register_dispatch(
        spec.name,
        "cuda",
        variant="reg",
        priority=10,
        when=[predicate(f"{spec.name}_reg", is_reg_ewise(spec))],
    )
    def _dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _spec=spec) -> PrimFunc:
        return emit_reg(op, _spec, sctx)


def _register_smem(spec) -> None:
    @register_dispatch(
        spec.name,
        "cuda",
        variant="smem",
        priority=10,
        when=[predicate(f"{spec.name}_smem", is_smem_ewise(spec))],
    )
    def _dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _spec=spec) -> PrimFunc:
        return emit_smem(op, _spec, sctx)


for _spec in ALL_OPS.values():
    _register_reg(_spec)
    _register_smem(_spec)


__all__: list[str] = []
