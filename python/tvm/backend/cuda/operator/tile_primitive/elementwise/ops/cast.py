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

"""Cast op: ``Tx.cast(dst, src)``. Outer ``Tx.cast(..., dst.dtype)`` in the
schedule handles the scalar conversion; the vec-impl packs pairs via
CUDA intrinsics like ``__float22half2_rn``."""

from __future__ import annotations

from tvm.tirx import BufferRegion, TilePrimitiveCall

from ..vec_emit.cast_vec2 import CAST_VEC2_IMPL
from . import OpSpec, Plan, SrcSpec


def _parse_cast(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    _dst: BufferRegion = op.args[0]
    _src = op.args[1]
    if not isinstance(_src, BufferRegion):
        return None, "cast src must be a buffer region"
    return Plan(dst=_dst, srcs=[SrcSpec(buf_region=_src)], extras={}), None


def _compute_cast(src_vals, extras, dt):
    # Schedule wraps with Tx.cast(..., dst.dtype) — just pass through.
    return src_vals[0]


CAST_OPS: dict[str, OpSpec] = {
    "cast": OpSpec("cast", _parse_cast, _compute_cast, vec_impls=[CAST_VEC2_IMPL]),
}
