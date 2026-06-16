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

"""FMA op: ``Tx.fma(dst, a, b, c)`` → ``dst = a*b + c``.

Attaches ``fma_f32x2`` VecImpl for sm_100+ f32; falls back to scalar
``a*b + c`` otherwise.
"""

from __future__ import annotations

from tvm.tirx import BufferRegion, TilePrimitiveCall

from ..vec_emit.fma_f32x2 import FMA_F32X2_IMPL
from . import OpSpec, Plan, SrcSpec


def _parse_fma(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    _dst: BufferRegion = op.args[0]
    args = op.args[1:4]
    srcs: list[SrcSpec] = []
    for a in args:
        if isinstance(a, BufferRegion):
            srcs.append(SrcSpec(buf_region=a))
        else:
            srcs.append(SrcSpec(scalar=a))
    return Plan(dst=_dst, srcs=srcs, extras={}), None


def _compute_fma(src_vals, extras, dt):
    return src_vals[0] * src_vals[1] + src_vals[2]


FMA_OPS: dict[str, OpSpec] = {
    "fma": OpSpec("fma", _parse_fma, _compute_fma, vec_impls=[FMA_F32X2_IMPL]),
}
