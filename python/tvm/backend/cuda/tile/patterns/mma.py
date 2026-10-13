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

"""The fixed m16n8k8/m16n8k16 warp MMA fragment images.

D/C: (*group, *batch, M, N, 16, 8), f32 accumulator elements.
A:   (*group, *batch, M, K, 16, k/2), b32 words packing two K elements.
B:   (*group, *batch, N, K,  8, k/2), b32 words packing two K elements.

M/N/K are instruction-repeat axes, not matrix element axes. The innermost K
loop feeds C to the first instruction and D to each subsequent instruction.
Scalar C is an explicit broadcast (e.g. f32(0)); alpha/beta preparation belongs
to the caller. PTX ISA 9.7.16.5.6 defines each lane's fragment coordinates.
"""

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Range, TensorRegion, Var
from tvm.script import tirx as T

from .common import Context


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 4 or params:
        raise ValueError("mma.sync requires D, A, B, C without extra operands")
    shape = modifiers["shape"]
    if shape not in ("m16n8k8", "m16n8k16") or not opcode.endswith(
        (".f32.f16.f16.f32", ".f32.bf16.bf16.f32")
    ):
        raise ValueError(
            "mma pattern requires m16n8k8/k16 with f16/bf16 inputs and f32 accumulation"
        )
    k = 8 if shape == "m16n8k8" else 16
    ctx = Context(sctx, "warp")
    ng = len(ctx.group_shape)
    d_shape = tuple(int(r.extent) for r in args[0].region)[ng:]
    a_shape = tuple(int(r.extent) for r in args[1].region)[ng:]
    if len(d_shape) < 4 or len(a_shape) != len(d_shape):
        raise ValueError("mma requires explicit M/N and M/K repeat axes before each atom")
    batch, m, n = d_shape[:-4], d_shape[-4], d_shape[-3]
    nk = a_shape[-3]
    nb = len(batch)
    ctx.loops((*batch, m, n, nk))
    mn_axes = (*range(nb), nb, nb + 1)
    mk_axes = (*range(nb), nb, nb + 2)
    nk_axes = (*range(nb), nb + 1, nb + 2)
    d = ctx.region("D", args[0], (16, 8), ("local",), repeat_axes=mn_axes)
    a = ctx.region("A", args[1], (16, k // 2), ("local",), repeat_axes=mk_axes)
    b = ctx.region("B", args[2], (8, k // 2), ("local",), repeat_axes=nk_axes)
    c = (
        ctx.region("C", args[3], (16, 8), ("local",), repeat_axes=mn_axes)
        if isinstance(args[3], TensorRegion)
        else args[3]
    )
    for operand in (d, a, b):
        operand.require_bits(32)
    d.nonoverlapping()
    lane = Var("atom_lane", "int32")
    ctx.analyzer.bind(lane, Range(0, 32))

    def a_coord(lane, reg):
        return lane // 4 + 8 * (reg % 2), lane % 4 + 4 * (reg // 2)

    def b_coord(lane, reg):
        return lane // 4, lane % 4 + 4 * reg

    def d_coord(lane, reg):
        return lane // 4 + 8 * (reg // 2), 2 * (lane % 4) + reg % 2

    for operand, count, coord in ((a, k // 4, a_coord), (b, k // 8, b_coord), (d, 4, d_coord)):
        for reg in range(count):
            operand.owner(coord(lane, reg), lane=lane)
    if isinstance(args[3], TensorRegion):
        c.require_bits(32)
        for reg in range(4):
            c.owner(d_coord(lane, reg), lane=lane)
    lane = T.cuda.lane_id()
    ds = [d.value(d_coord(lane, r)) for r in range(4)]
    cs = (
        [c.value(d_coord(lane, r)) for r in range(4)]
        if isinstance(args[3], TensorRegion)
        else [c] * 4
    )
    cs = [
        T.if_then_else(ctx.repeat[-1] == 0, initial, previous) for initial, previous in zip(cs, ds)
    ]
    native = [
        *ds,
        *(a.value(a_coord(lane, r)) for r in range(k // 4)),
        *(b.value(b_coord(lane, r)) for r in range(k // 8)),
        *cs,
    ]
    return ctx.finish([PTXNamespace()[opcode](*native)])
