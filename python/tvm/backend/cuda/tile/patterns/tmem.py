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

"""Fixed tcgen05.ld/st images in native 32-bit storage words.

Canonical axes: (*group, *repeat, row, column). The shape and xN qualifier
fix the final two extents. Groups select warps; a repeat may select a second
16-row half slab or another column range. Neither choice changes the opcode.
See PTX ISA 9.7.18.8 and the data movement shapes in 9.7.18.4.2.
"""

from tvm.backend.cuda.op import _tmem_address
from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Range, Var
from tvm.script import tirx as T

from .common import Context

# rows, columns per xN, registers per lane per xN
SHAPES = {"32x32b": (32, 1, 1), "16x64b": (16, 2, 1), "16x128b": (16, 4, 2), "16x256b": (16, 8, 4)}


def register_coordinate(shape, lane, register):
    if shape == "32x32b":
        return lane, register
    if shape == "16x64b":
        return lane // 4 + 8 * (lane % 2), lane // 2 % 2 + 2 * register
    if shape == "16x128b":
        return lane // 4 + 8 * (register % 2), lane % 4 + 4 * (register // 2)
    return lane // 4 + 8 * (register // 2 % 2), register % 2 + 2 * (lane % 4) + 8 * (register // 4)


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2 or params:
        raise ValueError("tcgen05.ld/st requires dst and src without extra operands")
    if modifiers.get("pack") or modifiers.get("unpack"):
        raise ValueError(
            "this pattern uses dense b32 words; pack/unpack requires a separate cell representation"
        )
    shape = modifiers["shape"]
    rows, columns, registers = SHAPES[shape]
    count = int(modifiers["num"][1:])
    columns, registers = columns * count, registers * count
    load = entry.name == "tcgen05_ld"
    ctx = Context(sctx, "warp")
    dst = ctx.region("dst", args[0], (rows, columns), ("local",) if load else ("tmem",))
    src = ctx.region("src", args[1], (rows, columns), ("tmem",) if load else ("local",))
    tmem, regs = (src, dst) if load else (dst, src)
    tmem.require_bits(32)
    regs.require_bits(32)
    if load:
        regs.nonoverlapping()
    lane = Var("atom_lane", "int32")
    reg = Var("atom_register", "int32")
    ctx.analyzer.bind(lane, Range(0, 32))
    ctx.analyzer.bind(reg, Range(0, registers))
    coord = register_coordinate(shape, lane, reg)
    regs.owner(coord, lane=lane)
    base = tmem.physical((0, 0))
    physical = tmem.physical(coord)
    if set(base) - {"TLane", "TCol"}:
        raise ValueError("TMEM pattern requires only TLane and TCol axes")
    row_base, col_base = base.get("TLane", 0), base.get("TCol", 0)
    ctx.require(
        physical.get("TLane", 0) == row_base + coord[0],
        "TMEM rows must be consecutive physical lanes",
    )
    ctx.require(
        physical.get("TCol", 0) == col_base + coord[1], "TMEM columns must be consecutive b32 cells"
    )
    ctx.require(row_base % rows == 0, "TMEM atom row start must align to its row count")
    warp = ctx.owners.get("wid_in_wg")
    if warp is None and "warpid" in ctx.owners:
        warp = ctx.owners["warpid"] % 4
    if warp is None:
        active_warp = sctx.inter.get("warpid")
        if active_warp is None or int(active_warp[0]) != 1:
            raise ValueError(
                "TMEM warp scope requires a statically selected warp; use explicit warp group axes"
            )
        warp = int(active_warp[1]) % 4
    ctx.require(row_base // 32 == warp, "TMEM lane partition must belong to the executing warp")
    address = T.cuda.get_tmem_addr(_tmem_address(tmem.tensor), row_base % 32, col_base)
    values = [regs.value(register_coordinate(shape, T.cuda.lane_id(), r)) for r in range(registers)]
    native_args = [*values, address] if load else [address, *values]
    return ctx.finish([PTXNamespace()[opcode](*native_args)])
