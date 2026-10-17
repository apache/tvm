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
Pack/unpack adds a halfword axis on the TMEM side. Split loads/stores take
separate regions for the two half-accesses and an explicit address immediate.
See PTX ISA 9.7.18.8 and the register images in 9.7.18.2.3.1.

Parameters: registers and TMEM regions in PTX destination/source order,
scope; only split 16x32bx2 accepts and requires imm_half_splitoff. No pred.
"""

from tvm_ffi import structural_equal

from tvm_tirx.backend.cuda.op import _tmem_address
from tvm_tirx.backend.cuda.ptx import PTXNamespace
from tvm_tirx.ir import Range, Var
from tvm_tirx.script import tirx as T

from .common import Context, scalar

# rows, columns per xN, registers per lane per xN
SHAPES = {
    "32x32b": (32, 1, 1),
    "16x64b": (16, 2, 1),
    "16x128b": (16, 4, 2),
    "16x256b": (16, 8, 4),
    "16x32bx2": (16, 1, 1),
}


def register_coordinate(shape, lane, register):
    if shape == "16x32bx2":
        return lane // 16, lane % 16, register
    if shape == "32x32b":
        return lane, register
    if shape == "16x64b":
        return lane // 4 + 8 * (lane % 2), lane // 2 % 2 + 2 * register
    if shape == "16x128b":
        return lane // 4 + 8 * (register % 2), lane % 4 + 4 * (register // 2)
    return lane // 4 + 8 * (register // 2 % 2), register % 2 + 2 * (lane % 4) + 8 * (register // 4)


def lower(opcode, entry, modifiers, args, params, sctx):
    shape = modifiers["shape"]
    split = shape == "16x32bx2"
    if len(args) != (3 if split else 2) or set(params) != (
        {"imm_half_splitoff"} if split else set()
    ):
        raise ValueError(
            "tcgen05.ld/st needs one register operand; "
            "16x32bx2 needs two TMEM regions and imm_half_splitoff"
        )
    split_offset = int(scalar(params["imm_half_splitoff"])) if split else 0
    packed = bool(modifiers.get("pack") or modifiers.get("unpack"))
    rows, columns, registers = SHAPES[shape]
    count = int(modifiers["num"][1:])
    columns, registers = columns * count, registers * count
    load = modifiers["action"] == "ld"
    ctx = Context(sctx, "warp")
    reg_shape = (*((2,) if split else ()), rows, columns)
    tmem_shape = (rows, columns, 2) if packed else (rows, columns)
    regs = ctx.region(
        "dst" if load else "src", args[0] if load else args[-1], reg_shape, ("local",)
    )
    memories = [
        ctx.region(f"tmem_{i}", arg, tmem_shape, ("tmem",))
        for i, arg in enumerate(args[1:] if load else args[:-1])
    ]
    for mem in memories:
        mem.require_bits(16 if packed else 32)
    tmem = memories[0]
    regs.require_bits(32)
    if load:
        regs.nonoverlapping()
    lane = Var("atom_lane", "int32")
    reg = Var("atom_register", "int32")
    ctx.analyzer.bind(lane, Range(0, 32))
    ctx.analyzer.bind(reg, Range(0, registers))
    coord = register_coordinate(shape, lane, reg)
    regs.owner(coord, lane=lane)
    base = tmem.physical((0,) * len(tmem_shape))
    if set(base) - {"TLane", "TCol"}:
        raise ValueError("TMEM pattern requires only TLane and TCol axes")
    row_base, col_base = base.get("TLane", 0), base.get("TCol", 0)
    ctx.require(row_base % rows == 0, "TMEM atom row start must align to its row count")
    elements_per_cell = 2 if packed else 1
    ctx.require(col_base % elements_per_cell == 0, "TMEM atom must start at the low half of a cell")
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
    address = T.cuda.get_tmem_addr(
        _tmem_address(tmem.tensor), row_base % 32, col_base // elements_per_cell
    )
    for part, mem in enumerate(memories):
        mem_base = mem.physical((0,) * len(tmem_shape))
        mem_row, mem_col = mem_base.get("TLane", 0), mem_base.get("TCol", 0)
        ctx.require(
            mem_col % elements_per_cell == 0,
            "TMEM half-access must start at the low half of a cell",
        )
        # The second half-access starts at taddr+immHalfSplitoff. There is no
        # implicit sixteen-row displacement for lanes 16..31 (ISA Figure 190).
        if part:
            if not structural_equal(mem.tensor.data, tmem.tensor.data):
                raise ValueError("16x32bx2 half-access regions must share the TMEM backing tensor")
            ctx.require(
                (mem_row % 32) * 65536 + mem_col // elements_per_cell
                == (row_base % 32) * 65536 + col_base // elements_per_cell + split_offset,
                "TMEM half-access address must equal taddr + imm_half_splitoff",
            )
        for half in range(2 if packed else 1):
            tcoord = (*coord[-2:], half) if packed else coord[-2:]
            physical = mem.physical(tcoord)
            ctx.require(
                physical.get("TLane", 0) == mem_row + coord[-2],
                "TMEM rows must be consecutive physical lanes",
            )
            col = mem_col + elements_per_cell * (elements_per_cell * coord[-1] + half)
            ctx.require(
                physical.get("TCol", 0) == col,
                "TMEM columns must follow the instruction's cell mapping",
            )
            ctx.require(
                (mem_row + coord[-2]) // 32 == warp,
                "TMEM half-access must belong to the executing warp",
            )
    values = [regs.value(register_coordinate(shape, T.cuda.lane_id(), r)) for r in range(registers)]
    immediate = [split_offset] if split else []
    native_args = [*values, address, *immediate] if load else [address, *immediate, *values]
    return ctx.finish([PTXNamespace()[opcode](*native_args)])
