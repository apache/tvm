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

"""Explicit tcgen05 MMA descriptors and canonical M/N/K repetition.

D: (*group, *batch, M_repeat, N_repeat, per_CTA_M, instruction_N)
A: (*group, *batch, M_repeat, K_repeat, per_CTA_M, instruction_K)
B: (*group, *batch, N_repeat, K_repeat, per_CTA_N, instruction_K)

For weight-stationary A-in-TMEM the atom includes its explicit leading bank
axis. Every call spells instruction_shape=(M,N,K), formats, major order, and
shared descriptor fields. The instruction's CTA group and .ws select the
hardware datapath; the operand's layout never selects an instruction mode.
"""

from tvm.backend.cuda.cpp.descriptors import (
    _get_tcgen05_mma_kind,
    _get_tcgen05_mma_scale_vec_size,
    encode_instr_descriptor_block_scaled_uint32,
    encode_instr_descriptor_dense_uint32,
)
from tvm.backend.cuda.layout import tmem_mma_operand_layout
from tvm.backend.cuda.op import _tmem_address
from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Range, Var
from tvm.ir.prim import Or
from tvm.runtime import MemoryScope
from tvm.script import tirx as T

from .common import Context, scalar
from .descriptors import matrix_descriptor, swizzle_bytes
from .tma import sequence


def _tmem_operand(ctx, operand, expected, atom_shape):
    origin = (0,) * len(atom_shape)
    base = operand.physical(origin)
    coords = [Var(f"{operand.name}_atom_{i}", "int32") for i in range(len(atom_shape))]
    for var, size in zip(coords, atom_shape):
        ctx.analyzer.bind(var, Range(0, size))
    physical = operand.physical(coords)
    image = expected.apply(*coords, shape=atom_shape)
    if set(physical) - {"TLane", "TCol"}:
        raise ValueError(f"{operand.name}: expected a TMEM image")
    for axis in ("TLane", "TCol"):
        ctx.require(
            physical.get(axis, 0) == base.get(axis, 0) + image.get(axis, 0),
            f"{operand.name}: layout disagrees with the explicit MMA datapath",
        )

    def replica(layout):
        return [(i.axis.name, int(i.extent), int(i.stride)) for i in layout.replica]

    if replica(operand.layout) != replica(expected):
        raise ValueError(f"{operand.name}: replicas disagree with the explicit MMA datapath")
    ctx.require(
        base.get("TCol", 0) * operand.bits % 32 == 0, "TMEM MMA origin requires whole b32 cells"
    )
    return T.cuda.get_tmem_addr(
        _tmem_address(operand.tensor),
        base.get("TLane", 0),
        base.get("TCol", 0) * operand.bits // 32,
    )


def _shared_operand(ctx, operand, atom_shape, descriptor, major):
    fields = tuple(int(scalar(x)) for x in sequence(descriptor))
    if len(fields) != 3:
        raise ValueError("shared descriptor is (ldo_bytes, sdo_bytes, swizzle)")
    ldo, sdo, swizzle = fields
    if major == "mn" and operand.bits == 32 and swizzle != 4:
        raise ValueError("32-bit MN-major requires 128B swizzling with 32B atomicity")
    if swizzle == 4 and major != "mn":
        raise ValueError("128B swizzling with 32B atomicity requires MN-major")
    if major == "mn" and operand.bits != 32 and swizzle == 4:
        raise ValueError("128B swizzling with 32B atomicity requires 32-bit MN-major elements")
    row_bytes = 128 if swizzle == 4 else 16 << swizzle
    mn, k = atom_shape
    # PTX's unswizzled MN-major atom exchanges leading and stride offsets.
    row_group, column_group = (ldo, sdo) if major == "mn" and swizzle == 0 else (sdo, ldo)
    base = operand.memory((0, 0)) * operand.bits
    if swizzle:
        base = swizzle_bytes(base // 8, swizzle) * 8 + base % 8
    # Atom coordinates are fixed matrix coordinates. Major order changes
    # the descriptor walk, not the logical matrix multiplication.
    rows_per_group = 4 if swizzle == 4 else 8
    for i in range(mn):
        for j in range(k):
            row, contiguous = (j, i) if major == "mn" else (i, j)
            offset = row // rows_per_group * row_group * 8 + row % rows_per_group * row_bytes * 8
            offset += contiguous * operand.bits // (
                row_bytes * 8
            ) * column_group * 8 + contiguous * operand.bits % (row_bytes * 8)
            expected_bits = base + offset
            if swizzle:
                expected_bits = swizzle_bytes(expected_bits // 8, swizzle) * 8 + expected_bits % 8
            ctx.require(
                operand.memory((i, j)) * operand.bits == expected_bits,
                f"{operand.name}: view disagrees with explicit shared descriptor/major order",
            )
    return matrix_descriptor(ctx, operand, ldo, sdo, swizzle, (0, 0))


def _scale_operand(ctx, region, name, rows, sf_per_mma, repeat_axes):
    operand = ctx.region(name, region, (rows, sf_per_mma), ("tmem",), repeat_axes=repeat_axes)
    operand.require_bits(8)
    base = operand.physical((0, 0))
    for row in range(rows):
        for sf in range(sf_per_mma):
            actual = operand.physical((row, sf))
            ctx.require(
                actual.get("TLane", 0) == base.get("TLane", 0) + row % 32,
                f"{name}: scale rows must use lane row%32",
            )
            ctx.require(
                actual.get("TCol", 0) == base.get("TCol", 0) + 4 * (row // 32) + sf,
                f"{name}: scale row groups must occupy consecutive b32 cells",
            )
    replicas = [(i.axis.name, int(i.extent), int(i.stride)) for i in operand.layout.replica]
    if replicas != [("TLane", 4, 32)]:
        raise ValueError(f"{name}: scale factors require four explicit 32-lane replicas")
    col = base.get("TCol", 0)
    ctx.require(
        col % sf_per_mma == 0,
        f"{name}: scale-factor ID must align to the instruction's scale vector",
    )
    return (
        T.cuda.get_tmem_addr(_tmem_address(operand.tensor), base.get("TLane", 0), col // 4),
        col % 4,
        str(operand.dtype),
    )


def lower(opcode, entry, modifiers, args, params, sctx):
    block_scale = bool(modifiers.get("block_scale"))
    ws = bool(modifiers.get("ws"))
    if len(args) != (5 if block_scale else 3):
        raise ValueError("tcgen05.mma requires D/A/B, plus SFA/SFB for block_scale")
    cta_group = int(modifiers["cta_group"].split("::")[1])
    shape = tuple(scalar(x) for x in sequence(params.pop("instruction_shape")))
    if len(shape) != 3 or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("instruction_shape requires three positive integers (M, N, K)")
    M, N, K = shape
    if ws and (cta_group != 1 or N not in (64, 128, 256)):
        raise ValueError("weight-stationary MMA requires CTA group 1 and N=64,128,256")
    am = str(scalar(params.pop("a_major", "k")))
    bm = str(scalar(params.pop("b_major", "k")))
    if am not in ("k", "mn") or bm not in ("k", "mn"):
        raise ValueError("MMA major order must be k or mn")
    ctx = Context(sctx)
    ng = len(ctx.group_shape)
    d_shape = tuple(int(r.extent) for r in args[0].region)[ng:]
    # Every operand carries two explicit repeat axes before its atom.
    if len(d_shape) < 4:
        raise ValueError("tcgen05.mma D requires M/N repeat axes before the atom")
    batch, mr, nr = d_shape[:-4], d_shape[-4], d_shape[-3]
    nb = len(batch)
    a_tmem = args[1].source.ty.scope == MemoryScope("tmem")
    a_atom = (2, M // cta_group, K) if a_tmem and ws and M == 64 else (M // cta_group, K)
    a_shape = tuple(int(r.extent) for r in args[1].region)[ng:]
    kr = a_shape[-len(a_atom) - 1]
    ctx.loops((*batch, mr, nr, kr))
    mn_axes, mk_axes, nk_axes = (
        (*range(nb), nb, nb + 1),
        (*range(nb), nb, nb + 2),
        (*range(nb), nb + 1, nb + 2),
    )
    d = ctx.region("D", args[0], (M // cta_group, N), ("tmem",), repeat_axes=mn_axes)
    a = ctx.region(
        "A", args[1], a_atom, ("tmem",) if a_tmem else ("shared", "shared.dyn"), repeat_axes=mk_axes
    )
    b = ctx.region("B", args[2], (N // cta_group, K), ("shared", "shared.dyn"), repeat_axes=nk_axes)
    if d.bits not in (16, 32):
        raise ValueError("MMA accumulators require 16-bit or 32-bit storage")
    a_format = str(scalar(params.pop("a_format", str(a.dtype))))
    b_format = str(scalar(params.pop("b_format", str(b.dtype))))
    da = _tmem_operand(
        ctx,
        d,
        tmem_mma_operand_layout(
            "D", (M // cta_group, N), str(d.dtype), M=M, cta_group=cta_group, ws=ws
        ),
        (M // cta_group, N),
    )
    if a_tmem:
        if am != "k":
            raise ValueError("TMEM A requires K major order")
        aa = _tmem_operand(
            ctx,
            a,
            tmem_mma_operand_layout("A", a_atom, str(a.dtype), M=M, cta_group=cta_group, ws=ws),
            a_atom,
        )
    else:
        aa = _shared_operand(ctx, a, a_atom, params.pop("a_descriptor"), am)
    ba = _shared_operand(ctx, b, (N // cta_group, K), params.pop("b_descriptor"), bm)
    neg_a, neg_b = (
        bool(scalar(params.pop("neg_a", False))),
        bool(scalar(params.pop("neg_b", False))),
    )
    common = dict(
        M=M,
        N=N,
        K=K,
        d_dtype=str(d.dtype),
        a_dtype=a_format,
        b_dtype=b_format,
        trans_a=am == "mn",
        trans_b=bm == "mn",
        cta_group=cta_group,
        neg_a=neg_a,
        neg_b=neg_b,
    )
    accum = Or(ctx.repeat[-1] != 0, T.cast(params.pop("accum", False), "bool"))
    pred = params.pop("pred", None)
    if block_scale:
        vec = int(modifiers["scale_vec"].split("::")[1][:-1])
        sf_per_mma = vec
        sfa, a_id, a_dtype = _scale_operand(
            ctx, args[3], "SFA", M // cta_group, sf_per_mma, mk_axes
        )
        sfb, b_id, b_dtype = _scale_operand(ctx, args[4], "SFB", N, sf_per_mma, nk_axes)
        if a_dtype != b_dtype:
            raise ValueError("SFA and SFB must have the same format")
        kind = _get_tcgen05_mma_kind(str(d.dtype), a_format, b_format, a_dtype, b_dtype)
        if vec != _get_tcgen05_mma_scale_vec_size(kind, a_dtype):
            raise ValueError("explicit scale vector disagrees with operand formats")
        descriptor = encode_instr_descriptor_block_scaled_uint32(**common, sf_dtype=a_dtype)
        idesc = (
            T.uint32(descriptor)
            | (T.cast(a_id, "uint32") << T.uint32(29))
            | (T.cast(b_id, "uint32") << T.uint32(4))
        )
        native = [da, aa, ba, idesc, sfa, sfb, accum]
    else:
        kind = _get_tcgen05_mma_kind(str(d.dtype), a_format, b_format)
        idesc = T.uint32(
            encode_instr_descriptor_dense_uint32(
                **common, sat_d=bool(scalar(params.pop("sat_d", False)))
            )
        )
        if ws:
            native = [da, aa, ba, idesc, accum, params.pop("zero_column_mask", 0)]
        else:
            masks = sequence(params.pop("disable_output_lane", (0,) * (4 * cta_group)))
            if len(masks) != 4 * cta_group:
                raise ValueError("disable_output_lane has four entries per CTA")
            native = [da, aa, ba, idesc, *(T.cast(x, "uint32") for x in masks), accum]
    if modifiers["kind"] != "kind::" + kind:
        raise ValueError("explicit MMA kind disagrees with operand formats")
    if params:
        raise ValueError(f"{opcode}: unknown operands {tuple(params)}")
    return ctx.finish([PTXNamespace()[opcode](*native, pred=pred)])
