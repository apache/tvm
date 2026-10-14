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

"""tcgen05.cp: (*group, *repeat, row, b32_word).

The instruction spells the shape and multicast. Shared descriptor fields are
extracted from the canonical view. Destination replicas describe the hardware's
multicast copies; no shape selection, layout permutation, or synchronization
is performed here.

The optional b8x16.b4x16_p64 / b8x16.b6x16_p32 qualifiers keep this b32
footprint: each 16-byte source block contains sixteen packed values followed
by padding, and the destination contains sixteen expanded bytes. Hardware
performs the conversion; the pattern does not insert scalar unpacking code.

Parameters: dst, src, scope; pred=None. Shared descriptor byte strides and
swizzle are extracted from the operand image. One repeat emits one cp.
"""

from tvm.backend.cuda.op import _tmem_address
from tvm.backend.cuda.ptx import PTXNamespace
from tvm.script import tirx as T

from .common import Context
from .descriptors import matrix_descriptor


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2:
        raise ValueError("tcgen05.cp requires dst and src")
    shape = modifiers["shape"]
    rows, bits = shape.split("x")
    rows, words = int(rows), int(bits[:-1]) // 32
    pred = params.pop("pred", None)
    if params:
        raise ValueError(f"{opcode}: unknown operands {tuple(params)}")
    ctx = Context(sctx)
    dst = ctx.region("dst", args[0], (rows, words), ("tmem",))
    src = ctx.region("src", args[1], (rows, words), ("shared", "shared.dyn"))
    dst.require_bits(32)
    src.require_bits(32)
    multicast = modifiers.get("multicast", "")
    expected_replica = {
        "": [],
        "warpx4": [(4, 32)],
        "warpx2::02_13": [(2, 64)],
        "warpx2::01_23": [(2, 32)],
    }[multicast]
    actual_replica = [
        (int(it.extent), int(it.stride)) for it in dst.layout.replica if it.axis.name == "TLane"
    ]
    if actual_replica != expected_replica or any(
        it.axis.name != "TLane" for it in dst.layout.replica
    ):
        raise ValueError("TMEM replicas must match the explicit multicast qualifier")
    base = dst.physical((0, 0))
    if set(base) - {"TLane", "TCol"}:
        raise ValueError("tcgen05.cp destination requires TLane/TCol coordinates")
    row_base, col_base = base.get("TLane", 0), base.get("TCol", 0)
    ctx.require(col_base % 4 == 0, "tcgen05.cp column start must be 128-bit aligned")
    last_row = 96 if rows == 4 else 95 if multicast == "warpx2::01_23" else rows - 1
    replica_offset = sum((extent - 1) * stride for extent, stride in expected_replica)
    ctx.require(row_base >= 0, "tcgen05.cp row start must be nonnegative")
    ctx.require(
        row_base + last_row + replica_offset < 128,
        "tcgen05.cp footprint exceeds the 128-lane TMEM space",
    )
    ctx.require(col_base >= 0, "tcgen05.cp column start must be nonnegative")
    ctx.require(col_base + words <= 512, "tcgen05.cp footprint exceeds 512 TMEM columns")
    for row in range(rows):
        lane = row * 32 if rows == 4 else row
        if multicast == "warpx2::01_23":
            lane = row // 32 * 64 + row % 32
        for word in range(words):
            p = dst.physical((row, word))
            ctx.require(
                p.get("TLane", 0) == row_base + lane,
                "TMEM row map disagrees with cp shape/multicast",
            )
            ctx.require(
                p.get("TCol", 0) == col_base + word, "TMEM words must occupy consecutive columns"
            )
    desc = matrix_descriptor(ctx, src, (rows, words), copy=True)
    address = T.cuda.get_tmem_addr(_tmem_address(dst.tensor), row_base, col_base)
    return ctx.finish([PTXNamespace()[opcode](address, desc, pred=pred)])
