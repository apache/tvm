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

"""m8n8 ldmatrix/stmatrix: (*group, *repeat, matrix, row, word).

Each matrix has eight independent 16B-aligned shared rows of four b32 words.
Registers belong to lane 4*row+word; each lane supplies/receives x1/x2/x4
independent b32 registers. For .trans, the register image is the transpose of
the shared b16 matrix (each word packs two adjacent values of that image).

Address providers are lanes matrix*8+row. For x1/x2, unused providers repeat
valid row addresses. Address providers and result owners are separate maps.

Parameters: dst, src, scope; no instruction keywords. x1/x2/x4 and trans
are opcode qualifiers. Storage strides count b32 words, not b16 halves.
"""

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.script import tirx as T

from .common import Context


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2 or params:
        raise ValueError("ldmatrix/stmatrix requires dst and src without extra operands")
    if modifiers["shape"] != "m8n8" or modifiers["type"] != "b16":
        raise ValueError("matrix tile pattern requires m8n8.b16")
    count = int(modifiers["num"][1:])
    load = entry.ptx_name == "ldmatrix"
    ctx = Context(sctx, "warp")
    shape = (count, 8, 4)
    dst = ctx.region("dst", args[0], shape, ("local",) if load else ("shared", "shared.dyn"))
    src = ctx.region("src", args[1], shape, ("shared", "shared.dyn") if load else ("local",))
    smem, regs = (src, dst) if load else (dst, src)
    smem.require_bits(32)
    regs.require_bits(32)
    for matrix in range(count):
        for row in range(8):
            smem.contiguous(((matrix, row, word) for word in range(4)), 16)
            for word in range(4):
                regs.owner((matrix, row, word), lane=4 * row + word)
    if load:
        regs.nonoverlapping()
    lane = T.cuda.lane_id()
    words = [regs.value((matrix, lane // 4, lane % 4)) for matrix in range(count)]
    ptr = smem.pointer(((lane // 8) % count, lane % 8, 0))
    native_args = [*words, ptr] if load else [ptr, *words]
    return ctx.finish([PTXNamespace()[opcode]._make_call(*native_args)])
