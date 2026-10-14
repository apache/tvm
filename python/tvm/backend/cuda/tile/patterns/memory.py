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

"""ld/st: (*group, *repeat, word), one native memory instruction per repeat.

The memory words are contiguous and naturally vector-aligned. Register words
can occupy arbitrary distinct slots of the issuing thread. Width, memory space,
and cache policy are selected by the written PTX, never by the tensor layout.

Parameters: dst, src, scope; pred=None, preserve_dst=False, and cache_policy
(required only when the opcode has a cache-hint operand). Atom strides count
scalar storage elements; vector alignment counts bytes.
"""

from tvm.backend.cuda.ptx import PTXNamespace

from .common import Context


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2:
        raise ValueError("ld/st requires dst and src tensor operands")
    load = entry.ptx_name == "ld"
    count = int((modifiers.get("vec") or "v1")[1:])
    bits = int(modifiers["type"][1:])
    space = modifiers.get("space")
    if space not in ("global", "shared", "shared::cta", "local"):
        raise ValueError(
            "ld/st tile requires an explicit global, shared::cta, shared, or local space"
        )
    ctx = Context(sctx)
    scopes = ("shared", "shared.dyn") if space.startswith("shared") else (space,)
    dst = ctx.region("dst", args[0], (count,), ("local",) if load else scopes)
    src = ctx.region("src", args[1], (count,), scopes if load else ("local",))
    memory, regs = (src, dst) if load else (dst, src)
    memory.require_bits(bits)
    regs.require_bits(bits)
    memory.contiguous(((i,) for i in range(count)), max(1, count * bits // 8))
    for i in range(count):
        regs.owner((i,))
    if load:
        regs.nonoverlapping()
    native_args = [regs.value((i,)) for i in range(count)]
    pointer = memory.pointer((0,))
    native_args = [*native_args, pointer] if load else [pointer, *native_args]
    if modifiers.get("cache"):
        native_args.append(params.pop("cache_policy"))
    pred = params.pop("pred", None)
    preserve_dst = bool(params.pop("preserve_dst", False))
    if params:
        raise ValueError(f"{opcode}: unknown operands {tuple(params)}")
    return ctx.finish(
        [PTXNamespace()[opcode]._make_call(*native_args, pred=pred, preserve_dst=preserve_dst)]
    )
