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

"""Register instructions: each operand ends in its declared PTX register group.

For example cvt.rn.f16x2.f32 has dst (...,1), a (...,1), b (...,1).
Packed arithmetic takes its native packed register representation. Packing and
unpacking are explicit mov instructions, rather than opportunistic vectorization.
"""

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.backend.cuda.ptx.table import operand_layout
from tvm.ir import TensorRegion

from .common import Context

MNEMONICS = {
    "mov",
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "fma",
    "cvt",
    "sqrt",
    "ex2",
    "lg2",
    "rcp",
    "abs",
    "neg",
}


def lower(opcode, forms, args, params, sctx):
    candidates = []
    for entry, modifiers in forms:
        slots = [(slot, n) for slot, _, n in operand_layout(entry, modifiers) if n]
        if len(slots) != len(args):
            continue
        if all(
            (not isinstance(arg, TensorRegion) and slot.rw == "r" and n == 1)
            or (
                isinstance(arg, TensorRegion)
                and len(arg.region) > 0
                and int(arg.region[-1].extent) == n
            )
            for arg, (slot, n) in zip(args, slots)
        ):
            candidates.append(slots)
    if len(candidates) != 1:
        raise ValueError(
            f"{opcode}: operands must end in the instruction's explicit register groups"
        )
    ctx = Context(sctx)
    native_args = []
    for arg, (slot, n) in zip(args, candidates[0]):
        if not isinstance(arg, TensorRegion):
            native_args.append(arg)
            continue
        reg = ctx.region(slot.name, arg, (n,), ("local",))
        if slot.rw != "r":
            reg.nonoverlapping()
        for word in range(n):
            reg.owner((word,))
            native_args.append(reg.value((word,)))
    pred = params.pop("pred", None)
    preserve_dst = bool(params.pop("preserve_dst", False))
    if params:
        raise ValueError(f"{opcode}: unknown operands {tuple(params)}")
    return ctx.finish([PTXNamespace()[opcode](*native_args, pred=pred, preserve_dst=preserve_dst)])
