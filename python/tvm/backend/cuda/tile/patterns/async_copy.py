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

"""Explicit contiguous chunks for cp.async and cross-CTA cp.async.bulk.

Coordinates: (*group, *repeat, element), with the last axis exactly chunk_bytes
of storage. Each repeat issues one requested copy. No commit, wait, transaction
accounting, or thread election is inserted. Bulk DSMEM takes local shared
addresses and maps destination and completion barrier to remote_cta_id once.
"""

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Bind, Evaluate, Var
from tvm.runtime import DataType
from tvm.script import tirx as T

from .common import Context, scalar


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2:
        raise ValueError("async copy requires dst and src")
    bulk = entry.name == "cp_async_bulk_s2c"
    size = int(scalar(params.pop("chunk_bytes")))
    if bulk:
        if size <= 0 or size % 16:
            raise ValueError("bulk chunk_bytes must be a positive multiple of 16")
    elif size not in (4, 8, 16):
        raise ValueError("cp.async chunk_bytes must be 4, 8, or 16")
    ctx = Context(sctx)
    operands = []
    for name, region, scopes in (
        ("dst", args[0], ("shared", "shared.dyn")),
        ("src", args[1], ("shared", "shared.dyn") if bulk else ("global",)),
    ):
        bits = DataType(region.source.ty.dtype.dtype).bits
        if size * 8 % bits:
            raise ValueError(f"{name}: chunk_bytes must contain whole storage elements")
        elements = size * 8 // bits
        operand = ctx.region(name, region, (elements,), scopes)
        operand.contiguous(((i,) for i in range(elements)), 16 if bulk else size)
        operands.append(operand)
    dst, src = operands
    instruction = PTXNamespace()[opcode]
    pred = params.pop("pred", None)
    if bulk:
        barrier, remote = params.pop("mbar"), params.pop("remote_cta_id")
        mapped = Var("remote_addresses", T.Tensor((2,), "uint64", scope="local", layout="default"))
        ctx.bindings.append(Bind(mapped, T.alloc_tensor(ty_args=[mapped.ty])))
        ctx.bindings.append(Evaluate(T.ptx.mapa.u64(mapped[0], barrier, T.cast(remote, "uint32"))))
        body = [
            T.ptx.mapa.u64(mapped[1], dst.pointer((0,)), T.cast(remote, "uint32")),
            instruction(
                T.cast(mapped[1], "uint32"),
                src.pointer((0,)),
                T.uint32(size),
                T.cast(mapped[0], "uint32"),
                pred=pred,
            ),
        ]
    else:
        native_args = [dst.pointer((0,)), src.pointer((0,)), size]
        src_size = params.pop("src_size", None)
        ignore_src = params.pop("ignore_src", None)
        if src_size is not None and ignore_src is not None:
            raise ValueError("cp.async takes either src_size or ignore_src")
        if src_size is not None:
            native_args.append(src_size)
        if ignore_src is not None:
            native_args.append(T.ptx.pred(ignore_src))
        if modifiers.get("cache"):
            native_args.append(params.pop("cache_policy"))
        body = [instruction(*native_args, pred=pred)]
    if params:
        raise ValueError(f"{opcode}: unknown operands {tuple(params)}")
    return ctx.finish(body)
