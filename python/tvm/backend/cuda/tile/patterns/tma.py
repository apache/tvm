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

"""TMA with an explicit descriptor box and explicit coordinate steps.

The global operand is the descriptor's tensor view (rank equals opcode rank).
The shared operand is (*group, *repeat, *box). ``box`` and ``coords`` use
outer-to-inner tensor order; ``coord_steps`` has one such coordinate increment
per group/repeat axis. Hardware descriptor fields are encoded verbatim. No
rank promotion, dimension folding, source selection, or prefetch is inserted.
"""

from itertools import product
from math import prod

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Range, TensorRegion, Tuple, Var
from tvm.runtime import DataType
from tvm.script import tirx as T

from .common import Context, scalar
from .descriptors import swizzle_bytes


def sequence(value):
    return tuple(value.fields if isinstance(value, Tuple) else value)


def _tensor_map(ctx, region, rank, box, fields):
    if not isinstance(region, TensorRegion) or region.source.ty.storage_scope != "global":
        raise ValueError("TMA descriptor requires a global tensor region")
    tensor = region.source
    ty = tensor.ty
    if len(region.region) != rank:
        raise ValueError("TMA global tensor rank must equal the instruction rank")
    bits = DataType(ty.dtype.dtype).bits
    dims = tuple(r.extent for r in region.region)
    starts = tuple(r.min for r in region.region)
    # Tensor strides already specify a rectangular descriptor. A layout is
    # checked at the same coordinates, without folding or permuting its axes.
    layout = ty.layout
    if layout is None:
        strides = list(ty.strides) if ty.strides else [prod(ty.shape[i + 1 :]) for i in range(rank)]
        base = {"m": sum(i * s for i, s in zip(starts, strides))}
    else:
        base = layout.apply(*starts, shape=ty.shape)
    if set(base) - {"m"}:
        raise ValueError("TMA global view must use only the memory axis")
    if layout is not None:
        strides = []
    coordinates = []
    for i, size in enumerate(dims):
        unit = list(starts)
        unit[i] += 1
        if layout is not None:
            stride = ctx.analyzer.simplify(layout.apply(*unit, shape=ty.shape)["m"] - base["m"])
            # A unit dimension has no address movement; choose the smallest
            # legal descriptor stride for this unobserved coordinate.
            if ctx.analyzer.can_prove(size == 1):
                stride = 1 if i == rank - 1 else max(1, 128 // bits)
            strides.append(stride)
        var = Var(f"descriptor_dim_{i}", "int32")
        ctx.analyzer.bind(var, Range(0, size))
        coordinates.append(var)
    if layout is not None:
        actual = layout.apply(*(a + b for a, b in zip(starts, coordinates)), shape=ty.shape)["m"]
        ctx.require(
            actual == base["m"] + sum(c * s for c, s in zip(coordinates, strides)),
            "TMA global view must be rectangular with fixed strides; "
            "reorganize its axes explicitly",
        )
    ctx.require(strides[-1] == 1, "TMA innermost global stride must be one")
    ctx.require(
        (base["m"] + ty.elem_offset) * bits % 128 == 0,
        "TMA global origin requires 16-byte alignment",
    )
    for stride in strides[:-1]:
        ctx.require(stride * bits % 128 == 0, "TMA global byte strides must be multiples of 16")
    dtype = fields.pop("format", str(ty.dtype.dtype))
    dtype = str(scalar(dtype))
    force = 11 if dtype == "tf32" else -1
    if force == 11:
        dtype = "float32"
    if DataType(dtype).bits != bits:
        raise ValueError("TMA descriptor format must match the global storage element width")
    element_strides = tuple(
        int(scalar(x)) for x in sequence(fields.pop("element_strides", (1,) * rank))
    )
    if len(element_strides) != rank or any(x < 1 or x > 8 for x in element_strides):
        raise ValueError("TMA element_strides must have rank entries in 1..8")
    if element_strides != (1,) * rank:
        raise ValueError("the dense TMA tile pattern requires unit element_strides")
    swizzle = int(scalar(fields.pop("swizzle", 0)))
    interleave = int(scalar(fields.pop("interleave", 0)))
    l2 = int(scalar(fields.pop("l2_promotion", 0)))
    oob = int(scalar(fields.pop("oob_fill", 0)))
    if interleave != 0:
        raise ValueError("the dense TMA pattern requires interleave=0")
    key = "explicit_tma:" + ":".join(
        map(
            str,
            (
                hash(tensor),
                starts,
                dims,
                strides,
                dtype,
                force,
                box,
                element_strides,
                swizzle,
                interleave,
                l2,
                oob,
            ),
        )
    )
    cached = ctx.sctx.cache_get(key)
    if cached is not None:
        return cached, swizzle, element_strides
    tensor_map = T.Var("tile_tensormap", ty=T.handle("tensormap").ty)

    @T.function(check_well_formed=False)
    def encode():
        T.bind(T.stack_alloca("tensormap", 1), var=tensor_map)
        T.cuda.tensormap_encode_tiled(
            tensor_map,
            tensor.ptr_to(starts),
            *reversed(dims),
            *(s * bits // 8 for s in reversed(strides[:-1])),
            *reversed(box),
            *reversed(element_strides),
            descriptor_dtype=dtype,
            rank=rank,
            interleave=interleave,
            swizzle=swizzle,
            l2_promotion=l2,
            oob_fill=oob,
            force_cu_dtype=force,
        )
        T.kernel_replace_point()

    ctx.sctx.add_init_stmt(encode.body, host=True)
    ctx.sctx.cache_set(key, tensor_map)
    return tensor_map, swizzle, element_strides


def lower(opcode, entry, modifiers, args, params, sctx):
    if len(args) != 2:
        raise ValueError("TMA requires dst and src")
    if entry.name not in (
        "cp_async_bulk_tensor_g2s_cta",
        "cp_async_bulk_tensor_g2s_cluster",
        "cp_async_bulk_tensor_s2g",
        "cp_reduce_async_bulk_tensor",
        "cp_async_bulk_tensor_g2s_cluster_multicast16",
        "cp_async_bulk_tensor_g2s_cluster_multicast32",
    ):
        raise ValueError(
            "TMA tile pattern requires tile/gather4/scatter4 mode without descriptor overrides"
        )
    load = modifiers["src"] == "global"
    shared, global_ = (args[0], args[1]) if load else (args[1], args[0])
    rank = int(modifiers["dim"][0])
    box = tuple(int(scalar(x)) for x in sequence(params.pop("box")))
    if len(box) != rank or any(x < 1 or x > 256 for x in box):
        raise ValueError("TMA box must have rank extents in 1..256")
    gather = modifiers.get("load_mode") in ("tile::gather4", "tile::scatter4")
    if gather and (rank != 2 or box[0] != 1):
        raise ValueError("gather4/scatter4 requires a 2D descriptor box with one row")
    ctx = Context(sctx)
    tensor_map, swizzle, element_strides = _tensor_map(ctx, global_, rank, box, params)
    shared_shape = (4, box[1]) if gather else box
    smem = ctx.region("shared", shared, shared_shape, ("shared", "shared.dyn"))
    if smem.bits != DataType(global_.source.ty.dtype.dtype).bits:
        raise ValueError("TMA global and shared storage widths must agree")
    if swizzle not in range(4):
        raise ValueError("TMA swizzle must be 0, 1, 2, or 3")
    origin = (0,) * rank
    smem.aligned(origin, 128)
    base = smem.memory(origin)
    # Check the documented packed box image directly. Element strides select
    # global samples; the shared destination remains compact.
    for indices in product(*(range(size) for size in shared_shape)):
        flat = sum(var * prod(shared_shape[i + 1 :]) for i, var in enumerate(indices))
        absolute_bits = (base + flat) * smem.bits
        expected_bits = (
            (swizzle_bytes(absolute_bits // 8, swizzle) * 8 + absolute_bits % 8)
            if swizzle
            else absolute_bits
        )
        ctx.require(
            smem.memory(indices) * smem.bits == expected_bits,
            "shared view must match the descriptor's compact/swizzled box",
        )
    coords = list(sequence(params.pop("coords")))
    if len(coords) != rank:
        raise ValueError("TMA coords must have one entry per descriptor dimension")
    steps = sequence(params.pop("coord_steps", ()))
    axes = [*ctx.group, *ctx.repeat]
    if len(steps) != len(axes):
        raise ValueError("coord_steps requires one coordinate vector per group/repeat axis")
    for var, step in zip(axes, steps):
        step = sequence(step)
        if len(step) != rank:
            raise ValueError("each coordinate step must have descriptor rank")
        coords = [c + var * d for c, d in zip(coords, step)]
    native_coords = list(reversed(coords))
    if gather:
        rows = sequence(params.pop("gather4"))
        if len(rows) != 4:
            raise ValueError("gather4 requires four explicit row coordinates")
        native_coords = [coords[-1], *rows]
    native_coords = [T.cast(c, "int32") for c in native_coords]
    tmap = T.address_of(tensor_map)
    pointer = smem.pointer(origin)
    if load:
        native = [pointer, tmap, *native_coords, params.pop("mbar")]
        if modifiers.get("multicast"):
            native.append(
                T.cast(
                    params.pop("cta_mask"),
                    "uint32" if entry.name.endswith("multicast32") else "uint16",
                )
            )
    else:
        native = [tmap, *native_coords, pointer]
    if modifiers.get("cache"):
        native.append(T.cast(params.pop("cache_policy"), "uint64"))
    pred = params.pop("pred", None)
    if params:
        raise ValueError(f"{opcode}: unknown fields {tuple(params)}")
    return ctx.finish([PTXNamespace()[opcode](*native, pred=pred)])
