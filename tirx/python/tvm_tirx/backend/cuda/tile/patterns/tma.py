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

The global operand is the descriptor's tensor view. Non-interleaved views have
the opcode rank. Interleaved views have axes (N, C_slices, *spatial, Ci), with
one implicit 16B/32B Ci slice. ``box`` and ``coords`` omit that implicit axis.
The shared operand is (*group, *repeat, *box_image). ``coord_steps`` has one
coordinate increment per group/repeat axis. Hardware fields are encoded
verbatim. No rank promotion, folding, source selection, or prefetch is inserted.

Parameters: box, coords (required); coord_steps=(), format=operand dtype,
element_strides=(1,)*rank, swizzle=0, interleave=0, l2_promotion=0,
oob_fill=0, pred=None. Loads require mbar, multicast requires cta_mask,
cache-hint forms require cache_policy, gather4/scatter4 requires gather4.
Coordinates and box use outer-to-inner order. See the instruction guide for
opaque TensorMap and packed/interleaved representations.
"""

from itertools import product
from math import prod

from tvm_tirx.backend.cuda.ptx import PTXNamespace
from tvm_tirx.ir import Range, TensorRegion, Tuple, Var
from tvm_tirx.runtime import DataType, MemoryScope
from tvm_tirx.script import tirx as T

from .common import Context, scalar
from .descriptors import swizzle_bytes


def sequence(value):
    return tuple(value.fields if isinstance(value, Tuple) else value)


def _encode_requirement(ctx, condition, message):
    # cuTensorMapEncodeTiled validates runtime descriptor parameters. Reject
    # statically invalid values here, but keep symbolic strides/dimensions for
    # that API rather than imposing a compile-time proof on runtime arguments.
    if ctx.analyzer.can_prove(T.Not(condition)):
        raise ValueError(message)


def _descriptor_fields(fields, rank, box, default_dtype):
    dtype = fields.pop("format", default_dtype)
    dtype = str(scalar(dtype))
    force = 11 if dtype == "tf32" else -1
    if force == 11:
        dtype = "float32"
    bits = DataType(dtype).bits
    element_strides = tuple(
        int(scalar(x)) for x in sequence(fields.pop("element_strides", (1,) * rank))
    )
    if len(element_strides) != rank or any(x < 1 or x > 8 for x in element_strides):
        raise ValueError("TMA element_strides must have rank entries in 1..8")
    swizzle = int(scalar(fields.pop("swizzle", 0)))
    interleave = int(scalar(fields.pop("interleave", 0)))
    l2 = int(scalar(fields.pop("l2_promotion", 0)))
    oob = int(scalar(fields.pop("oob_fill", 0)))
    if interleave not in (0, 1, 2):
        raise ValueError("TMA interleave must be 0, 1, or 2")
    if interleave and rank < 3:
        raise ValueError("TMA interleave requires rank 3..5")
    if interleave == 2 and swizzle != 1:
        raise ValueError("TMA interleave=2 requires swizzle=1")
    if not interleave and element_strides[-1] != 1:
        raise ValueError("TMA innermost element stride must be one without interleaving")
    if swizzle not in range(4) or l2 not in range(4) or oob not in range(2):
        raise ValueError("TMA requires swizzle/l2_promotion in 0..3 and oob_fill in 0..1")
    if not interleave and box[-1] * bits % 128:
        raise ValueError("TMA innermost box width must be a multiple of 16 bytes")
    if not interleave and swizzle and box[-1] * bits // 8 > 16 << swizzle:
        raise ValueError("TMA innermost box width must not exceed its swizzle span")
    if oob and dtype not in ("float16", "float32", "float64", "bfloat16"):
        raise ValueError("TMA NaN OOB fill requires a floating-point descriptor format")
    if dtype == "float4_e2m1fn" and (box[-1] != 128 or swizzle not in (0, 3)):
        raise ValueError("TMA packed FP4 requires inner box 128 and swizzle 0 or 3")
    return dtype, force, element_strides, swizzle, interleave, l2, oob


def _tensor_map(ctx, region, rank, box, encoding):
    if not isinstance(region, TensorRegion) or region.source.ty.scope != MemoryScope("global"):
        raise ValueError("TMA descriptor requires a global tensor region")
    tensor = region.source
    ty = tensor.ty
    bits = DataType(ty.dtype.dtype).bits
    packed = encoding[0] == "float4_e2m1fn"
    interleave = encoding[4]
    channels = (16 * interleave * 8 // bits) if interleave else 1
    tensor_rank = rank + bool(interleave)
    if len(region.region) != tensor_rank:
        raise ValueError("TMA global rank must equal instruction rank plus its implicit Ci axis")
    view_dims = tuple(r.extent for r in region.region)
    dims = view_dims[:rank]
    for size in dims:
        wide = T.cast(size, ty="int64")
        _encode_requirement(
            ctx, T.And(wide > 0, wide <= 2**32), "TMA dimensions must be in 1..2^32"
        )
    if packed:
        _encode_requirement(
            ctx, dims[-1] % 128 == 0, "TMA packed FP4 inner dimension must be a multiple of 128"
        )
    starts = tuple(r.min for r in region.region)
    if interleave:
        ctx.require(view_dims[-1] == channels, "TMA Ci extent must equal one interleave slice")
    # Tensor strides already specify a rectangular descriptor. A layout is
    # checked at the same coordinates, without folding or permuting its axes.
    layout = ty.layout
    if layout is None:
        strides = (
            list(ty.strides)
            if ty.strides
            else [prod(ty.shape[i + 1 :]) for i in range(tensor_rank)]
        )
        base = {"m": sum(i * s for i, s in zip(starts, strides))}
    else:
        base = layout.apply(*starts, shape=ty.shape)
    if set(base) - {"m"}:
        raise ValueError("TMA global view must use only the memory axis")
    if layout is not None:
        strides = []
    coordinates = []
    for i, size in enumerate(view_dims):
        unit = list(starts)
        unit[i] += 1
        if layout is not None:
            stride = ctx.analyzer.simplify(layout.apply(*unit, shape=ty.shape)["m"] - base["m"])
            # A unit dimension has no address movement; choose the smallest
            # legal descriptor stride for this unobserved coordinate.
            if ctx.analyzer.can_prove(size == 1):
                stride = 1 if i == tensor_rank - 1 else max(channels, 128 // bits)
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
    if interleave:
        strides = strides[:-1]
        ctx.require(strides[-1] == channels, "TMA W stride must equal one interleave slice")
    alignment = 32 if packed or interleave == 2 else 16
    ctx.require(
        (base["m"] + ty.elem_offset) * bits % (8 * alignment) == 0,
        f"TMA global origin requires {alignment}-byte alignment",
    )
    for stride in strides[:-1]:
        byte_stride = T.cast(stride, ty="int64") * bits // 8
        _encode_requirement(
            ctx,
            byte_stride % alignment == 0,
            f"TMA global byte strides must be multiples of {alignment}",
        )
        _encode_requirement(
            ctx,
            T.And(byte_stride > 0, byte_stride < 2**40),
            "TMA byte strides must be in 1..2^40-1",
        )
    dtype, force, element_strides, swizzle, interleave, l2, oob = encoding
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
        return cached
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
    return tensor_map


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
    if not isinstance(shared, TensorRegion):
        raise ValueError("TMA shared operand requires a tensor region")
    default_dtype = str(
        (global_ if isinstance(global_, TensorRegion) else shared).source.ty.dtype.dtype
    )
    encoding = _descriptor_fields(params, rank, box, default_dtype)
    dtype, _, element_strides, swizzle, interleave, _, _ = encoding
    bits = DataType(dtype).bits
    packed = dtype == "float4_e2m1fn"
    if packed and not load:
        raise ValueError("TMA 16U4_ALIGN16B supports global-to-shared loads only")
    if isinstance(global_, TensorRegion):
        if bits != DataType(global_.source.ty.dtype.dtype).bits:
            raise ValueError("TMA descriptor format must match the global storage element width")
        tensor_map = _tensor_map(ctx, global_, rank, box, encoding)
        tmap = T.address_of(tensor_map)
    else:
        # A caller-encoded descriptor is an ordinary instruction operand. The
        # caller may select/prefetch it explicitly; its fields must agree with
        # the box, format and swizzle supplied to this pattern.
        tmap = global_
    shared_shape = (
        (4, box[1]) if gather else tuple((b + s - 1) // s for b, s in zip(box, element_strides))
    )
    if interleave:
        channels = 16 * interleave * 8 // bits
        # Interleaved tiled accesses transfer one channel slice per issue.
        # C box/element-stride fields do not iterate slices. The caller uses
        # a repeat axis and coord_steps to visit multiple C coordinates.
        shared_shape = (shared_shape[0], 1, *shared_shape[2:], channels)
    smem = ctx.region("shared", shared, shared_shape, ("shared", "shared.dyn"))
    if smem.bits != bits:
        raise ValueError("TMA global and shared storage widths must agree")
    if swizzle not in range(4):
        raise ValueError("TMA swizzle must be 0, 1, 2, or 3")
    origin = (0,) * len(shared_shape)
    physical_bits = smem.memory(origin) * smem.bits
    base_bits = physical_bits
    pointer = smem.pointer(origin)
    if swizzle:
        period = 128 << swizzle
        if int(smem.ty.data_alignment) % period:
            raise ValueError("TMA swizzled base requires alignment to the full swizzle period")
        # A slice's first element can be displaced by the swizzle phase.
        # The instruction takes its unswizzled origin and applies that phase.
        base_bits = swizzle_bytes(physical_bits // 8, swizzle) * 8 + physical_bits % 8
        ctx.require(base_bits % 1024 == 0, "TMA atom origin must be 128-byte aligned")
        pointer = (
            T.cuda.cvta_generic_to_shared(pointer)
            - T.cast(physical_bits // 8, ty="uint32")
            + T.cast(base_bits // 8, ty="uint32")
        )
    else:
        smem.aligned(origin, 128)
    # Check the documented packed box image directly. Element strides select
    # global samples; the shared destination remains compact.
    for indices in product(*(range(size) for size in shared_shape)):
        flat = sum(var * prod(shared_shape[i + 1 :]) for i, var in enumerate(indices))
        # 16U4_ALIGN16B expands each sixteen packed nibbles into a 16-byte
        # shared block: eight payload bytes followed by eight undefined bytes.
        # The mbarrier transaction count is the payload bytes, excluding padding.
        relative_bits = (flat // 16 * 128 + flat % 16 * 4) if packed else flat * smem.bits
        absolute_bits = base_bits + relative_bits
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
    native_coords = [T.cast(c, ty="int32") for c in native_coords]
    if load:
        native = [pointer, tmap, *native_coords, params.pop("mbar")]
        if modifiers.get("multicast"):
            native.append(
                T.cast(
                    params.pop("cta_mask"),
                    ty="uint32" if entry.name.endswith("multicast32") else "uint16",
                )
            )
    else:
        native = [tmap, *native_coords, pointer]
    if modifiers.get("cache"):
        native.append(T.cast(params.pop("cache_policy"), ty="uint64"))
    pred = params.pop("pred", None)
    if params:
        raise ValueError(f"{opcode}: unknown fields {tuple(params)}")
    return ctx.finish([PTXNamespace()[opcode](*native, pred=pred)])
