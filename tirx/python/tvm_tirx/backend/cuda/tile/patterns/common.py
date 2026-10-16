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

"""Canonical coordinates, physical operands, and explicit group/repeat loops.

Pattern modules describe hardware formulas. This module only evaluates the
caller's view at those coordinates: it never factors, permutes, or searches a
layout, and never chooses an instruction width or a data representation.
"""

from itertools import product
from math import prod

import tvm_ffi

from tvm_tirx.ir import (
    Bind,
    Call,
    Evaluate,
    For,
    ForKind,
    IntImm,
    Op,
    Range,
    SeqStmt,
    Stmt,
    TensorRegion,
    Var,
)
from tvm_tirx.ir.prim import Let
from tvm_tirx.runtime import DataType, MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.tirx.layout import TileLayout


def scalar(value):
    """Read a constant field stored as an ordinary IR operand."""
    return value.value if hasattr(value, "value") else value


class Context:
    """One explicit instruction repeated independently by each execution group."""

    def __init__(self, sctx, atom_scope="thread"):
        self.sctx = sctx
        self.analyzer = Analyzer()
        self.static_ranges = {}
        for var, extent in sctx.launch_params.values():
            self.analyzer.bind(var, Range(0, extent))
        for var, bound in sctx.var_range_map.items():
            self.analyzer.bind(
                var, Range.from_min_extent(self.index_coordinates(bound.min), bound.extent)
            )
            low, extent = (self.analyzer.simplify(x) for x in (bound.min, bound.extent))
            if isinstance(low, IntImm) and isinstance(extent, IntImm):
                self.static_ranges[var] = range(int(low), int(low) + int(extent))
        self.bindings = []
        self.group = []
        self.group_shape = []
        self.owners = {}
        intra = sctx.intra
        if atom_scope == "warp":
            active = intra.get("laneid")
            if (
                sctx.scope_kind == "thread"
                or active is None
                or int(active[0]) != 32
                or int(active[1]) != 0
            ):
                raise ValueError("instruction requires all 32 lanes of a warp to be active")
        if sctx.scope_kind == "cluster" and "clusterCtaIdx.x" not in sctx.launch_params:
            raise ValueError("cluster tile scope requires an explicit LaunchConfig.cluster")
        axes = (
            ("cta_id", lambda: T.cuda.cluster_cta_id("x")),
            ("cbz", lambda: T.cuda.cluster_cta_id("z")),
            ("cby", lambda: T.cuda.cluster_cta_id("y")),
            ("cbx", lambda: T.cuda.cluster_cta_id("x")),
            ("warpid", T.cuda.warp_id),
            ("wid_in_wg", T.cuda.warp_in_warpgroup),
            ("laneid", T.cuda.lane_id),
        )
        if set(intra) - {name for name, _ in axes}:
            raise ValueError("tile groups require block/cluster CTA coordinates and warp/lane axes")
        for name, index in axes:
            if name not in intra or (name == "laneid" and atom_scope == "warp"):
                continue
            extent, offset = (int(x) for x in intra[name])
            var = Var("group_" + name, "int32")
            self.analyzer.bind(var, Range(0, extent))
            self.group.append(var)
            self.group_shape.append(int(extent))
            self.owners["cbx" if name == "cta_id" else name] = var + offset
            # A singleton group coordinate is known without a CUDA index
            # read, including the sole warp in a block smaller than 32 lanes.
            coordinate = T.int32(0) if extent == 1 else T.cast(index(), ty="int32") - offset
            self.bindings.append(Bind(var, coordinate))
        self.repeat = None
        self.repeat_shape = None

    def index_coordinates(self, expr):
        """Express CUDA index calls in the launch variables used by the prover."""

        def axis(prefix, dim, extent=False):
            binding = self.sctx.launch_params.get(prefix + dim)
            return binding[int(extent)] if binding is not None else T.int32(int(extent))

        def replace(call):
            if not isinstance(call.op, Op):
                return call
            name = call.op.name
            for operation, prefix in (("thread_idx", "threadIdx."), ("block_idx", "blockIdx.")):
                if name == "tirx.cuda." + operation:
                    return axis(prefix, str(call.args[0].value))
            if name in (
                "tirx.cuda.linear_thread_id",
                "tirx.cuda.lane_id",
                "tirx.cuda.warp_id",
                "tirx.cuda.thread_in_warpgroup",
                "tirx.cuda.warp_in_warpgroup",
                "tirx.cuda.warpgroup_id",
            ):
                thread = axis("threadIdx.", "x") + axis("threadIdx.", "x", True) * (
                    axis("threadIdx.", "y")
                    + axis("threadIdx.", "y", True) * axis("threadIdx.", "z")
                )
                return {
                    "tirx.cuda.linear_thread_id": thread,
                    "tirx.cuda.lane_id": thread % 32,
                    "tirx.cuda.warp_id": thread // 32,
                    "tirx.cuda.thread_in_warpgroup": thread % 128,
                    "tirx.cuda.warp_in_warpgroup": thread // 32 % 4,
                    "tirx.cuda.warpgroup_id": thread // 128,
                }[name]
            return call

        return tvm_ffi.structural_map(expr, (Call, replace), order="post")

    def require(self, condition, message):
        condition = self.index_coordinates(condition)
        # Layout.apply uses expression-local lets to share its address terms.
        # Inline these pure terms before collecting free pattern coordinates.
        condition = tvm_ffi.structural_map(
            condition,
            (
                Let,
                lambda expr: tvm_ffi.structural_map(
                    expr.body,
                    (Var, lambda var: expr.value if var.same_as(expr.var) else var),
                    order="post",
                ),
            ),
            order="post",
        )
        if self.analyzer.can_prove(condition):
            return
        # Bitwise swizzles are not fully covered by the arithmetic analyzer.
        # Check the already fixed group/repeat and enclosing-loop coordinates
        # directly. This does not search for another layout, atom, or instruction.
        variables = []

        def collect(var):
            if var not in variables:
                variables.append(var)
            return var

        tvm_ffi.structural_map(condition, (Var, collect))
        domain = dict(self.static_ranges)
        domain.update(
            zip(
                [*self.group, *(self.repeat or ())],
                (range(n) for n in [*self.group_shape, *(self.repeat_shape or ())]),
            )
        )
        if variables and all(var in domain for var in variables):
            for values in product(*(domain[var] for var in variables)):
                replacements = dict(zip(variables, values))
                expr = tvm_ffi.structural_map(
                    condition,
                    (Var, lambda var: IntImm(var.ty.dtype, replacements[var])),
                    order="pre",
                )
                if not self.analyzer.can_prove(expr):
                    break
            else:
                return
        raise ValueError(f"{message}: cannot prove {self.analyzer.simplify(condition)}")

    def loops(self, shape):
        self.repeat_shape = tuple(shape)
        self.repeat = [Var(f"repeat_{i}", "int32") for i in range(len(shape))]
        for var, size in zip(self.repeat, shape):
            self.analyzer.bind(var, Range(0, size))
        return self.repeat

    def region(self, name, region, atom_shape, scopes, *, repeat_axes=None, bounds=True):
        return Region(self, name, region, atom_shape, scopes, repeat_axes, bounds)

    def finish(self, statements) -> Stmt:
        body = SeqStmt(
            [
                Evaluate(s) if not isinstance(s, Bind | SeqStmt) and hasattr(s, "ty") else s
                for s in statements
            ]
        )
        for var, extent in reversed(list(zip(self.repeat or (), self.repeat_shape or ()))):
            body = For(var, 0, extent, ForKind.UNROLLED, body)
        return SeqStmt([*self.bindings, body])


class Region:
    """An operand's group prefix, repeat axes, and its own fixed atom suffix."""

    def __init__(self, ctx, name, region, atom_shape, scopes, repeat_axes, bounds):
        if not isinstance(region, TensorRegion):
            raise TypeError(f"{name}: expected a tensor region")
        self.ctx, self.name, self.region = ctx, name, region
        self.tensor, self.ty = region.source, region.source.ty
        self.dtype = DataType(self.ty.dtype.dtype)
        self.bits = self.dtype.bits
        self.layout = self.ty.effective_layout()
        if self.layout is None:
            raise ValueError(f"{name}: operand requires a physical layout")
        if self.ty.scope not in tuple(MemoryScope(scope) for scope in scopes):
            raise ValueError(f"{name}: expected storage scope {scopes}, got {self.ty.scope}")
        shape = tuple(int(ctx.analyzer.simplify(r.extent)) for r in region.region)
        if any(n <= 0 for n in shape):
            raise ValueError(f"{name}: pattern requires positive static region extents")
        ng, na = len(ctx.group_shape), len(atom_shape)
        if shape[:ng] != tuple(ctx.group_shape):
            raise ValueError(
                f"{name}: expected leading group shape {tuple(ctx.group_shape)}, got {shape}"
            )
        if len(shape) < ng + na or (na and shape[-na:] != tuple(atom_shape)):
            raise ValueError(
                f"{name}: expected trailing atom shape {tuple(atom_shape)}, got {shape}"
            )
        self.atom_shape = tuple(atom_shape)
        repeats = shape[ng : len(shape) - na if na else len(shape)]
        if ctx.repeat is None:
            ctx.loops(repeats)
        self.repeat_axes = (
            tuple(range(len(ctx.repeat))) if repeat_axes is None else tuple(repeat_axes)
        )
        expected = tuple(ctx.repeat_shape[i] for i in self.repeat_axes)
        if repeats != expected:
            raise ValueError(f"Repeat shapes must agree: {name}={repeats}, expected {expected}")
        if bounds:
            for r, size in zip(region.region, self.ty.shape):
                ctx.require(r.min >= 0, f"{name}: cannot prove region start is in bounds")
                ctx.require(r.min + r.extent <= size, f"{name}: region exceeds tensor shape")
        self.raw = None

    def indices(self, atom, repeat=None):
        repeats = [self.ctx.repeat[i] for i in self.repeat_axes] if repeat is None else repeat
        coords = [*self.ctx.group, *repeats, *atom]
        if len(coords) != len(self.region.region):
            raise ValueError(f"{self.name}: incorrect canonical coordinate rank")
        return tuple(
            self.ctx.analyzer.simplify(r.min + i) for r, i in zip(self.region.region, coords)
        )

    def physical(self, atom, repeat=None):
        result = dict(self.layout.apply(*self.indices(atom, repeat), shape=self.ty.shape))
        if self.ty.scope != MemoryScope("tmem"):
            result["m"] = result.get("m", 0) + self.ty.elem_offset
        return {key: self.ctx.analyzer.simplify(value) for key, value in result.items()}

    def memory(self, atom):
        physical = self.physical(atom)
        if set(physical) - {"m"}:
            raise ValueError(f"{self.name}: memory operand has thread or TMEM axes")
        return physical["m"]

    def owner(self, atom, *, lane=None):
        physical = self.physical(atom)
        owners = dict(self.ctx.owners)
        if lane is not None:
            owners["laneid"] = lane
        # These are equivalent CUDA ownership coordinates, not alternative
        # operand layouts. Evaluate their fixed radix relation at the caller's
        # coordinates; never rearrange the tile to find a matching owner.
        for linear, warp_axis in (("tx", "warpid"), ("tid_in_wg", "wid_in_wg")):
            if linear in physical:
                value = physical.pop(linear)
                physical[warp_axis] = physical.get(warp_axis, 0) + value // 32
                physical["laneid"] = physical.get("laneid", 0) + value % 32
        if "warpid" in owners and "wid_in_wg" in physical:
            physical["warpid"] = (
                physical.get("warpid", 0) + physical.pop("wid_in_wg") + 4 * physical.pop("wgid", 0)
            )
        elif "wid_in_wg" in owners and "warpid" in physical:
            warp = physical.pop("warpid")
            physical["wid_in_wg"] = physical.get("wid_in_wg", 0) + warp % 4
            physical["wgid"] = physical.get("wgid", 0) + warp // 4
        for axis in tuple(set(physical) - {"m", *owners}):
            if self.ctx.analyzer.can_prove(physical[axis] == 0):
                del physical[axis]
        if set(physical) - {"m", *owners}:
            raise ValueError(f"{self.name}: unexpected ownership axes {tuple(physical)}")
        for axis, owner in owners.items():
            self.ctx.require(
                physical.get(axis, 0) == owner,
                f"{self.name}: incorrect {axis} owner; expected {owner}",
            )
        return physical["m"]

    def aligned(self, atom, byte_alignment):
        alignment = int(self.ty.data_alignment)
        if alignment < byte_alignment or alignment % byte_alignment:
            raise ValueError(f"{self.name}: requires a known {byte_alignment}-byte-aligned base")
        self.ctx.require(
            (self.memory(atom) * self.bits) % (8 * byte_alignment) == 0,
            f"{self.name}: atom address must be {byte_alignment}-byte aligned",
        )

    def contiguous(self, coordinates, byte_alignment):
        coordinates = list(coordinates)
        self.aligned(coordinates[0], byte_alignment)
        base = self.memory(coordinates[0])
        for offset, coord in enumerate(coordinates):
            self.ctx.require(
                self.memory(coord) == base + offset,
                f"{self.name}: atom needs consecutive elements in canonical order",
            )

    def nonoverlapping(self):
        if isinstance(self.layout, TileLayout):
            span = 1
            storage = sorted(
                (int(i.stride), int(i.extent))
                for i in self.layout.shard
                if i.axis.name == "m" and int(i.extent) > 1
            )
            for stride, extent in storage:
                if stride < span:
                    raise ValueError(
                        f"{self.name}: register layout must have nonoverlapping storage strides"
                    )
                span += (extent - 1) * stride

    def value(self, atom):
        if self.ty.scope != MemoryScope("local"):
            raise ValueError(f"{self.name}: scalar values must be in registers")
        if self.raw is None:
            span = self.layout.span("m") + self.ty.elem_offset
            self.raw = Var(
                "tile_" + self.name,
                T.Tensor((span,), str(self.dtype), scope="local", layout="default"),
            )
            self.ctx.bindings.append(
                Bind(self.raw, T.decl_tensor(self.tensor.data, ty_args=[self.raw.ty]))
            )
        return self.raw[self.physical(atom)["m"]]

    def pointer(self, atom):
        return self.tensor.ptr_to(self.indices(atom))

    def require_bits(self, bits):
        if self.bits != bits or self.dtype.lanes != 1:
            raise ValueError(
                f"{self.name}: expected scalar {bits}-bit elements; use an explicit dtype view"
            )

    @property
    def atom_size(self):
        return prod(self.atom_shape)
