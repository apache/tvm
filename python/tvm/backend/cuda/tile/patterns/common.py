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

"""Shared region addressing and repeat loops for explicit instruction patterns.

No layout is reordered here. Pattern checks evaluate the caller's layout at
canonical coordinates; emitters use those same coordinates in TensorLoads.
"""

from tvm.backend.cuda.ptx import PTXNamespace
from tvm.ir import Bind, Evaluate, For, ForKind, PrimType, Range, Var
from tvm.script import tirx as T
from tvm.sym import Analyzer
from tvm.tirx import Function
from tvm.tirx.layout import TileLayout


class Operands:
    """Two equally shaped regions: repeat prefix followed by a fixed atom suffix."""

    def __init__(self, dst, src, atom_shape, src_scope, sctx):
        self.dst, self.src = dst, src
        self.analyzer = Analyzer()
        for var, bound in sctx.var_range_map.items():
            self.analyzer.bind(var, bound)
        shapes = []
        self.layouts = {}
        for name, region in (("dst", dst), ("src", src)):
            ty = region.source.ty
            if str(ty.dtype.dtype) not in ("float32", "int32", "uint32"):
                raise ValueError(f"{name}: pattern requires scalar 32-bit elements")
            if ty.storage_scope not in (("local",) if name == "dst" else src_scope):
                raise ValueError(f"{name}: incompatible storage scope {ty.storage_scope}")
            shape = tuple(int(self.analyzer.simplify(r.extent)) for r in region.region)
            if any(n <= 0 for n in shape):
                raise ValueError(f"{name}: pattern requires positive static region extents")
            if len(shape) < len(atom_shape) or shape[-len(atom_shape) :] != atom_shape:
                raise ValueError(f"{name}: expected trailing atom shape {atom_shape}, got {shape}")
            for r, size in zip(region.region, ty.shape):
                self.require(r.min >= 0, f"{name}: cannot prove region start is in bounds")
                self.require(r.min + r.extent <= size, f"{name}: region exceeds tensor shape")
            layout = ty.effective_layout()
            if not isinstance(layout, TileLayout) or layout.replica:
                raise ValueError(f"{name}: prototype requires a TileLayout without replicas")
            self.layouts[name] = layout
            shapes.append(shape)
        if shapes[0] != shapes[1]:
            raise ValueError(f"Repeat shapes must agree: dst={shapes[0]}, src={shapes[1]}")
        self.repeat_shape = shapes[0][: -len(atom_shape)]
        self.repeat = [Var(f"repeat_{i}", "int32") for i in range(len(self.repeat_shape))]
        for var, size in zip(self.repeat, self.repeat_shape):
            self.analyzer.bind(var, Range(0, size))
        alignment = int(src.source.ty.data_alignment)
        if alignment < 16 or alignment % 16:
            raise ValueError("src: pattern requires a known 16-byte-aligned base (Tensor align=16)")
        # Registers may have gaps/permuted strides, but must name distinct slots.
        # This sufficient mixed-radix test is deliberately not an alias solver.
        span = 1
        storage = sorted(
            (int(it.stride), int(it.extent))
            for it in self.layouts["dst"].shard
            if it.axis.name == "m" and int(it.extent) > 1
        )
        for stride, extent in storage:
            if stride < span:
                raise ValueError("dst: register layout must have nonoverlapping storage strides")
            span += (extent - 1) * stride
        # Ownership has been checked against the pattern before this view is
        # used. Bind the same physical storage, so PTX outputs can address
        # nonadjacent registers without a temporary tensor or a remap copy.
        dst_ty = dst.source.ty
        raw_span = self.layouts["dst"].span("m") + dst_ty.elem_offset
        self.dst_raw = Var(
            "tile_dst",
            T.Tensor((raw_span,), str(dst_ty.dtype.dtype), scope="local", layout="default"),
        )

    def require(self, condition, message):
        if not self.analyzer.can_prove(condition):
            raise ValueError(message)

    def indices(self, name, atom, repeat=None):
        region = getattr(self, name)
        coord = [*(self.repeat if repeat is None else repeat), *atom]
        return tuple(self.analyzer.simplify(r.min + i) for r, i in zip(region.region, coord))

    def physical(self, name, atom):
        region = getattr(self, name)
        result = dict(
            self.layouts[name].apply(*self.indices(name, atom), shape=region.source.ty.shape)
        )
        result["m"] = result.get("m", 0) + region.source.ty.elem_offset
        return {key: self.analyzer.simplify(value) for key, value in result.items()}

    def memory(self, name, atom, *, lane=None):
        physical = self.physical(name, atom)
        allowed = {"m"} if lane is None else {"m", "laneid"}
        if set(physical) - allowed:
            raise ValueError(
                f"{name}: pattern expects only axes {sorted(allowed)}, got {tuple(physical)}"
            )
        if lane is not None:
            self.require(
                physical.get("laneid", 0) == lane,
                "dst: ldmatrix requires owner lane = 4 * row + word",
            )
        return physical["m"]

    def aligned_row(self, atom):
        base = self.memory("src", atom)
        self.require(base % 4 == 0, "src: every instruction row must start at a 16-byte boundary")
        return base

    def value(self, name, atom):
        assert name == "dst"
        return self.dst_raw[self.physical(name, atom)["m"]]

    def pointer(self, name, atom):
        return getattr(self, name).source.ptr_to(self.indices(name, atom))

    def lower(self, opcode, emit):
        body = Evaluate(emit(self, PTXNamespace()[opcode]))
        for var, extent in reversed(list(zip(self.repeat, self.repeat_shape))):
            body = For(var, 0, extent, ForKind.UNROLLED, body)
        declaration = Bind(
            self.dst_raw, T.decl_tensor(self.dst.source.data, ty_args=[self.dst_raw.ty])
        )
        return Function([], [declaration, body], ret_type=PrimType("void"))
