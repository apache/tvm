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

"""TIRx script buffers."""

import math

import pytest
import tvm_ffi

import tvm
import tvm.script
import tvm.testing
from tvm.ir import TensorRegion, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TCol, TLane, laneid, warpid


def test_roundtrip_layout():
    def get_layout1():
        return T.TileLayout(T.S[(8, 8, 8, 4, 2) : (6, 4 @ laneid, 2, 1 @ laneid, 1)])

    def get_layout2():
        return T.TileLayout(T.S[(8, 8, 8, 4, 2) : (64, 4 @ laneid, 8, 2, 1)])

    def get_layout3():
        return T.TileLayout(T.S[(8, 16, 8, 16) : (1024, 16, 128, 1)])

    def get_layout4():
        return T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(512,)]))

    def get_layout5():
        return T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 4) : (64, 1, 64 * 64)]))

    # fmt: off
    @T.prim_func
    def test(_: T.Buffer((64,), 'float32', scope='global')) -> None:

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        C = T.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
        D = T.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())
        A_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
        B_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())

        E = T.alloc_buffer([64, 256], dtype="float16", scope="shared", layout=get_layout5())
        T.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0] + E[0, 0])
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_roundtrip_layout_replica_and_offset():
    """Round-trip layouts that exercise the replica and offset (single- and
    multi-axis) printer paths. The multi-axis case relies on
    `_LayoutSpec.__add__` correctly merging successive offset terms instead
    of overwriting (see `_merge_offset` in `tvm.tirx.layout`)."""

    def get_shard_replica():
        return T.TileLayout(T.S[8 : 4 @ laneid] + T.R[4 : 1 @ laneid])

    def get_shard_offset_single():
        return T.TileLayout(T.S[8 : 4 @ laneid] + 1 @ laneid)

    def get_shard_offset_multi():
        return T.TileLayout(T.S[8 : 4 @ laneid] + 1 @ laneid + 2 @ warpid + 64)

    def get_full():
        return T.TileLayout(T.S[(1,) : (1,)] + T.R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid)

    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_replica())
        B = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_single())
        C = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_multi())
        D = T.alloc_buffer([32], dtype="float16", scope="shared", layout=get_full())
        T.evaluate(A[0] + B[0] + C[0] + D[0])
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        A_warp_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
        A_warp = A.view(8, 8, layout=A_warp_layout)
        A_local = A_warp.local(2)
        A_local[0] = T.float16(0)

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


L_LANE = T.TileLayout(T.S[32 : 1 @ laneid])


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @T.prim_func
    def test(out: T.Buffer(2, 'float32', scope='global')) -> None:

        T.device_entry()
        bx, by, bz = T.cta_id([32, 32, 1])
        tx, ty, tz = T.thread_id([16, 8, 1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        A = T.alloc_buffer([2,], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
        B = A.view(8, 8, layout=B_layout)
        D = B.local(2)
        out[0] = A[0] + B[0, 0] + D[0]
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get3():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 8], dtype="float32", scope="local")
        A_f16 = A.view("float16")
        A_f64 = A.view("float64")
        A_f16[0, 0] = T.float16(0)
        A_f64[0, 0] = T.float64(0)

        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_allocated_addr():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10], "float32", scope="trn.sbuf", allocated_addr=1024)
        for i in T.serial(2):
            Tx.memset(A[i*5:i*5+5], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_implicit_buffer_region():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10, 10, 10), 'float32', layout=T.TileLayout(T.S[10, 10, 10]))):

        T.device_entry()
        Tx.memset(A[0], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_alloc_under_any_scope():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        for i in T.serial(10):
            A = T.alloc_buffer([100], "float32", scope="trn.sbuf", allocated_addr=1024)
            Tx.memset(A[i*10:i*10+10], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_alloc_apis():
    # fmt: off
    @T.meta_class
    class Test:
        def __init__(self, Ta, inner_pool):
            self.Ta = Ta
            self.inner_pool = inner_pool
            self.Tb = T.shared_scalar("float16")
            self.idx = T.local_scalar("int32")
            self.inner_pool2 = T.decl_scalar("float16", self.inner_pool.data, "shared.dyn", 5)

        @T.inline
        def init(self):
            self.Ta = self.Ta + T.float16(1)
            self.Tb = self.Tb + T.float16(2)
            self.idx.source[0] = T.int32(0)
            self.idx = self.idx + T.int32(1)
            self.inner_pool2 = self.inner_pool2 + T.float16(1)
            T.evaluate(T.address_of(self.Ta))
            T.evaluate(T.address_of(self.Tb))
            T.evaluate(T.address_of(self.idx))
            T.evaluate(T.address_of(self.inner_pool))
            T.evaluate(T.address_of(self.inner_pool2))

    @T.prim_func
    def test():
        T.device_entry()
                # normal buffer
        A = T.alloc_shared([10], "float16")
        B = T.alloc_local([10], "float16")
                # scalar buffer (alloc)
        C = T.shared_scalar("float16")
        D: T.float16
        pool = T.alloc_buffer([10], "uint8", scope="shared.dyn")
                # scalar buffer (decl)
        E = T.decl_scalar("float16", pool.data, "shared.dyn", 0)
                # normal 1-dim buffer with shape (1,)
        F = T.alloc_local((1,), "float16")
        Ta: T.float16
        inner_pool = T.decl_buffer(shape=[10], data=pool.data, dtype="uint8", scope="shared.dyn")
        test = Test(Ta, inner_pool)  # noqa: F821
        test.init()
        A[0] = C
        A[0] = C + D  # noqa: F821
        A[1] = B[0] * C
        D.source[0] = D + T.float16(1)  # noqa: F821
        D = D + T.float16(1)  # noqa: F821
        C = D
        T.evaluate(E)
        E = E + T.float16(1)
                # normal 1-dim buffer with shape (1,) can be assigned directly,
                # but not loaded directly
        F = F[0] + T.float16(1)
        C += D
        D += E + C + D
        T.evaluate(T.address_of(C))
        T.evaluate(C.source.access_ptr("rw", offset=0))
        T.evaluate(C.source.data)
        T.evaluate(D)
        T.evaluate(T.address_of(D))
        # fmt: on

    code = test.script()
    print(code)
    assert ".buffer" not in code
    assert from_source(code).script() == code


def test_alloc_apis_reject_name_argument():
    with pytest.raises(TypeError):
        T.alloc_buffer((1,), "int32", name="buf")

    with pytest.raises(TypeError):
        T.local_scalar("int32", name="idx")


def test_buffer():
    # fmt: off
    @T.prim_func(private=True)
    def test(
        A: T.Buffer((10, 11), "float32", layout=None),
        B: T.Buffer((10, 11), "float32", scope="global"),
        C: T.Buffer((10, 11), "float32", layout="default"),
        D: T.Buffer((10, 11), "float32", layout=T.TileLayout(T.S[(10, 11) : (1, 10)])),
        _E: T.Buffer([10, 11], 'float16', layout=None),
        _F: T.Buffer([10, 11], 'float16', scope='global'),
        _G: T.Buffer([10, 11], 'float16', layout='default'),
        _H: T.Buffer([10, 11], 'float16', layout=T.TileLayout(T.S[(10, 11):(1, 10)])),
    ):


        _A0 = T.decl_buffer((10, 11), "float32", data=A.data, layout=None)
        _B0 = T.decl_buffer((10, 11), "float32", data=B.data, scope="global")
        _C0 = T.decl_buffer((10, 11), "float32", data=C.data, layout="default")
        _D0 = T.decl_buffer((10, 11), "float32", data=D.data, layout=T.TileLayout(T.S[(10, 11) : (1, 10)]))  # noqa: E501
        _A1 = T.alloc_buffer((10, 11), "float32", layout=None)
        _B1 = T.alloc_buffer((10, 11), "float32", scope="global")
        _C1 = T.alloc_buffer((10, 11), "float32", layout="default")
        _D1 = T.alloc_buffer((10, 11), "float32", layout=T.TileLayout(T.S[(10, 11) : (1, 10)]))

        pass
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_buffer_shape_repeated_var_prints_out_of_line():
    n = tvm.tirx.Var("n", "int32")
    buffer = tvm.tirx.decl_buffer((n + n,), name="A")
    func = tvm.tirx.PrimFunc([buffer], tvm.tirx.Evaluate(0))

    code = func.script(extra_config={"script.use_pep695": False})
    assert 'n = I.dynamic("n", dtype="int32")' in code
    assert_structural_equal(func, from_source(code))


def test_scalar_allocbuffer_annotation_and_init_merge():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        phase_mma = T.alloc_local((1,), "int32")
        phase_mma[0] = T.int32(0)
        phase_aux = T.alloc_local((1,), "int32")
        T.evaluate(phase_mma[0] + phase_aux[0])
        # fmt: on

    code = test.script()
    assert "phase_mma: T.int32 = 0" in code
    assert "phase_aux: T.int32" in code
    assert "phase_mma = T.alloc_local" not in code
    assert "phase_aux = T.alloc_local" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_layout_none_keeps_alloc_local():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        phase_mma = T.alloc_local((1,), "int32", layout=None)
        phase_mma[0] = T.int32(0)
        T.evaluate(phase_mma[0])
        # fmt: on

    code = test.script()
    assert 'phase_mma = T.alloc_local((1,), "int32", layout=None)' in code
    assert "phase_mma: T.int32" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_annotation_sugar():
    # fmt: off
    @T.prim_func
    def test():
        x = T.alloc_buffer((1,), "int32", scope="local")
        x[0] = T.int32(0)
        T.evaluate(x[0])
    # fmt: on

    code = test.script()
    assert "x: T.int32 = 0" in code
    assert "x = T.alloc_buffer" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_permute():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 4], dtype="float16", scope="local",
                            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]))
        B = A.permute(1, 0)
        B[0, 0] = T.float16(0)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_local_auto():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))
        B_local = B.local()
        B_local[0] = T.float16(0)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_buffer_local_ir():
    """Verify .local() infers the physical span and uses an identity layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))
        B_local = B.local()
        B_local[0] = T.float16(0)
        # fmt: on

    _, b_buf, b_local = _collect_buffers(func)

    # Shared data pointer
    assert_structural_equal(_buffer_source(func, b_local), b_buf.data)
    # Shape: single dim matching the raw physical storage span
    assert len(b_local.ty.shape) == 1
    storage = b_buf.ty.layout.storage()
    assert int(b_local.ty.shape[0]) == int(storage.span())
    # The inferred view uses physical storage order, not storage-iterator order.
    assert b_local.ty.layout.is_trivial()

    # Round-trip
    code = func.script()
    assert "buffer_1 = buffer.local()" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def _collect_buffers(func):
    """Collect native buffers in declaration order, including anonymous views."""
    buffers = []

    def visit(node):
        if isinstance(node, tvm.tirx.DeclBuffer | tvm.tirx.AllocBuffer):
            buffers.append(node.buffer)

    tvm_ffi.structural_walk(func.body, visit)
    return buffers


def _buffer_source(func, buffer):
    """Find the unique declaration source by native buffer identity."""
    sources = []

    def visit(node):
        if isinstance(node, tvm.tirx.DeclBuffer) and node.buffer.same_as(buffer):
            sources.append(node.data)

    tvm_ffi.structural_walk(func.body, visit)
    assert len(sources) == 1
    return sources[0]


def test_buffer_local_physical_order():
    """Both inferred and explicit shapes map a non-trivial fragment physically."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_flat = B.local()
        B_2d = B.local(4, 8)
        B_flat[2] = T.float32(1)
        B_2d[0, 2] = T.float32(2)
        # fmt: on

    _, b_buf, b_flat, b_2d = _collect_buffers(func)

    # The parent storage view enumerates storage iters in a different order
    # from their physical strides, so inheriting it would permute registers.
    assert not b_buf.ty.layout.storage().is_trivial()

    for local in [b_flat, b_2d]:
        assert_structural_equal(_buffer_source(func, local), b_buf.data)
        assert local.ty.layout.is_trivial()
    assert [int(dim) for dim in b_flat.ty.shape] == [32]
    assert [int(dim) for dim in b_2d.ty.shape] == [4, 8]

    # Index 2 in either row-major shape is the same physical register.
    flat_offset = b_flat.ty.layout.apply(2, shape=list(b_flat.ty.shape))["m"]
    reshaped_offset = b_2d.ty.layout.apply(0, 2, shape=list(b_2d.ty.shape))["m"]
    assert int(flat_offset) == int(reshaped_offset) == 2

    code = func.script()
    assert "buffer_1 = buffer.local()" in code
    assert "buffer_2 = buffer.local(4, 8)" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_local_layout_overrides_roundtrip():
    """Storage and arbitrary mediated layouts remain explicit overrides."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_storage = B.local(layout=B.layout.storage())
        # An explicit layout is an escape hatch and may describe a smaller
        # mediated view than the parent's full per-thread storage.
        B_custom = B.local(2, 4, layout=T.TileLayout(T.S[(2, 4) : (1, 2)]))
        B_storage[0] = T.float32(1)
        B_custom[0, 0] = T.float32(2)
        # fmt: on

    _, b_buf, b_storage, b_custom = _collect_buffers(func)
    assert_structural_equal(b_storage.ty.layout, b_buf.ty.layout.storage())
    assert not b_storage.ty.layout.is_trivial()
    assert not b_custom.ty.layout.is_trivial()

    code = func.script()
    storage_line = next(line for line in code.splitlines() if "buffer_1 =" in line)
    custom_line = next(line for line in code.splitlines() if "buffer_2 =" in line)
    assert ".local(layout=" in storage_line
    assert ".local(2, 4, layout=" in custom_line
    assert_structural_equal(func, from_source(code))
    assert from_source(code).script() == code


def test_buffer_local_explicit_layout_without_parent_layout():
    """An explicit shape and layout do not inspect the parent's absent layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer((4,), dtype="float32", scope="local", layout=None)
        B = A.local(4, layout=T.TileLayout(T.S[4]))
        B[0] = T.float32(1)
        # fmt: on

    a_buf, b_buf = _collect_buffers(func)
    assert a_buf.ty.layout is None
    assert b_buf.ty.layout.is_trivial()
    code = func.script()
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_compose_layout_printer_roundtrip():
    """Generic view sugar keeps a physical local view's identity layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 8),
            dtype="float32",
            scope="local",
            layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(8, 8)])),
        )
        B = A.local()
        B[0] = T.float32(1)
        # fmt: on

    _, b_buf = _collect_buffers(func)
    assert [int(dim) for dim in b_buf.ty.shape] == [64]
    assert b_buf.ty.layout.is_trivial()
    code = func.script()
    local_line = next(line for line in code.splitlines() if "buffer =" in line)
    assert ".view(64, layout=" in local_line
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_inference_without_parent_layout_has_clear_diagnostic():
    """Shape inference requires a parent storage layout."""

    with pytest.raises(ValueError, match="parent buffer has layout=None"):
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.alloc_buffer((4,), dtype="float32", scope="local", layout=None)
            B = A.local(layout=T.TileLayout(T.S[4]))
            B[0] = T.float32(1)


def test_buffer_local_physical_span_includes_gaps_and_offset():
    """The raw local view includes every slot up to the storage span."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([6], dtype="float32", scope="local")
        B = A.view(32, 2, layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)] + 3))
        B_flat = B.local()
        B_2d = B.local(2, 3)
        B_storage = B.local(2, layout=B.layout.storage())
        B_flat[5] = T.float32(1)
        B_2d[1, 2] = T.float32(2)
        B_storage[1] = T.float32(3)
        # fmt: on

    _, b_buf, b_flat, b_2d, b_storage = _collect_buffers(func)
    assert int(b_buf.ty.layout.storage().span()) == 6
    assert int(b_buf.ty.layout.storage().size()) == 2
    assert [int(dim) for dim in b_flat.ty.shape] == [6]
    assert [int(dim) for dim in b_2d.ty.shape] == [2, 3]
    assert [int(dim) for dim in b_storage.ty.shape] == [2]
    for local in [b_flat, b_2d]:
        assert local.ty.layout.is_trivial()
    for i in range(6):
        assert int(b_flat.ty.layout.apply(i, shape=list(b_flat.ty.shape))["m"]) == i
    assert int(b_2d.ty.layout.apply(1, 2, shape=list(b_2d.ty.shape))["m"]) == 5
    assert_structural_equal(b_storage.ty.layout, b_buf.ty.layout.storage())
    assert int(b_storage.ty.layout.apply(0, shape=list(b_storage.ty.shape))["m"]) == 3
    assert int(b_storage.ty.layout.apply(1, shape=list(b_storage.ty.shape))["m"]) == 5

    code = func.script()
    storage_line = next(line for line in code.splitlines() if "buffer_3 =" in line)
    assert ".local(layout=" in storage_line
    assert_structural_equal(func, from_source(code))
    assert from_source(code).script() == code


def test_buffer_local_printer_is_stable_with_multiple_aliases():
    """Thread-layout parents win deterministically over sibling aliases."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_flat = B.local()
        B_2d = B.local(4, 8)
        B_storage = B.local(layout=B.layout.storage())
        B_flat[0] = B_2d[0, 0] + B_storage[0]
        # fmt: on

    expected = func.script()
    assert "buffer_1 = buffer.local()" in expected
    assert "buffer_2 = buffer.local(4, 8)" in expected
    storage_line = next(line for line in expected.splitlines() if "buffer_3 =" in line)
    assert ".local(layout=" in storage_line
    for _ in range(20):
        parsed = from_source(expected)
        assert parsed.script() == expected
        assert_structural_equal(func, parsed)


def test_buffer_local_printer_preserves_inherited_metadata():
    """Local sugar falls back when it would discard Buffer metadata."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            [32, 2],
            dtype="float32",
            elem_offset=8,
            scope="local",
            layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)]),
        )
        B_align = T.decl_buffer(
            (2,),
            dtype="float32",
            data=A.data,
            elem_offset=8,
            scope="local",
            align=128,
        )
        B_factor = T.decl_buffer(
            (2,),
            dtype="float32",
            data=A.data,
            elem_offset=8,
            scope="local",
            offset_factor=8,
        )
        B_align[0] = B_factor[0]
        # fmt: on

    code = func.script()
    align_line = next(line for line in code.splitlines() if "B_align =" in line)
    factor_line = next(line for line in code.splitlines() if "B_factor =" in line)
    assert "T.decl_buffer" in align_line and "align=128" in align_line
    assert "T.decl_buffer" in factor_line and "offset_factor=8" in factor_line
    assert ".local(" not in align_line
    assert ".local(" not in factor_line
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_rejects_shape_that_does_not_match_physical_span():
    """An explicit local shape product must preserve the physical span."""

    with pytest.raises(ValueError, match="physical storage span 6 per thread"):
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.alloc_buffer([6], dtype="float32", scope="local")
            B = A.view(32, 2, layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)] + 3))
            B_local = B.local(2)
            B_local[0] = T.float32(0)


def test_buffer_permute_ir():
    """Verify .permute(1, 0): shape swapped, layout permuted, shared data."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 4], dtype="float16", scope="local",
                            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]))
        B = A.permute(1, 0)
        B[0, 0] = T.float16(0)
        # fmt: on

    a_buf, b_buf = _collect_buffers(func)

    # Shared data pointer
    assert_structural_equal(_buffer_source(func, b_buf), a_buf.data)
    # Shape: [4, 8] from [8, 4]
    assert int(b_buf.ty.shape[0]) == 4
    assert int(b_buf.ty.shape[1]) == 8
    # Layout: permuted
    assert_structural_equal(b_buf.ty.layout, a_buf.ty.layout.permute_dims([1, 0]))

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_rearrange_allows_arbitrary_axis_names():
    @T.prim_func
    def ordinary_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(outer inner) tail -> outer tail inner", outer=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def buf_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(buf inner) tail -> buf tail inner", buf=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def self_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(self inner) tail -> self tail inner", self=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def pattern_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(pattern inner) tail -> pattern tail inner", pattern=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def keyword_pattern() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange(pattern="(outer inner) tail -> outer tail inner", outer=2)
        B[0, 0, 0] = T.float16(0)

    _, _, _, expected = _collect_buffers(ordinary_axis)
    for func in (buf_axis, self_axis, pattern_axis, keyword_pattern):
        _, _, _, actual = _collect_buffers(func)
        assert_structural_equal(actual.shape, expected.shape)
        assert_structural_equal(actual.layout, expected.layout)


def test_buffer_permute_compose_layout_ir():
    """Verify .permute on a swizzle-composed layout: the swizzle is preserved
    and the inner tile layout's dim groups are permuted (the reshape-permute-
    reshape idiom used to refactor gather views without restating strides)."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            [4, 4, 4, 64], dtype="bfloat16", scope="shared.dyn",
            layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 4, 4, 64) : (1024, 256, 64, 1)])),
        )
        B = A.permute(1, 0, 2, 3)
        B[0, 0, 0, 0] = T.bfloat16(0)
        # fmt: on

    a_buf, b_buf = _collect_buffers(func)

    assert_structural_equal(_buffer_source(func, b_buf), a_buf.data)
    assert [int(s) for s in b_buf.shape] == [4, 4, 4, 64]
    expected = tvm.tirx.layout.ComposeLayout(
        a_buf.layout.per_element,
        a_buf.layout.swizzle_len,
        a_buf.layout.atom_len,
        a_buf.layout.tile_layout.permute_dims([1, 0, 2, 3]),
        a_buf.layout.swizzle_inner,
    )
    assert_structural_equal(b_buf.layout, expected)

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_sub_multi_iter_dim_ir():
    """sub with an int index on a dim carried by several layout iters
    decomposes the index mixed-radix across the iters' strides."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 16], dtype="float16", scope="local",
                           layout=T.TileLayout(T.S[(2, 4, 16) : (1024, 64, 1)]))
        B = A.sub[5]
        B[0] = T.float16(0)
        # fmt: on

    a_buf, b_buf = _collect_buffers(func)
    # 5 -> (5 // 4, 5 % 4) = (1, 1) -> 1 * 1024 + 1 * 64
    assert int(tvm.sym.Analyzer().simplify(b_buf.elem_offset - a_buf.elem_offset)) == 1088
    assert [int(s) for s in b_buf.shape] == [16]
    assert_structural_equal(b_buf.layout, tvm.tirx.layout.TileLayout(T.S[(16,) : (1,)]))

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_sub_multi_iter_misaligned_rejected():
    buf = tvm.tirx.decl_buffer(
        (8, 16), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(2, 4, 16) : (1024, 64, 1)])
    )
    # sub[2:6] narrows the multi-iter dim 0 at a misaligned offset.
    with pytest.raises(ValueError, match="multiples of the inner iter block"):
        buf.sub[2:6]


def test_buffer_sub_ir():
    """buf.sub follows numpy basic indexing as a view constructor: int drops
    the dim, a:b narrows, a::s strides. Offsets fold into elem_offset through
    the dim's layout iter strides; the derived layout carries the survivors."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([4, 8, 16], dtype="float16", scope="local",
                           layout=T.TileLayout(T.S[(4, 8, 16) : (256, 16, 1)]))
        B = A.sub[1, 2:6]
        B[0, 0] = T.float16(0)
        C = A.sub[:, 1::2]
        C[0, 0, 0] = T.float16(0)
        # fmt: on

    a_buf, _, b_buf, _, c_buf = _collect_buffers(func)
    # sub[1, 2:6]: drop dim 0 at 1 (1 * 256) then narrow dim 1 to [2, 6) (2 * 16)
    assert [int(s) for s in b_buf.shape] == [4, 16]
    assert int(tvm.sym.Analyzer().simplify(b_buf.elem_offset - a_buf.elem_offset)) == 288
    assert_structural_equal(b_buf.layout, tvm.tirx.layout.TileLayout(T.S[(4, 16) : (16, 1)]))
    # sub[:, 1::2]: keep dim 0, split dim 1 into (4, 2) and fix the remainder at 1
    assert [int(s) for s in c_buf.shape] == [4, 4, 16]
    assert int(tvm.sym.Analyzer().simplify(c_buf.elem_offset - a_buf.elem_offset)) == 16
    assert_structural_equal(
        c_buf.layout, tvm.tirx.layout.TileLayout(T.S[(4, 4, 16) : (256, 32, 1)])
    )

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_view_surgery_static_bounds_rejected():
    """Statically-known out-of-range sub arguments must be rejected loudly
    (review finding: OOB offsets were silent)."""
    buf = tvm.tirx.decl_buffer(
        (10,), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(10,) : (1,)])
    )
    grid = tvm.tirx.decl_buffer(
        (4, 8), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(4, 8) : (8, 1)])
    )
    # int index: static bounds
    with pytest.raises(ValueError, match="out of range"):
        buf.sub[10]
    with pytest.raises(ValueError, match="out of range"):
        buf.sub[-1]
    # slice narrow: static range bounds
    with pytest.raises(ValueError, match="exceeds"):
        buf.sub[8:12]
    with pytest.raises(ValueError, match="must be non-negative"):
        buf.sub[-2:2]
    with pytest.raises(ValueError, match="must be positive"):
        buf.sub[5:3]
    # grid.sub: out-of-range int, exceeding narrow, stepped-start out of range
    with pytest.raises(ValueError, match="out of range"):
        grid.sub[10, :]
    with pytest.raises(ValueError, match="exceeds"):
        grid.sub[:, 4:12]
    with pytest.raises(ValueError, match=r"in \[0, 2\)"):
        grid.sub[:, -1::2]


def test_buffer_sub_swizzle_commutation():
    """A folded view offset moves into elem_offset only when it commutes
    with the swizzle, i.e. is a multiple of the swizzle period
    2^(per_element + atom_len + swizzle_len). Sub-period offsets stay inside
    the derived tile layout's offset so the swizzle keeps applying to them
    (review finding: folding them outside produced wrong addresses). Both
    placements must be address-equivalent to the parent layout."""

    def addr(buf, base, *coords):
        analyzer = tvm.sym.Analyzer()
        if len(coords) == 1:
            rel = buf.layout.apply(coords[0])["m"]
        else:
            rel = buf.layout.apply(*coords, shape=[int(s) for s in buf.shape])["m"]
        return int(analyzer.simplify((buf.elem_offset - base) + rel))

    analyzer = tvm.sym.Analyzer()
    compose = T.ComposeLayout(
        3, 3, 3, T.TileLayout(T.S[(4, 1024) : (1024, 1)])
    )  # period = 2^(3+3+3) = 512 elements

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([4, 1024], dtype="bfloat16", scope="shared.dyn", layout=compose)
        B = A.sub[1]  # offset 1024 = 2 * period: folds into elem_offset
        B[0] = T.bfloat16(0)
        C = A.sub[:, 512:1024]  # offset 512 = period: folds into elem_offset
        C[0, 0] = T.bfloat16(0)
        # fmt: on

    a_buf, b_buf, c_buf = _collect_buffers(func)
    base = a_buf.elem_offset
    assert int(analyzer.simplify(b_buf.elem_offset - base)) == 1024
    for j in (0, 1, 63, 511, 1023):
        assert addr(a_buf, base, 1024 + j) == addr(b_buf, base, j)
    for j in (0, 1, 255, 511):
        assert addr(a_buf, base, 512 + j) == addr(c_buf, base, 0, j)

    code = func.script()
    assert from_source(code).script() == code

    # Sub-period offsets do not commute: they stay inside the tile layout's
    # offset (elem_offset unchanged) and every address matches the parent.
    compose2 = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 16, 8) : (128, 8, 1)]))

    # fmt: off
    @T.prim_func
    def func2() -> None:
        T.device_entry()
        A = T.alloc_buffer([2, 16, 8], dtype="float16", scope="shared.dyn", layout=compose2)
        B = A.sub[:, 1]  # offset 8
        B[0, 0] = T.float16(0)
        C = A.sub[:, :, 1]  # offset 1
        C[0, 0] = T.float16(0)
        D = A.sub[:, 1:3]  # offset 8
        D[0, 0, 0] = T.float16(0)
        E = A.sub[:, 1:3]  # narrow via sub
        E[0, 0, 0] = T.float16(0)
        for w in T.serial(16):
            F = A.sub[:, w]  # dynamic sub-period offset
            F[0, 0] = T.float16(0)
        # fmt: on

    a2, b2, c2, d2, e2, _ = _collect_buffers(func2)
    base2 = a2.elem_offset
    shape2 = [2, 16, 8]
    for name, child, to_parent in [
        ("B", b2, lambda c: (c[0], 1, c[1])),
        ("C", c2, lambda c: (c[0], c[1], 1)),
        ("D", d2, lambda c: (c[0], 1 + c[1], c[2])),
        ("E", e2, lambda c: (c[0], 1 + c[1], c[2])),
    ]:
        assert int(analyzer.simplify(child.elem_offset - base2)) == 0
        child_shape = [int(s) for s in child.shape]
        for flat in range(math.prod(child_shape)):
            coords, rem = [], flat
            for extent in reversed(child_shape):
                coords.append(rem % extent)
                rem //= extent
            coords = tuple(reversed(coords))
            assert addr(a2, base2, *to_parent(coords)) == addr(child, base2, *coords), (
                name,
                coords,
            )

    code = func2.script()
    assert from_source(code).script() == code

    # fixed-point windows (all touched addresses below 2^(per_element +
    # atom_len)) are correct through the same layout-offset placement
    compose3 = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64,) : (1,)]))

    # fmt: off
    @T.prim_func
    def func3() -> None:
        T.device_entry()
        A = T.alloc_buffer([64], dtype="bfloat16", scope="shared.dyn", layout=compose3)
        B = A.sub[8:16]
        B[0] = T.bfloat16(0)
        # fmt: on

    a3, b3 = _collect_buffers(func3)
    for j in range(8):
        assert addr(a3, a3.elem_offset, 8 + j) == addr(b3, a3.elem_offset, j) == 8 + j


def test_buffer_tile_ir():
    """buf.tile((dim, factors))[picks] splits dims into factors and picks
    chunks in one call: int/Expr picks a factor, ':' keeps it, kept
    factors merge back. Equivalent to the view (reshape) + sub chain."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([3, 64, 512], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(3, 64, 512) : (64 * 512, 512, 1)]))
        for w in T.serial(4):
            B = A.tile((1, (-1, 4, 4)))[:, w, :]
            B[0, 0, 0] = T.float16(0)
            C = A.view(3, 4, 4, 4, 512).sub[:, :, w].view(3, 16, 512)
            C[0, 0, 0] = T.float16(0)
        D = A.tile((1, (-1, 4)))[:, 2]
        D[0, 0, 0] = T.float16(0)
        E = A.sub[:, 2::4]
        E[0, 0, 0] = T.float16(0)
        F = A.tile((1, (4, -1)))[2, :]
        F[0, 0, 0] = T.float16(0)
        G = A.sub[:, 32:48]
        G[0, 0, 0] = T.float16(0)

    @T.prim_func
    def func_multi() -> None:
        T.device_entry()
        A = T.alloc_buffer([64, 128], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(64, 128) : (128, 1)]))
        for wx in T.serial(4):
            for wy in T.serial(2):
                H = A.tile((0, (-1, 4, 2)), (1, (-1, 2, 8)))[:, wx, :, :, wy, :]
                H[0, 0] = T.float16(0)
                J = (A.view(8, 4, 2, 128).sub[:, wx].view(16, 128)
                      .view(16, 8, 2, 8).sub[:, :, wy].view(16, 64))
                J[0, 0] = T.float16(0)

    @T.prim_func
    def func_multipick() -> None:
        T.device_entry()
        A = T.alloc_buffer([128, 16], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(128, 16) : (16, 1)]))
        for a in T.serial(2):
            for b in T.serial(4):
                K = A.tile((0, (2, 4, -1)))[a, b, :]
                K[0, 0] = T.float16(0)
                L = A.view(2, 4, 16, 16).sub[a, b]
                L[0, 0] = T.float16(0)
    # fmt: on

    # Tile/view chains also declare intermediate reshaped and selected buffers.
    _, _, _, b, _, _, c, _, d, _, e, _, f, g = _collect_buffers(func)
    assert [int(s) for s in b.shape] == [3, 16, 512]
    assert_structural_equal(b.layout, c.layout)
    assert_structural_equal(d.layout, e.layout)
    assert_structural_equal(f.layout, g.layout)
    _, _, _, _, _, _, h, _, _, _, _, _, j = _collect_buffers(func_multi)
    assert [int(s) for s in h.shape] == [16, 64]
    assert_structural_equal(h.layout, j.layout)
    _, _, _, k, _, _, l_buf = _collect_buffers(func_multipick)
    assert [int(s) for s in k.shape] == [16, 16]
    assert_structural_equal(k.layout, l_buf.layout)

    code = func_multipick.script()
    assert from_source(code).script() == code


def test_buffer_tile_rejected():
    buf = tvm.tirx.decl_buffer(
        (3, 64, 512),
        "float16",
        layout=tvm.tirx.layout.TileLayout(T.S[(3, 64, 512) : (64 * 512, 512, 1)]),
    )
    with pytest.raises(ValueError, match="takes a dim and a factors"):
        buf.tile(1, 4, 4)  # positional single-dim form takes exactly (dim, factors)
    with pytest.raises(ValueError, match="picks no factor"):
        buf.tile(1, (-1, 4))[:, :]  # a chunk must pick at least one factor
    with pytest.raises(ValueError, match="non-empty tuple"):
        buf.tile((1, 4))  # factors must be a tuple
    with pytest.raises(ValueError, match="non-empty tuple"):
        buf.tile((1, ()))
    with pytest.raises(ValueError, match="tiled more than once"):
        buf.tile((1, (4, -1)), (1, (2, -1)))
    with pytest.raises(ValueError, match="index"):
        buf.tile((1, (-1, 4)))[2]  # 2 factors, 1 index
    with pytest.raises(ValueError, match="must be ':'"):
        buf.tile((1, (-1, 4)))[:, 1:3]  # sub-slice on a factor


def test_buffer_chunk_ir():
    """buf.chunk(spec)[picks] narrows each chunked dim to its picked chunk's
    contiguous [c*k : (c+1)*k) range (k = E // n), rank-preserving: a per-dim
    tuple where None passes the pick straight through and n divides that dim
    into n equal chunks. chunk(spec)[picks] is the exact same BufferRegion as
    the hand-written a*k:(a+1)*k slice — no reshape, no extra dim."""

    compose = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 512) : (512, 1)]))
    A = tvm.tirx.decl_buffer(
        (4, 8, 16), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(4, 8, 16) : (128, 16, 1)])
    )
    C = tvm.tirx.decl_buffer((4, 512), "bfloat16", layout=compose)

    # chunk((None, None, 2))[:, :, 1] narrows dim 2 (extent 16) to chunk 1 of 2
    # → [8:16] (k = 16 // 2 = 8); rank preserved, dims 0/1 pass through as ':'.
    reg = A.chunk((None, None, 2))[:, :, 1]
    assert isinstance(reg, TensorRegion)
    assert len(reg.region) == 3  # rank-preserving: no extra extent-1 chunk dim
    assert (int(reg.region[2].min), int(reg.region[2].extent)) == (8, 8)
    assert_structural_equal(reg, A[:, :, 8:16])

    # a None dim passes an int pick straight through (int → extent-1 region),
    # while the chunked dim still narrows to its picked chunk.
    reg2 = A.chunk((None, None, 2))[3, :, 0]
    assert_structural_equal(reg2, A[3, :, 0:8])

    # chunk((None, 4))[:, 2] on the swizzle-carrying compose layout: dim 1
    # (extent 512) → chunk 2 of 4 → [256:384] (k = 128), byte-identical slice.
    reg_c = C.chunk((None, 4))[:, 2]
    assert (int(reg_c.region[1].min), int(reg_c.region[1].extent)) == (256, 128)
    assert_structural_equal(reg_c, C[:, 256:384])

    # a symbolic (Expr) chunk index translates to c*k : (c+1)*k as well.
    c = T.Var(name="c", ty="int32")
    assert_structural_equal(A.chunk((None, None, 2))[:, :, c], A[:, :, c * 8 : c * 8 + 8])

    # validation
    with pytest.raises(ValueError, match="per-dim tuple"):
        A.chunk(2)  # spec must be a per-dim tuple, not a bare int
    with pytest.raises(ValueError, match="spec length"):
        A.chunk((None, 2))  # length 2 != rank 3
    with pytest.raises(ValueError, match="None or a positive int"):
        A.chunk((None, None, 0))  # 0 is not a positive chunk count
    with pytest.raises(ValueError, match="chunk index, not a slice"):
        A.chunk((None, None, 2))[:, :, 0:1]  # a chunked dim takes a chunk index
    with pytest.raises(ValueError, match="rank-3 spec"):
        A.chunk((None, None, 2))[0, 0, 0, 0]  # too many indices


def test_buffer_view_dtype_ir():
    """Verify .view('float32') on float16: dtype correct, last dim halved, shared data."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 8], dtype="float16", scope="local")
        B = A.view("float32")
        B[0, 0] = T.float32(0)
        # fmt: on

    a_buf, b_buf = _collect_buffers(func)

    # Shared data pointer
    assert_structural_equal(_buffer_source(func, b_buf), a_buf.data)
    # dtype
    assert str(b_buf.ty.dtype) == "float32"
    # Shape: [8, 4] (last dim halved since float32 is 2x float16)
    assert int(b_buf.ty.shape[0]) == 8
    assert int(b_buf.ty.shape[1]) == 4

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_slice_region():
    """Verify A[slice] returns BufferRegion (not DeclBuffer)."""

    buf = tvm.tirx.decl_buffer((128, 64), "float16")
    br = buf[32:64, 0:32]
    assert isinstance(br, TensorRegion)
    assert br.source.same_as(buf)
    assert int(br.region[0].extent) == 32
    assert int(br.region[1].extent) == 32

    load = buf[1, 2]
    assert isinstance(load, tvm.ir.TensorLoad)

    partial = buf[1]
    assert isinstance(partial, TensorRegion)

    narrowed = br[4:12, 2:10]
    assert isinstance(narrowed, TensorRegion)
    assert narrowed.source.same_as(buf)
    assert [(int(dim.min), int(dim.extent)) for dim in narrowed.region] == [
        (36, 8),
        (2, 8),
    ]

    chained_load = br[3, 4]
    assert isinstance(chained_load, tvm.ir.TensorLoad)
    assert chained_load.source.same_as(buf)
    assert [int(index) for index in chained_load.indices] == [35, 4]

    point_then_region = br[3]
    assert isinstance(point_then_region, TensorRegion)
    assert [(int(dim.min), int(dim.extent)) for dim in point_then_region.region] == [
        (35, 1),
        (0, 32),
    ]

    with pytest.raises(ValueError, match="non-unit step"):
        _ = br[::2]


def test_global_call_realizes_buffer_elements():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def add(a: T.float32, b: T.float32) -> T.float32:
            return a + b

        @T.prim_func
        def main(
            A: T.Buffer((16,), "float32"),
            B: T.Buffer((16,), "float32"),
            C: T.Buffer((16,), "float32"),
        ):
            for i in range(16):
                C[i] = Module.add(A[i], B[i])

    assert isinstance(Module["main"], tvm.tirx.PrimFunc)


def test_buffer_sub_tmem_offset_uses_physical_columns():
    """A tmem layout measures TCol in elements, but allocated_addr measures
    physical 32-bit columns.  Folding a sub-view offset must scale by dtype
    width exactly once (the FlashMLA Q-tail view is the bf16 regression)."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        Q = T.decl_buffer(
            (2, 64, 288), "bfloat16", scope="tmem", allocated_addr=256,
            layout=T.TileLayout(T.S[(2, 64, 288) : (64 @ TLane, 1 @ TLane, 1 @ TCol)]),
        )
        Q_tail = Q.sub[:, :, 256:288]
        F8 = T.decl_buffer(
            (64, 128), "float8_e4m3fn", scope="tmem", allocated_addr=32,
            layout=T.TileLayout(T.S[(64, 128) : (1 @ TLane, 1 @ TCol)]),
        )
        F8_tail = F8.sub[:, 64:96]
        F32 = T.decl_buffer(
            (64, 128), "float32", scope="tmem", allocated_addr=64,
            layout=T.TileLayout(T.S[(64, 128) : (1 @ TLane, 1 @ TCol)]),
        )
        F32_tail = F32.sub[:, 32:64]
        T.evaluate(Q_tail[0, 0, 0])
        T.evaluate(F8_tail[0, 0])
        T.evaluate(F32_tail[0, 0])
        # fmt: on

    _, q_tail, _, f8_tail, _, f32_tail = _collect_buffers(func)
    assert int(q_tail.allocated_addr[0]) == 384  # 256 + 256 * 16 / 32
    assert int(f8_tail.allocated_addr[0]) == 48  # 32 + 64 * 8 / 32
    assert int(f32_tail.allocated_addr[0]) == 96  # 64 + 32 * 32 / 32
    for buffer in (q_tail, f8_tail, f32_tail):
        assert int(buffer.layout.offset.get(TCol, 0)) == 0

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_sub_tmem_rejects_partial_column_offset():
    buf_layout = tvm.tirx.layout.TileLayout(T.S[(64, 16) : (1 @ TLane, 1 @ TCol)])

    def build():
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.decl_buffer(
                (64, 16), "bfloat16", scope="tmem", allocated_addr=0, layout=buf_layout,
            )
            _ = A.sub[:, 1:3]
            # fmt: on

        return func

    with pytest.raises(ValueError, match="aligned to a physical 32-bit column"):
        build()


def test_roundtrip_tmem_decl_buffer():
    """DeclBuffer with tmem scope: data kwarg must be suppressed, allocated_addr
    must print as Expr (not Array), and scalar buffer index must not get
    a .source suffix."""

    # fmt: off
    @T.prim_func
    def func():
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            addr = T.alloc_shared((1,), "uint32", layout=None)
            addr_alias = T.decl_buffer((1,), "uint32", data=addr.data, scope="shared")
            buf = T.decl_buffer((64,), scope="tmem", layout=None, allocated_addr=addr_alias[0])
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))
    decls = []
    tvm_ffi.structural_walk(
        func.body,
        lambda node: decls.append(node) if isinstance(node, tvm.tirx.DeclBuffer) else None,
    )
    # The shared alias has an explicit definition before the tensor-memory use.
    assert len(decls) == 2
    tmem_decl = next(decl for decl in decls if decl.buffer.scope() == "tmem")
    assert tmem_decl.data.op.name == "tirx.reinterpret"
