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
import os
import subprocess
import sys

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import assert_structural_equal, load_json, save_json
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.testing import env
from tvm_tirx.tirx import decl_tensor
from tvm_tirx.tirx.layout import Axis, S, TileLayout
from tvm_tirx.tirx.script import ir_builder as I
from tvm_tirx.tirx.thread_scope import ThreadScope


def test_generic_layout_without_backend_autoload():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import decl_tensor
from tvm_tirx.tirx.layout import Axis, S, TileLayout
assert Axis.m.backend is None
assert T.Axis.m.same_as(Axis.m)
assert TileLayout(S[8]).apply(3, shape=[8])["m"] == 3
assert decl_tensor((8,), "float32").layout is None
tensor_type = T.Tensor((8,), "float32")
assert tensor_type.layout is None
tvm_tirx.ir.assert_structural_equal(tensor_type.effective_layout(), TileLayout(S[8]))
assert tensor_type.layout is None
assert "tvm_tirx.backend.cuda.axis" not in sys.modules
assert "tvm_tirx.backend.trn.axis" not in sys.modules
""",
        ],
        env={**os.environ, "TVM_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_backend_axis_and_scope_identity():
    assert Axis.m.backend is None
    assert T.cuda.Axis.laneid.backend == "cuda"
    assert T.trn.Axis.P.backend == "trn"
    assert T.cuda.Axis.laneid.get_scope().name == "cuda.warp"
    assert T.cuda.Axis.laneid.get_subscope().name == "cuda.thread"
    assert not T.trn.Axis.P.is_thread()
    with pytest.raises(AttributeError):
        _ = Axis.laneid
    with pytest.raises(AttributeError):
        _ = T.cuda.Axis.P
    with pytest.raises(AttributeError):
        _ = T.trn.Axis.laneid
    cuda = T.cuda.ThreadScope("thread")
    trn = T.trn.ThreadScope("thread")
    assert cuda.backend == "cuda" and trn.backend == "trn"
    assert not tvm_ffi.structural_equal(cuda, trn)
    assert_structural_equal(cuda, load_json(save_json(cuda)))
    assert_structural_equal(trn, load_json(save_json(trn)))
    for backend, factory in (("cuda", T.cuda.ThreadScope), ("trn", T.trn.ThreadScope)):
        root = factory("kernel")
        assert_structural_equal(root, ThreadScope(f"{backend}.kernel"))
        assert_structural_equal(root, load_json(save_json(root)))
        assert f'T.ThreadScope("{backend}.kernel")' in root.script()
    assert_structural_equal(T.cuda.Axis.bx.get_scope(), T.cuda.ThreadScope("kernel"))
    with pytest.raises(ValueError):
        T.trn.ThreadScope("warp")


@pytest.mark.parametrize(
    "scope", ["global", "local", "shared", "shared.dyn", "trn.sbuf", "trn.psum"]
)
def test_omitted_and_explicit_none_layout(scope):
    assert T.Tensor((4, 8), "float32", scope=scope).layout is None
    assert T.Tensor((4, 8), "float32", scope=scope, layout=None).layout is None
    assert decl_tensor((4, 8), "float32", scope=scope).layout is None
    explicit = T.Tensor((4, 8), "float32", scope=scope, layout="default")
    assert_structural_equal(explicit.layout, TileLayout(S[4, 8]))
    with IRBuilder():
        with I.function():
            tensor_type = I.Tensor((4, 8), "float32", scope=scope)
            allocated = I.bind(I.alloc_tensor(ty_args=[tensor_type]))
            declared = I.bind(I.decl_tensor(allocated.data, ty_args=[tensor_type]))
            assert allocated.layout is None and declared.layout is None


@pytest.mark.parametrize(
    "shape,strides",
    [((8, 16), None), ((8, 16), (16, 1)), ((8, 1, 16), (16, 99, 1))],
)
def test_effective_layout_on_tensor_type(shape, strides):
    tensor_type = T.Tensor(shape, "float32", strides=strides)
    before = save_json(tensor_type)
    assert_structural_equal(tensor_type.effective_layout(), TileLayout(S[shape]))
    assert save_json(tensor_type) == before


def test_effective_global_layout_preserves_source_and_region():
    tensor = decl_tensor((8, 16), "float32", elem_offset=13)
    before = save_json(tensor)
    region = tensor[2:4, 3:7]
    layout = region.source.ty.effective_layout()
    assert int(layout.apply(2, 3, shape=tensor.shape)["m"]) == 35
    starts = [r.min for r in region.region]
    assert int(layout.apply(*starts, shape=tensor.shape)["m"]) == 35
    assert int(layout.apply(starts[0] + 1, starts[1] + 3, shape=tensor.shape)["m"]) == 54
    assert save_json(tensor) == before
    compact = decl_tensor((8, 16), "float32", strides=(16, 1))
    assert_structural_equal(compact.ty.effective_layout(), layout)
    strided = decl_tensor((8, 16), "float32", strides=(32, 1))
    with pytest.raises(ValueError, match="noncompact"):
        strided.ty.effective_layout()
    with pytest.raises(ValueError, match="explicit layout"):
        decl_tensor((8, 16), "float32", scope="shared").ty.effective_layout()
    explicit = decl_tensor((8, 16), "float32", scope="shared", layout=layout)
    assert explicit.ty.effective_layout().same_as(layout)


def test_none_layout_views_keep_offsets_and_strides():
    builder = IRBuilder()
    with builder:
        with I.function():
            source = I.bind(
                I.decl_tensor(
                    I.ptr("float32"), ty_args=[I.Tensor((4, 8), "float32", elem_offset=5)]
                )
            )
            reshaped = source.view(2, 2, 8)
            permuted = source.permute(1, 0)
            sliced = source.sub[1:3, 2:6]
            row = source.sub[2, :]
            stepped = source.sub[:, 1::2]
            for view in (reshaped, permuted, sliced, row, stepped):
                assert view.layout is None
            assert list(map(int, reshaped.shape)) == [2, 2, 8]
            assert list(map(int, permuted.strides)) == [1, 8]
            assert list(map(int, sliced.strides)) == [8, 1]
            assert int(sliced.elem_offset) == 15
            assert int(row.elem_offset) == 21
            assert list(map(int, stepped.strides)) == [8, 2]
            assert int(stepped.elem_offset) == 6
            with pytest.raises(ValueError, match="non-contiguous"):
                permuted.view(32)
    parents = {bind.var: bind.value.args[0].args[0] for bind in builder.get().body.seq[1:]}
    for view in (reshaped, permuted, sliced, row, stepped):
        root = view
        while root in parents:
            root = parents[root]
        assert root.same_as(source)


@pytest.mark.parametrize(
    "dtype,new_dtype,old_shape,new_shape",
    [
        ("float32", "float16", (2, 8), (2, 16)),
        ("float16", "float32", (2, 8), (2, 4)),
        ("uint8", "float4_e2m1fn", (2, 8), (2, 16)),
    ],
)
def test_none_layout_dtype_views(dtype, new_dtype, old_shape, new_shape):
    builder = IRBuilder()
    with builder:
        with I.function():
            source = I.bind(I.decl_tensor(I.ptr(dtype), ty_args=[I.Tensor(old_shape, dtype)]))
            viewed = source.view(new_dtype)
            assert viewed.layout is None
            assert list(map(int, viewed.shape)) == list(new_shape)
    assert_structural_equal(builder.get().body.seq[-1].value.args[0], source.data)


def test_none_layout_dtype_view_accepts_symbolic_extent():
    with IRBuilder():
        with I.function():
            size = tvm_tirx.tirx.Var("size", "int32")
            source = I.bind(I.decl_tensor(I.ptr("float32"), ty_args=[I.Tensor((size,), "float32")]))
            viewed = source.view("uint64")
            assert viewed.layout is None
            assert_structural_equal(viewed.shape[0], size // 2)
            misaligned = I.bind(
                I.decl_tensor(I.ptr("float32"), ty_args=[I.Tensor((8,), "float32", elem_offset=1)])
            )
            with pytest.raises(ValueError, match="align"):
                misaligned.view("uint64")


def test_tile_call_rejects_cross_backend_scope():
    tensor = decl_tensor((8,), "float32", scope="local", layout="default")
    with pytest.raises(ValueError, match="scope belongs to trn"):
        T.cuda.tile.apply(
            T.ptx["add.f32"], tensor[:], tensor[:], tensor[:], scope=T.trn.ThreadScope("thread")
        )


def test_none_layout_script_roundtrip():
    @T.function
    def kernel(A: T.Tensor((8,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tmp = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32", scope="local")])
        tmp[0] = A[0]

    assert kernel.params[0].ty.layout is None
    assert_structural_equal(
        kernel, tvm_tirx.script.from_source(kernel.script(), extra_vars={"T": T, "I": tvm_tirx.script.ir})
    )


@pytest.mark.skipif(not env.has_cuda(), reason="requires CUDA")
def test_none_layout_views_execute_with_offsets_and_strides():
    @T.function
    def kernel(A: T.Tensor((4, 8), "float32"), D: T.Tensor((3, 16), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        transposed = A.permute(1, 0)
        window = A.sub[1:3, 2:6]
        stepped = A.sub[:, 1::2]
        reshaped = A.view(2, 2, 8)
        for i in range(16):
            D[0, i] = transposed[i // 4, i % 4]
            D[1, i] = stepped[i // 4, i % 4]
            D[2, i] = reshaped[1, i // 8, i % 8]
        for i, j in T.grid(2, 4):
            D[2, i * 4 + j] = window[i, j]

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()})
    compiled = tvm_tirx.tirx.compile(kernel, target=target, pipeline="tirx")
    data = np.arange(32, dtype="float32").reshape(4, 8)
    expected = np.stack([data.T[:4].flatten(), data[:, 1::2].flatten(), data[2:].flatten()])
    expected[2, :8] = data[1:3, 2:6].flatten()

    def run():
        source = tvm_tirx.runtime.tensor(data, device=tvm_tirx.cuda(0))
        output = tvm_tirx.runtime.tensor(np.zeros((3, 16), dtype="float32"), device=tvm_tirx.cuda(0))
        compiled(source, output)
        np.testing.assert_array_equal(output.numpy(), expected)

    tvm_tirx.testing.run_with_gpu_lock(run)


@pytest.mark.skipif(not env.has_cuda(), reason="requires CUDA")
def test_tile_load_infers_full_global_layout_with_offset():
    @T.function
    def kernel(A: T.Tensor((144,), "float32"), D: T.Tensor((2, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        source = T.decl_tensor(A.data, ty_args=[T.Tensor((8, 16), "float32", elem_offset=7)])
        registers = T.alloc_local((2, 4), "float32", layout="default")
        source_words = source.sub[2:4, 3:7].view(
            2, 2, 2, layout=TileLayout(S[(2, 2, 2) : (16, 2, 1)])
        )
        register_words = registers.view(2, 2, 2)
        output_words = D.view(2, 2, 2)
        T.cuda.tile.apply(T.ptx["ld.global.v2.b32"], register_words, source_words)
        T.cuda.tile.apply(T.ptx["st.global.v2.b32"], output_words, register_words)

    assert kernel.params[0].ty.layout is None
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()})
    compiled = tvm_tirx.tirx.compile(kernel, target=target, pipeline="tirx")
    data = np.arange(144, dtype="float32")

    def run():
        source = tvm_tirx.runtime.tensor(data, device=tvm_tirx.cuda(0))
        output = tvm_tirx.runtime.tensor(np.zeros((2, 4), dtype="float32"), device=tvm_tirx.cuda(0))
        compiled(source, output)
        np.testing.assert_array_equal(output.numpy(), data[7:135].reshape(8, 16)[2:4, 3:7])

    tvm_tirx.testing.run_with_gpu_lock(run)
