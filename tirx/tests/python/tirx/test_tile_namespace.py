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
from tvm_tirx.tirx import TensorType, Var
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
from tvm_tirx.tirx import Var, TensorType
from tvm_tirx.tirx.layout import Axis, S, TileLayout
assert Axis.m.backend is None
assert T.Axis.m.same_as(Axis.m)
assert TileLayout(S[8]).apply(3, shape=[8])["m"] == 3
assert Var("tensor", TensorType((8,), "float32")).layout is None
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
    assert Var("tensor", TensorType((4, 8), "float32", scope=scope)).layout is None
    explicit = T.Tensor((4, 8), "float32", scope=scope, layout="default")
    assert_structural_equal(explicit.layout, TileLayout.from_iters(TileLayout(S[4, 8]).shard))
    with IRBuilder():
        with I.function():
            tensor_type = I.Tensor((4, 8), "float32", scope=scope)
            allocated_value = I.alloc_tensor(ty_args=[tensor_type])
            allocated = tvm_tirx.ir.Var("allocated", allocated_value.ty)
            I.emit_(I.Bind(allocated, allocated_value))
            declared_value = I.decl_tensor(allocated.data, ty_args=[tensor_type])
            declared = tvm_tirx.ir.Var("declared", declared_value.ty)
            I.emit_(I.Bind(declared, declared_value))
            assert allocated.layout is None and declared.layout is None


@pytest.mark.parametrize(
    "shape,strides",
    [((8, 16), None), ((8, 16), (16, 1)), ((8, 1, 16), (16, 99, 1))],
)
def test_effective_layout_on_tensor_type(shape, strides):
    tensor_type = T.Tensor(
        shape, "float32", layout=T.TileLayout(T.S[shape:strides]) if strides else None
    )
    before = save_json(tensor_type)
    effective = tensor_type.effective_layout()
    assert_structural_equal(
        effective, TileLayout(S[shape] if strides is None else S[shape:strides])
    )
    assert save_json(tensor_type) == before


def test_effective_global_layout_preserves_source_and_region():
    tensor = Var("tensor", TensorType((8, 16), "float32", layout=T.TileLayout(T.S[(8, 16)] + (13))))
    before = save_json(tensor)
    region = tensor[2:4, 3:7]
    layout = region.source.ty.effective_layout()
    assert int(layout.apply(2, 3, shape=tensor.shape)["m"]) == 48
    starts = [r.min for r in region.region]
    assert int(layout.apply(*starts, shape=tensor.shape)["m"]) == 48
    assert int(layout.apply(starts[0] + 1, starts[1] + 3, shape=tensor.shape)["m"]) == 67
    assert save_json(tensor) == before
    compact = Var("tensor", TensorType((8, 16), "float32", layout=TileLayout(S[(8, 16) : (16, 1)])))
    assert int(compact.ty.effective_layout().apply(2, 3, shape=compact.shape)["m"]) == 35
    strided = Var("tensor", TensorType((8, 16), "float32", layout=TileLayout(S[(8, 16) : (32, 1)])))
    assert int(strided.ty.effective_layout().apply(2, 3, shape=strided.shape)["m"]) == 67
    with pytest.raises(ValueError, match="explicit layout"):
        Var("tensor", TensorType((8, 16), "float32", scope="shared")).ty.effective_layout()
    explicit = Var("tensor", TensorType((8, 16), "float32", scope="shared", layout=layout))
    assert explicit.ty.effective_layout().same_as(layout)


def test_none_layout_views_keep_offsets_and_strides():
    builder = IRBuilder()
    with builder:
        with I.function():
            source_value = I.decl_tensor(
                I.ptr("float32"),
                ty_args=[I.Tensor((4, 8), "float32", layout=T.TileLayout(T.S[(4, 8)] + (5)))],
            )
            source = tvm_tirx.ir.Var("source", source_value.ty)
            I.emit_(I.Bind(source, source_value))
            reshaped = source.view(2, 2, 8)
            permuted = source.permute(1, 0)
            sliced = source.sub[1:3, 2:6]
            row = source.sub[2, :]
            stepped = source.sub[:, 1::2]
            for view in (reshaped, permuted, sliced, row, stepped):
                assert view.layout is not None
            assert list(map(int, reshaped.shape)) == [2, 2, 8]
            assert list(map(int, [it.stride for it in permuted.layout.shard])) == [1, 8]
            assert list(map(int, [it.stride for it in sliced.layout.shard])) == [8, 1]
            assert (
                int(sliced.layout.apply(*([0] * len(sliced.shape)), shape=sliced.shape).get("m", 0))
                == 15
            )
            assert int(row.layout.apply(*([0] * len(row.shape)), shape=row.shape).get("m", 0)) == 21
            assert list(map(int, [it.stride for it in stepped.layout.shard])) == [8, 2]
            assert (
                int(
                    stepped.layout.apply(*([0] * len(stepped.shape)), shape=stepped.shape).get(
                        "m", 0
                    )
                )
                == 6
            )
            linear = permuted.view(32)
            for i in range(32):
                assert int(linear.layout.apply(i, shape=linear.shape)["m"]) == (
                    int(permuted.layout.apply(i // 4, i % 4, shape=permuted.shape)["m"])
                )
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
            source_value = I.decl_tensor(I.ptr(dtype), ty_args=[I.Tensor(old_shape, dtype)])
            source = tvm_tirx.ir.Var("source", source_value.ty)
            I.emit_(I.Bind(source, source_value))
            viewed = source.view(new_dtype)
            assert viewed.layout is None
            assert list(map(int, viewed.shape)) == list(new_shape)
    assert_structural_equal(builder.get().body.seq[-1].value.args[0], source.data)


def test_none_layout_dtype_view_accepts_symbolic_extent():
    with IRBuilder():
        with I.function():
            size = tvm_tirx.tirx.Var("size", "int32")
            source_value = I.decl_tensor(I.ptr("float32"), ty_args=[I.Tensor((size,), "float32")])
            source = tvm_tirx.ir.Var("source", source_value.ty)
            I.emit_(I.Bind(source, source_value))
            viewed = source.view("uint64")
            assert viewed.layout is None
            assert_structural_equal(viewed.shape[0], size // 2)
            misaligned_value = I.decl_tensor(
                I.ptr("float32"),
                ty_args=[I.Tensor((8,), "float32", layout=T.TileLayout(T.S[(8,)] + (1)))],
            )
            misaligned = tvm_tirx.ir.Var("misaligned", misaligned_value.ty)
            I.emit_(I.Bind(misaligned, misaligned_value))
            with pytest.raises(ValueError, match="align"):
                misaligned.view("uint64")


def test_tile_call_rejects_cross_backend_scope():
    tensor = Var("tensor", TensorType((8,), "float32", scope="local", layout="default"))
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
        kernel,
        tvm_tirx.script.from_source(kernel.script(), extra_vars={"T": T, "I": tvm_tirx.script.ir}),
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
        output = tvm_tirx.runtime.tensor(
            np.zeros((3, 16), dtype="float32"), device=tvm_tirx.cuda(0)
        )
        compiled(source, output)
        np.testing.assert_array_equal(output.numpy(), expected)

    tvm_tirx.testing.run_with_gpu_lock(run)


@pytest.mark.skipif(not env.has_cuda(), reason="requires CUDA")
def test_tile_load_infers_full_global_layout_with_offset():
    @T.function
    def kernel(A: T.Tensor((144,), "float32"), D: T.Tensor((2, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        source = T.decl_tensor(
            A.data, ty_args=[T.Tensor((8, 16), "float32", layout=T.TileLayout(T.S[(8, 16)] + (7)))]
        )
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
