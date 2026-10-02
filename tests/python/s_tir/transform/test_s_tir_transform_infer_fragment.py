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
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import tirx as T


def allocations(func):
    result = []

    def collect(node):
        if isinstance(node, tvm.tirx.AttrStmt):
            assert node.attr_key not in ("fragment_shape", "fragment_layout")
        if (
            isinstance(node, tvm.tirx.Bind)
            and isinstance(node.value, tvm.ir.Call)
            and node.value.op == tvm.ir.Op.get("tirx.alloc_buffer")
        ):
            result.append(node)

    tvm_ffi.structural_walk(func.body, collect)
    return result


def infer(func):
    return tvm.s_tir.transform.InferFragment()(tvm.IRModule.from_expr(func))["main"]


def make_fragment(annotations=None):
    @T.prim_func(private=True)
    def main():
        A = T.alloc_buffer((256,), "float16", scope="wmma.matrix_a", annotations=annotations)
        T.tvm_fill_fragment(A.data, 16, 16, 16, 0, T.float16(0))

    return main


def test_fragment_attributes_and_roundtrip():
    before = make_fragment({"preserved": T.int32(7)})
    old_binding = allocations(before)[0]
    after = infer(before)
    binding = allocations(after)[0]
    assert binding.value.attrs["fragment_shape"] == "16, 16, 16"
    assert binding.value.attrs["fragment_layout"] == "row_major"
    assert binding.value.attrs["preserved"].value == 7
    assert "fragment_shape" not in old_binding.value.attrs
    assert binding.var.same_as(old_binding.var)
    assert binding.value.ty.same_as(old_binding.value.ty)
    assert binding.span.same_as(old_binding.span)
    assert binding.value.span == old_binding.value.span
    tvm.ir.assert_structural_equal(infer(after), after)
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(after.script(), extra_vars={"T": T, "I": tvm.script.ir}), after
    )
    tvm.ir.assert_structural_equal(tvm.ir.load_json(tvm.ir.save_json(after)), after)


@pytest.mark.parametrize(
    "annotations",
    [
        {"fragment_shape": "8, 32, 16"},
        {"fragment_layout": "col_major"},
        {"fragment_shape": 7},
    ],
)
def test_conflicting_fragment_attributes(annotations):
    with pytest.raises(ValueError, match="Conflicting fragment_"):
        infer(make_fragment(annotations))


def test_equal_fragment_attributes():
    before = make_fragment({"fragment_shape": "16, 16, 16", "fragment_layout": "row_major"})
    tvm.ir.assert_structural_equal(infer(before), before)


def test_nested_fragment_owners_and_renewal():
    @T.prim_func(private=True)
    def main(n: T.int32):
        A = T.alloc_buffer((n * 256,), "float32", scope="wmma.accumulator")
        T.tvm_fill_fragment(A.data, 16, 16, 16, 0, T.float32(0))
        if n > 0:
            B = T.alloc_buffer((256,), "float32", scope="wmma.accumulator")
            T.tvm_fill_fragment(B.data, 8, 32, 16, 0, T.float32(0))

    after = infer(main)
    for candidate in [
        after,
        tvm.s_tir.renew_defs(after),
        tvm.tirx.transform.ConvertSSA()(tvm.IRModule.from_expr(after))["main"],
        tvm.tirx.transform.FlattenBuffer()(tvm.IRModule.from_expr(after))["main"],
    ]:
        assert [node.value.attrs["fragment_shape"] for node in allocations(candidate)] == [
            "16, 16, 16",
            "8, 32, 16",
        ]
    assert not allocations(tvm.s_tir.renew_defs(after))[0].var.same_as(allocations(after)[0].var)


def test_storage_rewrite_after_inference():
    before = infer(make_fragment({"preserved": T.int32(7)}))
    mod = tvm.tirx.transform.StorageRewrite()(tvm.IRModule.from_expr(before))
    after = mod["main"]
    assert len(allocations(after)) == 1
    tvm.ir.assert_structural_equal(allocations(after)[0].value, allocations(before)[0].value)
    tvm.ir.assert_structural_equal(tvm.tirx.transform.StorageRewrite()(mod), mod)


@pytest.mark.parametrize("prepare_pipeline", [False, True])
def test_cuda_fragment_codegen(monkeypatch, prepare_pipeline):
    build = tvm.get_global_func("target.build.cuda", allow_missing=True)
    if build is None:
        pytest.skip("CUDA codegen is unavailable")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    func = make_fragment().with_attr("global_symbol", "main").with_attr("target", target)
    if prepare_pipeline:
        prepare, _, _ = tvm.s_tir.pipeline.default_s_tir_pipeline(prepare_only=True)
        func = prepare(tvm.IRModule.from_expr(func))["main"]
    else:
        func = infer(func)
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    source = build(tvm.IRModule.from_expr(func), target).inspect_source()
    assert (
        "wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>"
        in source
    )
    assert "[1]" in source


def test_conflicting_intrinsic_observations():
    @T.prim_func(private=True)
    def main():
        A = T.alloc_buffer((256,), "float32", scope="wmma.accumulator")
        T.tvm_fill_fragment(A.data, 16, 16, 16, 0, T.float32(0))
        T.tvm_fill_fragment(A.data, 8, 32, 16, 0, T.float32(0))

    with pytest.raises(tvm.error.InternalError):
        infer(main)


@pytest.mark.parametrize("annotations", [{}, {"fragment_layout": "col_major"}])
def test_accumulator_without_layout(annotations):
    @T.prim_func(private=True)
    def main(output: T.Buffer((256,), "float32")):
        A = T.alloc_buffer((256,), "float32", scope="wmma.accumulator", annotations=annotations)
        T.tvm_store_matrix_sync(A.data, 16, 16, 16, 0, output.data, 16, "row_major")

    after = infer(main)
    assert allocations(after)[0].value.attrs["fragment_shape"] == "16, 16, 16"
    assert allocations(after)[0].value.attrs.get("fragment_layout") == annotations.get(
        "fragment_layout"
    )


def test_spirv_fragment_codegen(monkeypatch):
    build = tvm.get_global_func("target.build.vulkan", allow_missing=True)
    if build is None:
        pytest.skip("SPIR-V codegen is unavailable")
    target = tvm.target.Target({"kind": "vulkan", "supports_cooperative_matrix": True})

    @T.prim_func(private=True)
    def main():
        A = T.alloc_buffer((256,), "float32", scope="wmma.accumulator")
        T.tvm_fill_fragment(A.data, 16, 16, 16, 0, T.float32(0))

    func = (
        infer(main)
        .with_attr("global_symbol", "main")
        .with_attr("calling_conv", 2)
        .with_attr("tirx.noalias", True)
    )
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    source = build(tvm.IRModule.from_expr(func), target).inspect_source()
    assert "OpTypeCooperativeMatrixNV" in source


if __name__ == "__main__":
    tvm.testing.main()
