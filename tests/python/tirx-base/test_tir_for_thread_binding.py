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
"""Semantic thread binding stored in For annotations."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import tirx
from tvm.script import tirx as T
from tvm.testing import env


def make_binding(tag="threadIdx.x", dom=None, var=None):
    return tirx.IterVar(dom, var if var is not None else "thread", tirx.IterVar.ThreadIndex, tag)


def make_loop(binding=None, annotations=None, kind=tirx.ForKind.PARALLEL, extent=8):
    i = tirx.Var("i", "int32")
    return tirx.For(i, 0, extent, kind, tirx.Evaluate(i), binding, annotations)


def test_annotation_representation_and_classification():
    binding = make_binding()
    loop = make_loop(binding, {"custom_hint": 7})
    assert loop.kind == tirx.ForKind.PARALLEL
    assert loop.annotations["thread_binding"].same_as(binding)
    assert loop.thread_binding.same_as(binding)
    assert int(loop.annotations["custom_hint"]) == 7
    assert loop.is_thread_binding()
    assert not loop.is_parallel()

    annotated = make_loop(annotations={"thread_binding": binding})
    assert annotated.thread_binding.same_as(binding)
    assert annotated.is_thread_binding()
    parallel = make_loop()
    assert parallel.thread_binding is None
    assert parallel.is_parallel()
    assert not parallel.is_thread_binding()
    serial = make_loop(kind=tirx.ForKind.SERIAL)
    assert not serial.is_parallel()
    assert not serial.is_thread_binding()


@pytest.mark.parametrize("value", ["threadIdx.x", 1])
def test_reject_invalid_annotation_value(value):
    with pytest.raises(TypeError, match="must be an IterVar"):
        make_loop(annotations={"thread_binding": value})


@pytest.mark.parametrize("via_annotation", [False, True])
def test_reject_empty_thread_tag(via_annotation):
    binding = make_binding("")
    with pytest.raises(ValueError, match="nonempty thread tag"):
        if via_annotation:
            make_loop(annotations={"thread_binding": binding})
        else:
            make_loop(binding)


@pytest.mark.parametrize(
    "kind", [tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED, tirx.ForKind.VECTORIZED]
)
@pytest.mark.parametrize("via_annotation", [False, True])
def test_reject_binding_on_nonparallel_loop(kind, via_annotation):
    binding = make_binding()
    with pytest.raises(ValueError, match="only valid on parallel loops"):
        if via_annotation:
            make_loop(annotations={"thread_binding": binding}, kind=kind)
        else:
            make_loop(binding, kind=kind)


def test_reject_removed_thread_binding_kind():
    with pytest.raises(ValueError, match="Invalid ForKind"):
        make_loop(make_binding(), kind=4)


def test_reject_conflicting_binding_sources():
    with pytest.raises(ValueError, match="Conflicting thread_binding"):
        make_loop(make_binding(), {"thread_binding": make_binding("blockIdx.x")})


def test_serialization_and_structural_identity():
    binding = make_binding(dom=tvm.ir.Range.from_min_extent(2, 8))
    loop = make_loop(binding)
    restored = tvm.ir.load_json(tvm.ir.save_json(loop))
    tvm.ir.assert_structural_equal(restored, loop, map_free_vars=True)
    assert tvm_ffi.structural_hash(restored, map_free_vars=True) == tvm_ffi.structural_hash(
        loop, map_free_vars=True
    )
    assert int(restored.thread_binding.dom.min) == 2
    assert int(restored.thread_binding.dom.extent) == 8
    assert restored.thread_binding.iter_type == tirx.IterVar.ThreadIndex
    assert restored.thread_binding.thread_tag == "threadIdx.x"
    assert restored.thread_binding.var.name == binding.var.name
    assert restored.annotations["thread_binding"].same_as(restored.thread_binding)
    via_annotation = make_loop(annotations={"thread_binding": binding})
    tvm.ir.assert_structural_equal(loop, via_annotation, map_free_vars=True)
    assert not tvm_ffi.structural_equal(loop, make_loop(), map_free_vars=True)
    assert not tvm_ffi.structural_equal(
        loop, make_loop(make_binding("blockIdx.x", binding.dom)), map_free_vars=True
    )
    assert not tvm_ffi.structural_equal(
        loop, make_loop(make_binding(dom=tvm.ir.Range.from_min_extent(3, 8))), map_free_vars=True
    )


@pytest.mark.parametrize("move", [False, True])
def test_structural_walk_and_mutation_reach_binding_metadata(move):
    extent = tirx.Var("extent", "int32")
    thread_var = tirx.Var("thread", "int32")
    replacement = tirx.Var("new_thread", "int32")
    binding = make_binding(dom=tvm.ir.Range.from_min_extent(2, extent), var=thread_var)
    loop = make_loop(binding, {"custom_hint": extent})
    visited = []
    tvm_ffi.structural_walk(loop, visited.append)
    assert any(node.same_as(binding) for node in visited)
    assert any(node.same_as(thread_var) for node in visited)
    assert any(node.same_as(extent) for node in visited)

    def rewrite(var):
        if var.same_as(thread_var):
            return replacement
        if var.same_as(extent):
            return tirx.IntImm("int32", 16)
        return var

    visited.clear()  # Do not retain the root while exercising ownership transfer.
    rewritten = tvm_ffi.structural_map(
        loop._move() if move else loop, (tirx.Var, rewrite), order="post"
    )
    assert rewritten.thread_binding.var.same_as(replacement)
    assert int(rewritten.thread_binding.dom.min) == 2
    assert int(rewritten.thread_binding.dom.extent) == 16
    assert rewritten.thread_binding.iter_type == binding.iter_type
    assert rewritten.thread_binding.thread_tag == binding.thread_tag
    # Auxiliary hints remain opaque; only the semantic annotation is traversed.
    assert rewritten.annotations["custom_hint"].same_as(extent)
    if not move:
        assert loop.thread_binding.var.same_as(thread_var)
        assert loop.thread_binding.dom.extent.same_as(extent)


def test_script_roundtrip_keeps_hints_separate():
    @T.prim_func
    def before():
        for i in T.thread_binding(8, thread="threadIdx.x", annotations={"custom_hint": 7}):
            T.evaluate(i)

    script = before.script()
    assert 'annotations={"thread_binding"' not in script
    restored = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(before, restored)
    assert restored.body.is_thread_binding()
    assert int(restored.body.annotations["custom_hint"]) == 7


@pytest.mark.parametrize("tag", ["threadIdx.x", "blockIdx.x", "vthread.x"])
@pytest.mark.parametrize("extent", [1, 8])
def test_gpu_binding_lowers_to_thread_extent(tag, extent):
    loop = make_loop(make_binding(tag), extent=extent)
    mod = tvm.IRModule.from_expr(tirx.PrimFunc([], loop))
    lowered = tirx.transform.LowerTIRxOpaque()(mod)["main"].body
    assert isinstance(lowered, tirx.AttrStmt)
    assert lowered.attr_key == ("virtual_thread" if tag == "vthread.x" else "thread_extent")
    assert lowered.node.thread_tag == tag
    assert int(lowered.value) == extent
    if extent == 1:
        assert int(lowered.body.value) == 0
    else:
        assert lowered.body.value.same_as(lowered.node.var)


@pytest.mark.parametrize(
    "config",
    [
        {"auto_max_extent": 1},
        {"auto_max_step": 16, "auto_max_depth": 8, "explicit_unroll": True},
    ],
)
def test_unit_thread_binding_survives_unroll(config):
    loop = make_loop(make_binding(), extent=1)
    mod = tvm.IRModule.from_expr(tirx.PrimFunc([], loop))
    with tvm.transform.PassContext(config={"tirx.UnrollLoop": config}):
        transformed = tirx.transform.UnrollLoop()(mod)["main"].body
    assert isinstance(transformed, tirx.For)
    assert transformed.is_thread_binding()
    assert int(transformed.extent) == 1
    tvm.ir.assert_structural_equal(transformed, loop)


def test_software_pipeline_cannot_discard_thread_scope():
    loop = make_loop(
        make_binding(),
        {"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]},
    )
    mod = tvm.IRModule.from_expr(tirx.PrimFunc([], loop))
    with pytest.raises(ValueError, match="cannot replace a thread-bound loop"):
        tvm.s_tir.transform.InjectSoftwarePipeline()(mod)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_cpu_parallel_execution():
    @T.prim_func
    def before(out: T.Buffer((16,), "int32")):
        for i in T.parallel(16):
            out[i] = i + 3

    assert before.body.is_parallel()
    compiled = tvm.compile(before.with_attr("global_symbol", "main"), target="llvm")
    output = np.zeros(16, dtype="int32")
    compiled(output)
    np.testing.assert_array_equal(output, np.arange(16, dtype="int32") + 3)


if __name__ == "__main__":
    tvm.testing.main()
