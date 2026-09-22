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
"""Construction regressions exercised by ordinary TVMScript entry points."""

import pytest

import tvm
from tvm.script import ir as I
from tvm.script import parser
from tvm.script import relax as R


def test_eager_module_annotations_resolve_global_info_in_module_scope():
    @I.ir_module
    class Module:
        I.module_global_infos(
            {
                "mesh": [R.device_mesh((2,), I.Range(0, 2))],
                "vdevice": [I.vdevice("llvm")],
            }
        )

        @R.function
        def distributed(x: R.DTensor((4,), "float32", "mesh[0]", "R")):
            return x

        @R.function
        def tensor(x: R.Tensor((4,), "float32", vdevice="llvm:0:global")):
            return x

    assert (
        Module["distributed"].params[0].ty.device_mesh.__chandle__()
        == Module.global_infos["mesh"][0].__chandle__()
    )
    assert (
        Module["tensor"].params[0].ty.vdevice.__chandle__()
        == Module.global_infos["vdevice"][0].__chandle__()
    )


def test_scoped_vdevice_selects_target_ordinal_and_roundtrips():
    module = parser.parse("""
@I.ir_module
class Module:
    I.module_global_infos({"vdevice": [
        I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("metal"), I.vdevice("cuda", 1),
    ]})
    @R.function
    def main(x: R.Tensor((4,), "float32", vdevice="cuda:1:global")):
        return x
""")
    assert (
        module["main"].params[0].ty.vdevice.__chandle__()
        == module.global_infos["vdevice"][3].__chandle__()
    )
    tvm.ir.assert_structural_equal(module, tvm.script.from_source(module.script()))


@pytest.mark.parametrize("constructor", ["Buffer", "match_buffer"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_implicit_buffer_strides_use_symbolic_default(constructor, index_dtype):
    if constructor == "Buffer":
        source = """
@T.prim_func(s_tir=True)
def main(A: T.Buffer((16, 16), "float32", strides=("s0", "s1"))):
    T.evaluate(A[0, 0])
"""
    else:
        source = """
@T.prim_func(s_tir=True)
def main(a: T.handle):
    A = T.match_buffer(a, (16, 16), "float32", strides=("s0", "s1"))
    T.evaluate(A[0, 0])
"""
    source = source.replace("(16, 16)", f"(T.{index_dtype}(16), T.{index_dtype}(16))")
    function = parser.parse(source)
    assert [str(value.ty.dtype) for value in function.params[0].strides] == ["int64", "int64"]


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_explicit_stride_declaration_keeps_its_dtype_and_identity(dtype):
    function = parser.parse(f"""
@T.prim_func(s_tir=True)
def main(A: T.Buffer((16, 16), "float32", strides=("s0", "s1")), s0: T.{dtype}):
    s1 = T.int64()
    T.evaluate(A[0, 0] + s0 + s1)
""")
    buffer, stride = function.params
    assert buffer.strides[0].same_as(stride)
    assert [str(value.ty.dtype) for value in buffer.strides] == [dtype, "int64"]


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_match_buffer_reuses_prior_explicit_stride(dtype):
    function = parser.parse(f"""
@T.prim_func(s_tir=True)
def main(a: T.handle):
    s0 = T.{dtype}()
    A = T.match_buffer(a, (16,), "float32", strides=("s0",))
    T.evaluate(s0)
""")
    buffer = function.params[0]
    stride_use = function.body.value
    assert buffer.strides[0].same_as(stride_use)
    assert str(buffer.strides[0].ty.dtype) == dtype


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_decl_buffer_stride_roundtrip_preserves_dtype(dtype):
    # Standalone buffer declarations can contain free stride symbols.
    function = parser.parse(
        f"""
@T.prim_func(s_tir=True)
def main(data: T.handle("float32")):
    s0 = T.{dtype}()
    A = T.decl_buffer((1,), "float32", data=data, strides=(s0,))
    T.evaluate(A[0])
""",
        check_well_formed=False,
    )
    printed = function.script()
    if dtype == "int64":
        assert 'strides=("s0",)' in printed
    else:
        assert "s0 = T.int32()" in printed
    tvm.ir.assert_structural_equal(
        function, parser.parse(printed, check_well_formed=False), map_free_vars=True
    )


@pytest.mark.parametrize(
    "axes",
    [
        "vi = T.axis.spatial(16, i)\n            vi = T.axis.spatial(16, j)",
        'vi, vi = T.axis.remap("SS", [i, j])',
        'vi, vj = T.axis.remap("SS", [i, j])\n            vi = T.axis.spatial(16, j)',
    ],
)
def test_duplicate_block_axis_source_name_is_rejected(axes):
    source = f"""
@T.prim_func(s_tir=True)
def main():
    for i, j in T.grid(16, 16):
        with T.sblock("block"):
            {axes}
            T.evaluate(vi)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        parser.parse(source)
