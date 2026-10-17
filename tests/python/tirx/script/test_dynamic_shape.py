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

"""TIRx script dynamic shape."""

from __future__ import annotations

import sys

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_tir_bound_prim_param_reused_in_dependent_annotations():
    @T.function
    def func(
        n: T.int32,
        direct: T.Tensor((n,), "float32"),
        repeated: T.Tensor((n,), "float32"),
        compound: T.Tensor((n + 1,), "float32"),
    ) -> T.Tensor((n,), "float32"):
        return repeated

    n, direct, repeated, compound = func.params
    assert direct.ty.shape[0].same_as(n)
    assert repeated.ty.shape[0].same_as(n)
    assert compound.ty.shape[0].a.same_as(n)
    assert func.ret_type.shape[0].same_as(n)


def test_tir_bound_prim_param_reused_in_declared_function_signature():
    @I.ir_module
    class Module:
        @T.function
        def main(n: T.int32, A: T.Tensor((n + 1,), "float32")):
            T.evaluate(n)

    n, A = Module["main"].params
    assert A.ty.shape[0].a.same_as(n)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_tir_external_symbol_adopted_by_later_prim_param(dtype):
    n = T.dynamic("n", dtype)

    @T.function
    def func(A: T.Tensor((n,), "float32"), n: n):
        T.evaluate(n)

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), n: n):
            T.evaluate(n)

    for function in [func, Module["main"]]:
        A, n = function.params
        assert A.ty.shape[0].same_as(n)
        assert str(n.ty.dtype) == dtype


def test_tir_external_dynamic_symbol_preserves_dtype():
    n = T.dynamic("n", "int64")

    @T.function
    def func(A: T.Tensor((n,), "float32")):
        T.evaluate(n)

    n = func.params[0].ty.shape[0]
    assert str(n.ty.dtype) == "int64"
    assert func.body[0].value.same_as(n)


def test_tir_undeclared_shape_symbol_is_undefined():
    with pytest.raises(NameError):

        @T.function
        def main(A: T.Tensor((n, n), "float32")):  # noqa: F821
            T.evaluate(0)


def test_tir_direct_later_prim_param_reuses_shape_symbol():
    @T.function
    def func(A: T.Tensor((n,), "float32"), n: T.int32):
        T.evaluate(n)

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), n: T.int32):
            T.evaluate(n)

    for function in [func, Module["main"]]:
        A, n = function.params
        assert A.ty.shape[0].same_as(n)
        assert str(n.ty.dtype) == "int32"
        assert function.body[0].value.same_as(n)


def test_tir_return_annotation_does_not_define_symbolic_var():
    with pytest.raises(NameError):

        @T.function
        def main() -> T.Tensor((n,), "float32"):  # noqa: F821
            A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])  # noqa: F821
            return A


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_type_vars_roundtrip():
    M = T.dynamic("M")
    UNUSED = T.dynamic("UNUSED")

    @T.function(private=True)
    def func(A: T.Tensor((M, M * 2), "float32")):
        A[0, 0] = T.float32(1)

    script = func.script()
    assert script.startswith("from __future__ import annotations\n\n")
    assert "def main[M](" in script
    assert "T.Tensor((M, M * 2), T.float32)" in script
    assert "M = T.int32()" not in script
    typed = tvm_tirx.script.from_source(
        """
@T.function(private=True)
def func[M: int](A: T.Tensor((M, M * 2), T.float32)):
    A[0, 0] = T.float32(1)
""",
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
    )
    tvm_tirx.ir.assert_structural_equal(func, typed)
    assert 'M = T.dynamic("M", dtype="int32")' not in script

    repeated = func.script()
    assert "from __future__ import annotations" in repeated
    assert "def main[M](" in repeated
    assert "T.Tensor((M, M * 2), T.float32)" in repeated
    assert "UNUSED" not in script
    assert "M = T.int32()" not in repeated
    assert len(func.params) == 1
    assert not hasattr(func, "type_params")
    assert func.attrs.get("tirx.type_vars") is None
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            script, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
        ),
    )
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            repeated, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
        ),
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_dynamic_int32_roundtrip():
    n = T.dynamic("n", "int32")

    @T.function(private=True)
    def func(A: T.Tensor((n,), "float32")):
        A[0] = T.float32(1)

    source = func.script()
    assert "def main[n](" in source
    assert 'n = T.dynamic("n", dtype="int32")' not in source
    repeated = func.script()
    assert "def main[n](" in repeated
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            source, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
        ),
    )
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            repeated, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
        ),
    )


def test_dynamic_module_body_identity():
    n = T.dynamic("n", "int32")
    m = T.dynamic("n", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.function(private=True)
        def first():
            T.evaluate(n)

        @T.function(private=True)
        def second():
            T.evaluate(n + m)

    source = mod.script()
    assert source.count('T.dynamic("n", dtype="int32")') == 2
    restored = tvm_tirx.script.from_source(
        source,
        check_well_formed=False,
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
    )
    shared = restored["first"].body[0].value
    summed = restored["second"].body[0].value
    assert shared.same_as(summed.a)
    assert not shared.same_as(summed.b)
    tvm_tirx.ir.assert_structural_equal(mod, restored, map_free_vars=True)


def test_captured_shape_requires_concrete_symbols():
    # Native shape construction preserves concrete symbols and rejects strings.
    def build(shape):
        @T.function
        def main(x: T.Tensor(shape, "float32")):
            T.evaluate(0)

        return main

    n = T.dynamic("n")
    function = build((n, 16))
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(TypeError, match="Expected `Array<ir.PrimExpr>`"):
        build(("n", 16))


def test_dynamic_symbols_are_fresh_and_scope_independent():
    assert T.dynamic is I.dynamic
    n = T.dynamic("n")
    same_name = T.dynamic("n")
    k = T.dynamic("k", "int64")
    assert n.ty.dtype == "int32"
    assert k.ty.dtype == "int64"
    assert not n.same_as(same_name)

    @I.ir_module
    class Module:
        @T.function
        def first(x: T.Tensor((n,), "float32")):
            T.evaluate(n)

    assert Module["first"].params[0].ty.shape[0].same_as(n)
    assert Module["first"].body[0].value.same_as(n)


def test_tir_starred_shape_expression():
    dims = (128, 128)

    @T.function(private=True)
    def starred(A: T.Tensor([128, *dims], "int32")) -> None:
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1)

    @T.function(private=True)
    def non_starred(A: T.Tensor([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm_tirx.ir.assert_structural_equal(starred, non_starred)


def test_inferred_ty_with_dynamic_tensor():
    """The inferred Type may contain dynamic shapes"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @T.function
    def func(A: T.Tensor([M, N], "float32"), B: T.Tensor([M * N], "float32")):
        for i, j in T.grid(M, N):
            B[i * N + j] = A[i, j]

    M = tvm_tirx.tirx.Var("M", "int64")
    N = tvm_tirx.tirx.Var("N", "int64")
    expected = tvm_tirx.ir.FuncType(
        [
            tvm_tirx.tirx.Var(
                "tensor", tvm_tirx.tirx.TensorType([M, N], "float32", layout=None)
            ).ty,
            tvm_tirx.tirx.Var(
                "tensor", tvm_tirx.tirx.TensorType([M * N], "float32", layout=None)
            ).ty,
        ],
        tvm_tirx.ir.TupleType([]),
    )
    tvm_tirx.ir.assert_structural_equal(func.ty, expected)


def loop_extent_dependent():
    @T.function
    def loop_extent_dependent(A: T.Tensor([], dtype="int32")) -> None:
        for i in T.serial(0, 128):
            for j in T.serial(0, i):
                A[()] = A[()] + j

    return loop_extent_dependent


def undefined_shape_in_decl_tensor():
    # uninitialized var
    size = T.dynamic("size", "int32")

    @T.function(check_well_formed=False)
    def func():
        buf = T.alloc_tensor(ty_args=[T.Tensor([size], "float32")])
        T.evaluate(buf[0])

    return func


def undefined_stride_in_decl_tensor():
    # uninitialized var
    stride = T.dynamic("stride", "int32")

    @T.function(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_tensor(
            data_ptr,
            ty_args=[
                T.Tensor([1], "float32", layout=T.TileLayout(T.S[[1] : [stride]]), data=data_ptr)
            ],
        )
        T.evaluate(buf[0])

    return func


def undefined_elem_offset_in_decl_tensor():
    # uninitialized var
    elem_offset = T.dynamic("elem_offset", "int32")

    @T.function(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_tensor(
            data_ptr,
            ty_args=[
                T.Tensor(
                    [1], "float32", layout=T.TileLayout(T.S[[1]] + (elem_offset)), data=data_ptr
                )
            ],
        )
        T.evaluate(buf[0])

    return func


@pytest.mark.parametrize(
    "ir_generator",
    [
        loop_extent_dependent,
        undefined_shape_in_decl_tensor,
        undefined_stride_in_decl_tensor,
        undefined_elem_offset_in_decl_tensor,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_dynamic_shape(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)
