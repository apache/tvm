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
"""Unittests for tvm.script.parser.tirx"""

import pytest

import tvm.testing
from tvm import ir, tirx
from tvm.script.parser import tirx as T


def test_tir_buffer_annotation():
    buffer_0 = T.Buffer((128, 128), "float32")
    assert (
        tirx.is_buffer_var(buffer_0)
        and list(buffer_0.shape) == [128, 128]
        and buffer_0.dtype == ir.PrimType("float32")
    )

    buffer_1 = T.Buffer((64, 64, 64), "int32")
    assert (
        tirx.is_buffer_var(buffer_1)
        and list(buffer_1.shape) == [64, 64, 64]
        and buffer_1.dtype == ir.PrimType("int32")
    )


def test_tir_bound_prim_param_reused_in_dependent_annotations():
    func = tvm.script.from_source(
        """
@T.prim_func
def main(
    n: T.int32,
    direct: T.Buffer((n,), "float32"),
    repeated: T.Buffer((n,), "float32"),
    compound: T.Buffer((n + 1,), "float32"),
) -> T.Buffer((n,), "float32"):
    return repeated
"""
    )

    n, direct, repeated, compound = func.params
    assert direct.ty.shape[0].same_as(n)
    assert repeated.ty.shape[0].same_as(n)
    assert compound.ty.shape[0].a.same_as(n)
    assert func.ret_type.shape[0].same_as(n)


def test_tir_bound_prim_param_reused_in_declared_function_signature():
    mod = tvm.script.from_source(
        """
@I.ir_module
class Module:
    @T.prim_func
    def main(n: T.int32, A: T.Buffer((n + 1,), "float32")):
        T.evaluate(n)
"""
    )

    n, A = mod["main"].params
    assert A.ty.shape[0].a.same_as(n)


def test_tir_external_symbol_adopted_by_later_prim_param():
    func = tvm.script.from_source(
        """
n = T.dynamic("n", "int32")
@T.prim_func
def main(A: T.Buffer((n,), "float32"), n: n):
    T.evaluate(n)
"""
    )

    A, n = func.params
    assert A.ty.shape[0].same_as(n)
    assert str(n.ty.dtype) == "int32"

    mod = tvm.script.from_source(
        """
n = T.dynamic("n", "int32")
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), n: n):
        T.evaluate(n)
"""
    )

    A, n = mod["main"].params
    assert A.ty.shape[0].same_as(n)
    assert str(n.ty.dtype) == "int32"


def test_tir_external_symbol_preserves_later_prim_param_dtype():
    func = tvm.script.from_source(
        """
n = T.dynamic("n", "int64")
@T.prim_func
def main(A: T.Buffer((n,), "float32"), n: n):
    T.evaluate(n)
"""
    )

    A, n = func.params
    assert A.ty.shape[0].same_as(n)
    assert str(n.ty.dtype) == "int64"


def test_tir_external_dynamic_symbol_preserves_dtype():
    func = tvm.script.from_source(
        """
n = T.dynamic("n", "int64")
@T.prim_func
def main(A: T.Buffer((n,), "float32")):
    T.evaluate(n)
"""
    )

    n = func.params[0].ty.shape[0]
    assert str(n.ty.dtype) == "int64"
    assert func.body.value.same_as(n)


def test_tir_undeclared_shape_symbol_is_undefined():
    with pytest.raises(NameError):
        tvm.script.from_source(
            """
@T.prim_func
def main(A: T.Buffer((n, n), "float32")):
    T.evaluate(0)
"""
        )


@pytest.mark.parametrize(
    "source",
    [
        """
@T.prim_func
def main(A: T.Buffer((n,), "float32"), n: T.int32):
    T.evaluate(n)
""",
        """
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), n: T.int32):
        T.evaluate(n)
""",
    ],
)
def test_tir_direct_later_prim_param_is_undefined(source):
    with pytest.raises(NameError):
        tvm.script.from_source(source)


def test_tir_return_annotation_does_not_define_symbolic_var():
    with pytest.raises(NameError):
        tvm.script.from_source(
            """
@T.prim_func
def main() -> T.Buffer((n,), "float32"):
    A = T.alloc_buffer((n,), "float32")
    return A
"""
        )


def test_tir_ptr_proxy():
    ptr_0 = T.handle("int32", "global")
    assert (
        isinstance(ptr_0, tirx.Var)
        and isinstance(ptr_0.ty, ir.PointerType)
        and ptr_0.ty.element_type == ir.PrimType("int32")
        and ptr_0.ty.storage_scope == "global"
    )

    ptr_1 = T.handle("float32", "shared")
    assert (
        isinstance(ptr_1, tirx.Var)
        and isinstance(ptr_1.ty, ir.PointerType)
        and ptr_1.ty.element_type == ir.PrimType("float32")
        and ptr_1.ty.storage_scope == "shared"
    )


if __name__ == "__main__":
    tvm.testing.main()
