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

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
""",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
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
def test_tir_direct_later_prim_param_reuses_shape_symbol(source):
    result = tvm.script.from_source(
        source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
    )
    function = result["main"] if isinstance(result, tvm.IRModule) else result
    A, n = function.params
    assert A.ty.shape[0].same_as(n)
    assert str(n.ty.dtype) == "int32"
    assert function.body.value.same_as(n)



def test_tir_return_annotation_does_not_define_symbolic_var():
    with pytest.raises(NameError):
        tvm.script.from_source(
            """
@T.prim_func
def main() -> T.Buffer((n,), "float32"):
    A = T.alloc_buffer((n,), "float32")
    return A
""",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
        )


def test_type_vars_roundtrip():
    func = tvm.script.from_source(
        """
M = I.dynamic("M")
UNUSED = I.dynamic("UNUSED")

@T.prim_func(private=True)
def func(A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )

    script = func.script()
    if sys.version_info >= (3, 12):
        assert script.startswith("from __future__ import annotations\n\n")
        assert "def main[M](" in script
        assert 'T.Buffer((M, M * T.int64(2)), "float32")' in script
        assert "M = T.int64()" not in script
        typed = tvm.script.from_source(
            """
@T.prim_func(private=True)
def func[M: int](A: T.Buffer((M, M * 2), "float32")):
    A[0, 0] = T.float32(1)
""",
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
        )
        tvm.ir.assert_structural_equal(func, typed)
    else:
        assert "from __future__ import annotations" not in script
        assert 'M = I.dynamic("M", dtype="int64")' in script
        assert "M = T.int64()" not in script

    portable = func.script(extra_config={"script.use_pep695": False})
    assert "from __future__ import annotations" not in portable
    assert 'M = I.dynamic("M", dtype="int64")' in portable
    assert 'T.Buffer((M, M * T.int64(2)), "float32")' in portable
    assert "UNUSED" not in script
    assert "M = T.int64()" not in portable
    assert len(func.params) == 1
    assert not hasattr(func, "type_params")
    assert func.attrs.get("tirx.type_vars") is None
    tvm.ir.assert_structural_equal(
        func, tvm.script.from_source(script, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    )
    tvm.ir.assert_structural_equal(
        func,
        tvm.script.from_source(portable, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}),
    )


def test_dynamic_int32_roundtrip():
    func = tvm.script.from_source(
        """
n = I.dynamic("n", "int32")
@T.prim_func(private=True)
def func(A: T.Buffer((n,), "float32")):
    A[0] = T.float32(1)
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )
    source = func.script()
    if sys.version_info >= (3, 12):
        assert "def main[n: T.int32](" in source
    else:
        assert 'n = I.dynamic("n", dtype="int32")' in source
    portable = func.script(extra_config={"script.use_pep695": False})
    assert 'n = I.dynamic("n", dtype="int32")' in portable
    tvm.ir.assert_structural_equal(
        func, tvm.script.from_source(source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    )
    tvm.ir.assert_structural_equal(
        func,
        tvm.script.from_source(portable, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}),
    )


def test_dynamic_module_body_identity():
    mod = tvm.script.from_source(
        """
n = I.dynamic("n", "int32")
m = I.dynamic("n", "int32")
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def first():
        T.evaluate(n)
    @T.prim_func(private=True)
    def second():
        T.evaluate(n + m)
""",
        check_well_formed=False,
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )
    source = mod.script()
    assert source.count('I.dynamic("n", dtype="int32")') == 2
    restored = tvm.script.from_source(
        source, check_well_formed=False, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
    )
    shared = restored["first"].body.value
    summed = restored["second"].body.value
    assert shared.same_as(summed.a)
    assert not shared.same_as(summed.b)
    tvm.ir.assert_structural_equal(mod, restored, map_free_vars=True)


def test_captured_shape_requires_concrete_symbols():
    # Native shape construction preserves concrete symbols and rejects strings.
    def build(shape):
        @T.prim_func
        def main(x: T.Buffer(shape, "float32")):
            T.evaluate(0)

        return main

    n = T.dynamic("n")
    function = build((n, 16))
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(AssertionError, match="data must be int or Expr, but got n"):
        build(("n", 16))


def test_dynamic_symbols_are_fresh_and_scope_independent():
    assert T.dynamic is I.dynamic
    n = T.dynamic("n")
    same_name = I.dynamic("n")
    k = I.dynamic("k", "int32")
    assert n.ty.dtype == "int64"
    assert k.ty.dtype == "int32"
    assert not n.same_as(same_name)

    @I.ir_module
    class Module:
        @T.prim_func
        def first(x: T.Buffer((n,), "float32")):
            T.evaluate(n)

    assert Module["first"].params[0].ty.shape[0].same_as(n)
    assert Module["first"].body.value.same_as(n)
