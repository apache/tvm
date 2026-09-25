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


import tvm
import tvm.testing
from tvm.script import relax as R
from tvm.script import tirx as T


def relax_extern_func():
    @R.function
    def func(A: R.Tensor([10, 20], "float32")):
        func = R.ExternFunc("dummy_func")

        B: R.Tensor([10, 20], "float32") = R.call_dps_packed(
            func, [A], out_ty=R.Tensor([10, 20], "float32")
        )

        C: R.Tensor(ndim=2, dtype="float32") = R.call_dps_packed(
            func, [B], out_ty=R.Tensor([10, 20], "float32")
        )

        return C

    return func


def relax_match_cast_ty_proxy():
    """Default type constructors may be used as expressions

    This is a regression test.  The TVMScript parser allows Type
    to be specified using a default-constructible class
    (e.g. `R.Tensor` or `R.Shape`) rather than an instance of that
    class (e.g. `R.Tensor()` or `R.Shape()`).  In previous
    implementations, this was only handled when the `Type` was
    used in an annotation context.  However, a `Type` may also
    appear as an argument, which is passed to `R.match_cast`.  Use of
    a default-constructible class must be handled in this context as
    well.
    """

    def make_ir_generator(proxy_subclass):
        def inner():
            @R.function
            def func(A: R.Any):
                B = R.match_cast(A, proxy_subclass)
                return B

            return func

        inner.__name__ = subclass.__name__
        return inner

    # Prim and DTensor require arguments; the remaining public type
    # constructors also work as bare values in match_cast expressions.
    subclasses = [R.Any, R.Tensor, R.Callable, R.Tuple, R.Shape]

    for subclass in subclasses:
        yield make_ir_generator(subclass)


def relax_symbolic_var():
    """Relax tensors may use symbolic variables."""
    N = T.dynamic("N", "int64")

    @R.function
    def func(A: R.Tensor([N], "float16")):
        B: R.Tensor([N], "float16") = A
        return B

    return func


def relax_float_symbolic_var():
    """Relax scalar variables may use any dtype."""

    @R.function
    def func(value: T.float16):
        return value

    return func


ir_generator = tvm.testing.parameter(
    *relax_match_cast_ty_proxy(), relax_symbolic_var, relax_float_symbolic_var
)


relax_ir_generator = tvm.testing.parameter(
    relax_extern_func,
)


show_all_relax_ty = tvm.testing.parameter(
    by_dict={
        "show_all_ty": True,
        "hide_inferable_ty": False,
    }
)


_NOT_ROUNDTRIP_STABLE: set[str] = set()


def test_roundtrip(ir_generator):
    if getattr(ir_generator, "__name__", "") in _NOT_ROUNDTRIP_STABLE:
        import pytest

        pytest.skip(f"{ir_generator.__name__}: not round-trip stable here")
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, True)


def test_relax_roundtrip(relax_ir_generator, show_all_relax_ty):
    original = relax_ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(
            show_meta=True,
            show_all_ty=show_all_relax_ty,
        ),
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, True)


if __name__ == "__main__":
    tvm.testing.main()
