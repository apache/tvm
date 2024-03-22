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

"""Test FNormalize usage"""

import tvm
import tvm.testing
import tvm.relax.testing.transform

from tvm import relax
from tvm.script.parser import ir as I, relax as R, tir as T

import pytest

define_normalization = tvm.testing.parameter(True)


@pytest.fixture
def custom_op(define_normalization):
    """A custom operator for testing purposes

    The custom operator ignores its second argument.  If there isn't a
    second argument which can be ignored, FNormalize appends an
    additional argument so that it can be properly ignored.
    """

    op_name = "custom_op.ignore_second_argument"

    def infer_struct_info(call: relax.Call, context: relax.BlockBuilder):
        return call.args[0].struct_info

    def normalize(context: relax.BlockBuilder, call: relax.Call):
        if len(call.args) == 1:
            return relax.Call(call.op, [call.args[0], relax.Tuple([])])
        else:
            return call

    def legalize(context: relax.BlockBuilder, call: relax.Call):
        return call.args[0]

    op_attrs = {
        "FInferStructInfo": infer_struct_info,
        "FLegalize": legalize,
        "FPurity": True,
    }
    if define_normalization:
        op_attrs["FNormalize"] = normalize

    for key, value in op_attrs.items():
        tvm.ir.register_op_attr(op_name, key, value)

    op = tvm.ir.Op.get(op_name)
    yield op

    for key in op_attrs:
        op.reset_attr(key)


def test_normalization_suppressed_for_tvmscript(custom_op):
    """FNormalize isn't applied when parsing TVMScript

    TVMScript should be able to produce un-normalized Relax IR for
    specifying test cases, and to ensure that no changes occur when
    performing a round-trip through TVMScript.
    """

    @R.function
    def func(A: R.Tensor):
        return relax.Call(custom_op, [A])

    call_expr = func.body.blocks[0].bindings[0].value
    assert isinstance(
        call_expr, relax.Call
    ), "Test implementation error, didn't extract the correct expression"
    assert (
        len(call_expr.args) == 1
    ), "Expected TVMScript to suppress use of FNormalize, produce arguments as written"


@pytest.mark.skip_well_formed_check_before_transform
def test_normalization_applied_during_cpp_mutator(custom_op):
    """FNormalize is applied by relax::ExprMutator subclasses"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A])

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A, R.tuple()])

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm.ir.structural_equal(Before, After)
    tvm.ir.assert_structural_equal(Expected, After)


def test_normalization_applied_during_python_mutator(custom_op):
    """FNormalize is applied by relax.ExprMutator subclasses"""

    @R.function(private=True)
    def before(A: R.Tensor):
        return relax.Call(custom_op, [A])

    @R.function(private=True)
    def expected(A: R.Tensor):
        return relax.Call(custom_op, [A, R.tuple()])

    @relax.expr_functor.mutator
    class EmptyPyExprMutator(relax.PyExprMutator):
        """Default ExprMutator"""

    after = EmptyPyExprMutator().visit_expr(before)

    assert not tvm.ir.structural_equal(before, after)
    tvm.ir.assert_structural_equal(expected, after)


def test_normalized_call_node_is_well_formed(custom_op):
    """If FNormalize wouldn't apply a change, the IR is well-formed"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A, A])

    assert relax.analysis.well_formed(Module)


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.parametrize("define_normalization", [True, False])
def test_un_normalized_call_node_is_ill_formed(custom_op, define_normalization):
    """If FNormalize would apply a change, the IR is ill-formed

    This only applies if FNormalize exists.  An operator without
    FNormalize has no corresponding check applied.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A])

    if define_normalization:
        assert not relax.analysis.well_formed(Module)
    else:
        assert relax.analysis.well_formed(Module)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, args],
                sinfo_args=[A.struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, relax.Tuple([A])],
                sinfo_args=[A.struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm.ir.structural_equal(Before, After)
    tvm.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_argument_to_inline_tuple_for_call_tir(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir

    Like `test_normalize_to_inline_tuple_for_call_tir`, but the
    argument tuple is provided as a relax function argument.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(args: R.Tuple([R.Tensor([16], "float32")])):
            cls = Before
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, args],
                sinfo_args=[args[0].struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @R.function
        def main(args: R.Tuple([R.Tensor([16], "float32")])):
            cls = Expected
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, relax.Tuple([args[0]])],
                sinfo_args=[args[0].struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm.ir.structural_equal(Before, After)
    tvm.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir_inplace(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir_inplace"""

    # The CallTIRInplaceAttrs cannot be constructed from the Python
    # API.  Therefore, declaring the Expected output first, so that
    # the attributes can be used for the non-normalized Before.
    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return R.call_tir_inplace(
                cls.multiply_by_two,
                A,
                inplace_indices=[0],
                out_sinfo=[A.struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0

    inplace_attrs = Expected["main"].body.blocks[0].bindings[1].value.attrs

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir_inplace"),
                [cls.multiply_by_two, args],
                attrs=inplace_attrs,
                sinfo_args=[A.struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm.ir.structural_equal(Before, After)
    tvm.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir_with_grad(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir_with_grad"""

    # The CallTIRWithGradAttrs cannot be constructed from the Python
    # API.  Therefore, declaring the Expected output first, so that
    # the attributes can be used for the non-normalized Before.
    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return R.call_tir_with_grad(
                cls.multiply_by_two,
                A,
                out_sinfo=[A.struct_info],
                te_grad_name="f_grad",
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

        @T.prim_func(private=True)
        def f_grad(
            A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32"), Grad: T.Buffer(16, "float32")
        ):
            for i in range(16):
                Grad[i] = 2.0

    with_grad_attrs = Expected["main"].body.blocks[0].bindings[1].value.attrs

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm.ir.Op.get("relax.call_tir_with_grad"),
                [cls.multiply_by_two, args],
                attrs=with_grad_attrs,
                sinfo_args=[A.struct_info],
            )

        @T.prim_func(private=True)
        def multiply_by_two(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

        @T.prim_func(private=True)
        def f_grad(
            A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32"), Grad: T.Buffer(16, "float32")
        ):
            for i in range(16):
                Grad[i] = 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm.ir.structural_equal(Before, After)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm.testing.main()
