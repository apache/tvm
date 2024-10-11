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
from tvm.relax.transform.transform import CanonicalizeBindings
import tvm.script
import tvm.testing
import pytest
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script import ir as I, relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(CanonicalizeBindings()(input), expected)


def test_simple_assignments():
    @I.ir_module
    class TestChainAssignments:
        @R.function
        def main(x: R.Tensor):
            y = x
            z = y
            q = z
            p = q
            o = p
            return o

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            return x

    verify(TestChainAssignments, Expected)


def test_dataflow_block():
    @I.ir_module
    class TestDataflowAssignments:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.const(1)
                z = y
                o = z
                p = o
                m = p
                n = m
                R.output(n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return n

    verify(TestDataflowAssignments, Expected)


def test_assign_to_output_in_dataflow_block():
    @I.ir_module
    class TestDataflowAssignments:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = x  # is not a dataflow var
                z = y
                o = z
                p = o
                m = p
                n = m
                R.output(n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            # we get a dataflow block where the
            # only assignment is n = x, which we can eliminate,
            # resulting in an empty block that is normalized away
            return x

    verify(TestDataflowAssignments, Expected)


def test_ops():
    @I.ir_module
    class TestOps:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            w = y
            q = x
            z = R.add(w, q)
            return R.add(q, z)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            z = R.add(y, x)
            return R.add(x, z)

    verify(TestOps, Expected)


@pytest.mark.xfail(reason="The lhs and rhs of an assignment should have the same struct info.")
def test_casting():
    @I.ir_module
    class TestCasting:
        @R.function
        def main(x: R.Tensor) -> R.Object:
            y = x
            # z will be treated as object type even though it's a tensor
            z: R.Object = y
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor) -> R.Object:
            # Cannot unify because the cast indicates user intent
            z: R.Object = x
            return z

    verify(TestCasting, Expected)


def test_match_cast():
    @I.ir_module
    class TestMatchCast:
        @R.function
        def main(x: R.Tensor):
            q = x
            m, n = T.int64(), T.int64()
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            # can't get rid of z because its struct_info is different from x's
            m, n = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((m, n)))
            return z

    verify(TestMatchCast, Expected)


def test_same_shape():
    @I.ir_module
    class TestSameShape:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.int64(), T.int64()
            y = x
            # trivial check
            z = R.match_cast(x, R.Tensor((m, n), "float32"))
            w = z
            q = R.add(w, y)
            return R.add(q, w)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            # the trivial check is canonicalized into a var binding
            # and then eliminated
            q = R.add(x, x)
            return R.add(q, x)

    verify(TestSameShape, Expected)


def test_change_shape():
    @I.ir_module
    class TestChangeShape:
        @R.function
        def main(x: R.Tensor(ndim=2)):
            y = x
            # The MatchCast is non-trivial, as it introduces new shape
            # vars.  Because the input tensor has an unknown shape
            # rather than a symbolic shape, these new shape vars
            # cannot be expressed in terms of previous variables.
            # Therefore, the match cast must be retained.
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            w = z
            q = R.add(w, y)
            return R.add(q, w)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(ndim=2)):
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            # the struct_info field on q will need to be updated
            q = R.add(z, x)
            return R.add(q, z)

    verify(TestChangeShape, Expected)


def test_replace_symbolic_variable_and_remove_match_cast():
    @I.ir_module
    class TestChangeShape:
        @R.function
        def main(x: R.Tensor(("m", "n"))):
            y = x
            # The MatchCast is non-trivial, as it introduces new shape
            # vars.  However, the new shape vars are redundant, and
            # are replaced by canonicalization.  After replacing the
            # new shape vars, the MatchCast is trivial and may be
            # removed.
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            w = z
            q = R.add(w, y)
            return R.add(q, w)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"))):
            m = T.int64()
            n = T.int64()
            q: R.Tensor([m, n]) = R.add(x, x)
            return R.add(q, x)

    verify(TestChangeShape, Expected)


def test_replace_symbolic_variable_and_remove_match_cast_of_tuple():
    """Symbolic variables may be defined in R.match_cast of tuple

    This test is similar to
    `test_replace_symbolic_variable_and_remove_match_cast`, except
    that the MatchCast is performed on a Relax tuple.

    This is a regression test.  Earlier implementations only inferred
    TIR variables from `R.match_cast` of tensors, shapes, and prim
    values, but omitted tuples.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tuple(R.Tensor(("m", "n")))):
            y = x
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tuple(R.Tensor((o, p))))
            w = z
            q = R.add(w[0], y[0])
            return R.add(q, w[0])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tuple(R.Tensor(("m", "n")))):
            q = R.add(x[0], x[0])
            return R.add(q, x[0])

    verify(Before, Expected)


def test_unwrap_tuple():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            tuple_var = (x, y)
            w = tuple_var[0]
            q = tuple_var[1]
            z = R.add(w, q)
            return R.add(q, z)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            tuple_var = (x, y)
            z = R.add(x, y)
            return R.add(y, z)

    verify(Before, Expected)


def test_basic_folding_example():
    @I.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                n = y
                R.output(n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return n

    verify(Input, Expected)


def test_fold_match_cast():
    @I.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                n = R.match_cast(y, R.Tensor((), "int32"))
                R.output(n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                # the cast is trivial, so it is removed
                n = R.const(1)
                R.output(n)
            return n

    verify(Input, Expected)


def test_fold_variables_from_match_cast():
    """Symbolic variables in R.match_cast may be inferred

    If the argument to `R.match_cast` has known shape parameters, they
    may be used to infer symbolic shape parameters.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            state: R.Tensor([16], dtype="float32"),
            A: R.Tensor([16, 16], dtype="float32"),
            B: R.Tensor([16, 16], dtype="float32"),
        ):
            N1 = T.int64()
            M = T.int64()
            N2 = T.int64()

            # The symbolic variables `N1`, `N2` and `M` are defined by
            # these `R.match_cast` statements.  Since the inputs have
            # a known shape, the values of these symbolic variables
            # may be inferred.
            lhs_A = R.match_cast(A, R.Tensor([N1, M], dtype="float32"))
            lhs_B = R.match_cast(B, R.Tensor([N2, M], dtype="float32"))
            rhs = R.match_cast(state, R.Tensor([M], dtype="float32"))

            # The symbolic shapes propagate downstream.
            lhs: R.Tensor([N1 + N2, M], "float32") = R.concat((lhs_A, lhs_B), axis=0)
            proj_concat: R.Tensor([N1 + N2], "float32") = R.matmul(lhs, rhs, out_dtype="void")
            proj_A = R.strided_slice(
                proj_concat,
                (R.prim_value(0),),
                (R.prim_value(0),),
                (R.prim_value(N1),),
                assume_inbound=False,
            )
            proj_B = R.strided_slice(
                proj_concat,
                [R.prim_value(0)],
                [R.prim_value(N1)],
                [R.prim_value(N1 + N2)],
                assume_inbound=False,
            )
            return (proj_A, proj_B)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            state: R.Tensor([16], dtype="float32"),
            A: R.Tensor([16, 16], dtype="float32"),
            B: R.Tensor([16, 16], dtype="float32"),
        ):
            # The function no longer depends on symbolic variables.
            # Shape inference is now propagated using the
            # statically-known shapes.

            lhs: R.Tensor([32, 16], dtype="float32") = R.concat((A, B), axis=0)
            proj_concat: R.Tensor([32], dtype="float32") = R.matmul(lhs, state, out_dtype="void")
            proj_A: R.Tensor([16], dtype="float32") = R.strided_slice(
                proj_concat,
                [R.prim_value(0)],
                [R.prim_value(0)],
                [R.prim_value(16)],
                assume_inbound=False,
            )
            proj_B: R.Tensor([16], dtype="float32") = R.strided_slice(
                proj_concat,
                [R.prim_value(0)],
                [R.prim_value(16)],
                [R.prim_value(32)],
                assume_inbound=False,
            )
            return (proj_A, proj_B)

    verify(Before, Expected)


def test_inconsistent_match_cast_raises_error():
    """Symbolic variables from R.match_cast must be consistent

    All match cast statements must provide consistent definitions for
    symbolic variables.  In this test, the value of `M` would be
    inferred as 16 from either `state` or `A`, but would be inferred
    as 32 from `B`.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            state: R.Tensor([16], dtype="float32"),
            A: R.Tensor([16, 16], dtype="float32"),
            B: R.Tensor([32, 32], dtype="float32"),
        ):
            N1 = T.int64()
            M = T.int64()
            N2 = T.int64()

            # These R.match_cast statements define inconsistent values
            # for the symbolic shape parameters.
            lhs_A = R.match_cast(A, R.Tensor([N1, M], dtype="float32"))
            lhs_B = R.match_cast(B, R.Tensor([N2, M], dtype="float32"))
            rhs = R.match_cast(state, R.Tensor([M], dtype="float32"))

            lhs: R.Tensor([N1 + N2, M], "float32") = R.concat((lhs_A, lhs_B), axis=0)
            proj_concat: R.Tensor([N1 + N2], "float32") = R.matmul(lhs, rhs, out_dtype="void")
            proj_A = R.strided_slice(
                proj_concat,
                (R.prim_value(0),),
                (R.prim_value(0),),
                (R.prim_value(N1),),
                assume_inbound=False,
            )
            proj_B = R.strided_slice(
                proj_concat,
                [R.prim_value(0)],
                [R.prim_value(N1)],
                [R.prim_value(N1 + N2)],
                assume_inbound=False,
            )
            return (proj_A, proj_B)

    with pytest.raises(ValueError, match="MatchCast statements must be consistent"):
        CanonicalizeBindings()(Before)


def test_match_cast_may_have_distinct_values_in_branches():
    """Conditional branches may have different values of symbolic variables

    Here, the value of `N` can be inferred as 16 within the `if`
    branch and as 32 within the `else` branch.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            state: R.Tensor(["N"], dtype="float32"),
            A: R.Tensor(["M", 16], dtype="float32"),
            B: R.Tensor(["M", 32], dtype="float32"),
            scale: R.Prim("float32"),
        ):
            N = T.int64()
            M = T.int64()

            if N == 16:
                weights: R.Tensor([M, 16], "float32") = A * scale
                weights: R.Tensor([M, N], "float32") = R.match_cast(
                    weights, R.Tensor([M, N], "float32")
                )
                weights: R.Tensor([M, N], "float32") = weights * scale
            else:
                weights: R.Tensor([M, 32], "float32") = B * scale
                weights: R.Tensor([M, N], "float32") = R.match_cast(
                    weights, R.Tensor([M, N], "float32")
                )
                weights: R.Tensor([M, N], "float32") = weights * scale

            weights: R.Tensor([M, N], "float32") = weights * scale

            out: R.Tensor([M], "float32") = R.matmul(weights, state)

            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            state: R.Tensor(["N"], dtype="float32"),
            A: R.Tensor(["M", 16], dtype="float32"),
            B: R.Tensor(["M", 32], dtype="float32"),
            scale: R.Prim("float32"),
        ):
            N = T.int64()
            M = T.int64()

            if N == 16:
                # Prior to the R.match_cast, the
                weights: R.Tensor([M, 16], "float32") = A * scale
                # The scaled weights within the branch may perform
                # shape inference knowing that N==16.
                weights: R.Tensor([M, 16], "float32") = weights * scale
                # The match cast on exiting the if branch restores the
                weights = R.match_cast(weights, R.Tensor([M, N], "float32"))
            else:
                # Prior to the R.match_cast, the
                weights: R.Tensor([M, 32], "float32") = B * scale
                # Within the else-branch, the R.match_cast implies
                # that N==32.  While this conflicts with the earlier
                # definition, the two occur in separate branches, so
                # this is legal.
                # The scaled weights within the branch may perform
                # shape inference knowing that N==32.
                weights: R.Tensor([M, 32], "float32") = weights * scale
                weights = R.match_cast(weights, R.Tensor([M, N], "float32"))

            # Outside of the conditional, we no longer have a known
            # value for N, so this shape inference must be done using
            # a dynamic shape for `N`.
            weights: R.Tensor([M, N], "float32") = weights * scale

            # After the conditional branch, we no longer have a known
            # value of N, so this shape inference must use the dynamic
            # shape.
            out: R.Tensor([M], "float32") = R.matmul(weights, state)

            return out

    verify(Before, Expected)


def test_multiple_outputs():
    @I.ir_module
    class Input:
        @R.function
        def main():
            with R.dataflow():
                x = R.const(1)
                y = R.const(1)
                z = R.const(1)
                l = x
                m = y
                n = z
                R.output(l, m, n)
            return (l, m, n)

    @I.ir_module
    class Expected:
        @R.function
        def main():
            with R.dataflow():
                l = R.const(1)
                m = R.const(1)
                n = R.const(1)
                R.output(l, m, n)
            return (l, m, n)

    verify(Input, Expected)


def test_single_output_multiple_nondataflow():
    """Non-dataflow vars being updated may also be part trivial bindings

    Like `test_multiple_outputs`, but only `n` is used in the return
    statement.
    """

    @I.ir_module
    class Input:
        @R.function
        def main():
            with R.dataflow():
                x = R.const(1)
                y = R.const(1)
                z = R.const(1)
                l = x
                m = y
                n = z
                R.output(l, m, n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main():
            with R.dataflow():
                l = R.const(1)
                m = R.const(1)
                n = R.const(1)
                R.output(n)
            return n

    verify(Input, Expected)


def test_fold_const_to_output():
    @I.ir_module
    class Before:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return n

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return R.const(1)

    verify(Before, Expected)


def test_canonicalize_var_to_dataflow_var_if_legal():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    DataflowVar instances may only be used inside a DataflowBlock.  If
    a trivial binding `y = x` occurs, where `x` is a `DataflowVar` and
    `y` is a `Var`, replacing `y` with `x` may result in usage of a
    `DataflowVar` outside of a `DataflowBlock`.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = R.add(y, R.const(1))
                R.output(y, z)
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = R.add(y, R.const(1))
                R.output(z)
            return z

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_update_dataflow_computations_if_var_replacement_occurs():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    DataflowBlocks may produce additional outputs after the first
    output Var, and these additional outputs may be in terms of the
    first output.  Computations that depend on a replaced var must be
    updated to remain well-formed.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                lv1 = R.add(x, R.const(1))
                gv1 = lv1
                gv2 = R.add(lv1, R.const(1))
                R.output(gv1, gv2)
            return (gv1, gv2)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                # lv1 has been replaced with gv1
                gv1 = R.add(x, R.const(1))
                # So gv1 must be used in the computation of gv2
                gv2 = R.add(gv1, R.const(1))
                R.output(gv1, gv2)
            return (gv1, gv2)

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_update_dataflow_computations_if_var_replacement_occurs_after_usage():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    Like test_update_dataflow_computations_if_var_replacement_occurs,
    but the usage of a DataflowVar occurs before the trivial binding
    that causes it to be replaced.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                lv1 = R.add(x, R.const(1))
                gv2 = R.add(lv1, R.const(1))
                gv1 = lv1
                R.output(gv1, gv2)
            return (gv1, gv2)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                # lv1 has been replaced with gv1
                gv1 = R.add(x, R.const(1))
                # So gv1 must be used in the computation of gv2
                gv2 = R.add(gv1, R.const(1))
                # Even though the trivial binding of "gv1 = lv1"
                # occurred in this position.
                R.output(gv1, gv2)
            return (gv1, gv2)

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_replace_var_with_dataflow_if_all_usage_within_dataflow_block():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    Like `test_update_dataflow_computations_if_var_replacement_occurs`,
    except that `gv1` is not part of the function's return value.  When
    deciding which variable to replace, the following logic is applied:

    1. Normally, when encountering `x = y`, replace usage of `x` with `y`.

    2. Unless the trivial binding is a `var_x = dataflow_y`, in which case
       replace `dataflow_y` with `var_x` at the point of definition.  This
       prevents usage of `dataflow_y` from escaping the dataflow block.

    3. Unless `var_x` has no usage outside the dataflow block, in which
       case we replace usage of `var_x` with `dataflow_y`.

    The third rule ensures that canonicalization can occur in a single
    step.  Otherwise, the output of this test case would contain a
    non-dataflow var defined within a dataflow block, and only used within
    that dataflow block.  (Equivalent to the input for the test case
    `test_canonicalize_var_to_dataflow_var_if_legal`.)
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                lv1 = R.add(x, R.const(1))
                gv1 = lv1
                gv2 = R.add(lv1, R.const(1))
                R.output(gv1, gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                gv1 = R.add(x, R.const(1))
                gv2 = R.add(gv1, R.const(1))
                R.output(gv2)
            return gv2

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_var_to_dataflow_with_trivial_binding():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    Like
    `test_replace_var_with_dataflow_if_all_usage_within_dataflow_block`,
    except the non-DataflowVar is on the right-hand side of the trivial
    binding.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                gv1 = R.add(x, R.const(1))
                lv1 = gv1
                gv2 = R.add(lv1, R.const(1))
                R.output(gv1, gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                gv1 = R.add(x, R.const(1))
                gv2 = R.add(gv1, R.const(1))
                R.output(gv2)
            return gv2

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_with_updated_struct_info():
    """CanonicalizeBindings and Normalizer may both replace a Var

    If the CanonicalizeBindings pass has no replacements to make for a
    variable, it must still delegate to the ExprMutator.  This is because
    a variable replacement may have occurred as part of the IRNormalizer,
    in order to provide better struct info.
    """

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor(("n", 16), dtype="int32")) -> R.Tensor(("n", 16), dtype="int32"):
            # CanonicalizeBindings recognizes this trivial binding, and
            # replaces `B` with `A`.
            B = A
            # The value is updated from `R.add(B,B)` to `R.add(A,A)`.
            # Changing the value triggers struct inference, allowing the
            # shape to be updated to `[n,16]`.  This requires a variable
            # replacement, which is tracked by the `ExprMutator`.
            C: R.Tensor(dtype="int32", ndim=2) = R.add(B, B)
            # Replacement of `C` is not explicitly tracked by
            # CanonicalizeBindings.  However, if CanonicalizeBindings just
            # returns `GetRef<Var>(var)`, `ExprMutator` cannot apply the
            # replacement, and this will try to return the old
            # version of `C` with `ndim=2`.
            return C

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def main(A: R.Tensor(("n", 16), dtype="int32")) -> R.Tensor(("n", 16), dtype="int32"):
            n = T.int64()
            C: R.Tensor([n, 16], "int32") = R.add(A, A)
            return C

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_trivial_binding_to_dataflow_var():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    DataflowVar instances may only be used inside a DataflowBlock.  If
    a trivial binding `y = x` occurs, where `x` is a `DataflowVar` and
    `y` is a `Var`, replacing `y` with `x` may result in usage of a
    `DataflowVar` outside of a `DataflowBlock`.

    If a binding exists solely to convert from DataflowVar into Var,
    then canonicalization replaces the earlier DataflowVar with a Var.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = y
                R.output(z)
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                R.output(y)
            return y

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_multiple_trivial_binding_to_dataflow_var():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    Like test_canonicalize_trivial_binding_to_dataflow_var, but there
    exist multiple trivial bindings to the DataflowVar.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(w: R.Tensor):
            with R.dataflow():
                x = R.add(w, R.const(1))
                y = x
                z = x
                R.output(y, z)
            return (y, z)

    @I.ir_module
    class Expected:
        @R.function
        def main(w: R.Tensor):
            with R.dataflow():
                x = R.add(w, R.const(1))
                R.output(x)
            return (x, x)

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_trivial_var_binding_inside_dataflow_block():
    """Canonicalize Var to DataflowVar inside DataflowBlock

    Canonicalization handles cases where a Var could be replaced by a
    DataflowVar, and where a Var is a trivial binding.  If these two
    cases both occur, should produce reasonable results.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = y
                R.output(y, z)
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                R.output(y)
            return y

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_across_non_dataflow_tuple():
    """Canonicalize Var to DataflowVar inside DataflowBlock"""

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = (y,)
                gv = R.add(z[0], R.const(1))
                R.output(z, gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.add(x, R.const(1))
                z = (y,)
                gv = R.add(y, R.const(1))
                R.output(gv)
            return gv

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_var_used_in_distinct_df_blocks():
    """If a var is used only in dataflow blocks,
    but outside of the one where it was originally defined,
    it should be exposed as an output."""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                # v must remain exposed!
                R.output(v)
            _ = R.print(format="Hi mom!")
            with R.dataflow():
                a = R.multiply(v, v)
                b = R.add(a, a)
                c = R.subtract(b, a)
                d = R.add(c, c)
                R.output(d)
            return d

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Before, after)


def test_inner_function():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():

                @R.function(pure=False)
                def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                    with R.dataflow():
                        z = R.add(x, y)
                        w = R.multiply(x, z)
                        v = R.add(y, w)
                        R.output(z, w, v)
                    _ = R.print(format="oops")
                    with R.dataflow():
                        a = R.multiply(v, v)
                        b = R.add(a, a)
                        c = R.multiply(a, b)
                        R.output(a, b, c)
                    return c

                z = R.add(x, y)
                w = R.multiply(z, z)
                v = R.divide(w, z)
                R.output(inner_func, z, v, w)
            q = inner_func(w, v)
            with R.dataflow():
                a = R.multiply(q, q)
                b = R.add(a, a)
                c = R.multiply(b, a)
                R.output(a, b, c)
            return c

    # expected: we do not need to expose all the outputs
    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():

                @R.function(pure=False)
                def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                    with R.dataflow():
                        z = R.add(x, y)
                        w = R.multiply(x, z)
                        v = R.add(y, w)
                        R.output(v)
                    _ = R.print(format="oops")
                    with R.dataflow():
                        a = R.multiply(v, v)
                        b = R.add(a, a)
                        c = R.multiply(a, b)
                        R.output(c)
                    return c

                z = R.add(x, y)
                w = R.multiply(z, z)
                v = R.divide(w, z)
                R.output(inner_func, v, w)
            q = inner_func(w, v)
            with R.dataflow():
                a = R.multiply(q, q)
                b = R.add(a, a)
                c = R.multiply(b, a)
                R.output(c)
            return c

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalize_inside_branches():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                R.output(z)
            if R.const(True):
                with R.dataflow():
                    w = R.add(z, z)
                    v = R.multiply(w, w)
                    # w does not need to be output
                    R.output(w, v)
                q = v
            else:
                with R.dataflow():
                    w = R.multiply(z, z)
                    v = R.add(w, w)
                    R.output(w, v)
                q = v
            return q

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                R.output(z)
            if R.const(True):
                with R.dataflow():
                    w = R.add(z, z)
                    v = R.multiply(w, w)
                    R.output(v)
                q = v
            else:
                with R.dataflow():
                    w = R.multiply(z, z)
                    v = R.add(w, w)
                    R.output(v)
                q = v
            return q

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_canonicalization_causes_struct_info_update():
    """Regression test for failure mode causing undefined variable

    The ExprMutator is only allowed to update a variable's struct info
    if the value bound to it has new struct info.  When
    CanonicalizeBindings replaces a trivial binding, this may provide
    better struct info as a result.  If this happens, the

    In previous implementations, ExprMutator::ReEmitBinding defined a
    remap for `binding->var->vid`, even if the derived class defined a
    replacement by overriding `VisitVarDef`.  If the derived class
    defines a new variable binding by overriding `VisitVarDef`, and
    also causes a variable replacement by overriding `VisitExpr` and
    returning a type with different struct info, then `ExprMutator`
    must check for both `binding->var->vid` *AND* `new_var->vid`.  The
    former may be present in the unmodified graph, and the latter may
    be produced by the derived class before delegating to the base
    class.
    """

    @I.ir_module
    class Before:
        @R.function
        def transform_params(
            A: R.Tensor(("vocab_size", 4096), dtype="float16"),
            B: R.Tensor((6144, 4096), dtype="float16"),
        ):
            with R.dataflow():
                # Trivial binding of `DataFlow = NonDataFlow`.
                # Wherever `C` is used, Canonicalization will attempt
                # to replace it with `B`.
                C = B

                # RHS contains `(A,C)`, which CanonicalizeBindings
                # replaces with `(A,B)`.  Because this changes the
                # RHS, a new LHS (and new struct info!) will be
                # generated.
                D: R.Tuple(
                    R.Tensor(dtype="float16", ndim=2),
                    R.Tensor((6144, 4096), dtype="float16"),
                ) = (A, C)

                # Trivial binding of `NonDataFlow = DataFlow`.  The
                # definition of `D` will be replaced with a definition
                # of `E`.  This definition of `E` will then be updated
                # to have a known shape.
                E = D
                R.output(E)

            # By the time `E` is encountered at a usage site, the
            # `ExprMutator` must have a replacement for the old
            # version of `E` with `ndim=2` to the new versions of `E`
            # with `shape=[vocab_size,4096]`.
            return E

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            A: R.Tensor(("vocab_size", 4096), dtype="float16"),
            B: R.Tensor((6144, 4096), dtype="float16"),
        ):
            vocab_size = T.int64()
            with R.dataflow():
                E: R.Tuple(
                    R.Tensor((vocab_size, 4096), dtype="float16"),
                    R.Tensor((6144, 4096), dtype="float16"),
                ) = (A, B)

                R.output(E)
            return E

    after = relax.transform.CanonicalizeBindings()(Before)
    assert_structural_equal(Expected, after)


def test_unwrap_tuple_of_constant():
    @I.ir_module
    class TestChainAssignments:
        @R.function
        def main():
            tup = (R.const(0, "int64"), R.const(1, "int64"))
            x = tup[0]
            y = tup[1]
            z = R.add(x, y)
            return z

    @I.ir_module
    class Expected:
        @R.function
        def main():
            tup = (R.const(0, "int64"), R.const(1, "int64"))
            x = tup[0]
            y = tup[1]
            z = R.add(R.const(0, "int64"), R.const(1, "int64"))
            return z

    verify(TestChainAssignments, Expected)


def test_trivial_binding_of_replaced_non_dataflow_var():
    @I.ir_module
    class Before:
        @R.function
        def main(param_tuple: R.Tuple([R.Tensor])):
            with R.dataflow():
                A = param_tuple[0]
                B = A
                C = R.add(A, B)
                R.output(A, B, C)
            return C

    @I.ir_module
    class Expected:
        @R.function
        def main(param_tuple: R.Tuple([R.Tensor])):
            with R.dataflow():
                A = param_tuple[0]
                C = R.add(A, A)
                R.output(C)
            return C

    After = CanonicalizeBindings()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    def _get_binding_names(mod):
        return [binding.var.name_hint for binding in mod["main"].body.blocks[0].bindings]

    expected_names = _get_binding_names(Expected)
    after_names = _get_binding_names(After)

    assert after_names == expected_names


def test_trace_tuple_through_round_trip():
    """Canonicalize to the orignal tuple, without unwrap/rewrap."""

    @I.ir_module
    class Before:
        @R.function
        def main(param_tuple: R.Tuple([R.Tensor, R.Tensor, R.Tensor])):
            with R.dataflow():
                A = param_tuple[0]
                B = param_tuple[1]
                C = param_tuple[2]
                output = (A, B, C)
                R.output(output)
            return output

    @I.ir_module
    class Expected:
        @R.function
        def main(param_tuple: R.Tuple([R.Tensor, R.Tensor, R.Tensor])):
            with R.dataflow():
                A = param_tuple[0]
                B = param_tuple[1]
                C = param_tuple[2]
                R.output()

            return param_tuple

    After = CanonicalizeBindings()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_trace_partial_tuple_through_round_trip():
    """Canonicalize to the orignal tuple, without unwrap/rewrap."""

    @I.ir_module
    class Before:
        @R.function
        def main(param_tuple: R.Tuple([R.Tensor, R.Tensor, R.Tensor])):
            with R.dataflow():
                A = param_tuple[0]
                B = param_tuple[1]
                output = (A, B)
                R.output(output)
            return output

    Expected = Before

    After = CanonicalizeBindings()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
