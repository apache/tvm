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
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T


class Base:
    def test_after_remove_symbolic_expression(self):
        """Run RemoveSymbolicExpressionInSubroutine and compare"""
        after = relax.transform.RemoveSymbolicExpressionInSubroutine()(self.before)
        tvm.ir.assert_structural_equal(self.expected, after)

    def test_after_remove_unused(self):
        """Run RemoveSymbolicExpressionInSubroutine, then remove unused

        The `RemoveSymbolicExpressionInSubroutine` transform is
        designed to allow an expression to be inferred, where
        previously the variables used within the expression were
        explicitly provided.  After
        `RemoveSymbolicExpressionInSubroutine`, the arguments
        providing the explicit definition are now unused, and can be
        removed using `RemoveUnusedParameters`.

        """
        after = tvm.ir.transform.Sequential(
            [
                relax.transform.RemoveSymbolicExpressionInSubroutine(),
                relax.transform.RemoveUnusedParameters(),
            ]
        )(self.before)
        tvm.ir.assert_structural_equal(self.expected_after_removing_unused, after)


class TestSimple(Base):
    """Replace PrimExpr with a single tir.Var

    Here, the `batch_size` and `seq_len` variables are only used as
    part of the expression `batch_size * seq_len`.  While the
    expression `batch_size * seq_len` must be propagated to the output
    shape, neither `batch_size` nor `seq_len` is otherwise required.
    """

    @property
    def before(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_flat_norm = Module.rms_norm_impl(A_flat, R.shape([batch_size, seq_len]))
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])
                return A_norm

            @R.function(private=True)
            def rms_norm_impl(
                A: R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"),
                _: R.Shape(["batch_size", "seq_len"]),
            ) -> R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"):
                A_squared = R.multiply(A, A)
                A_mean_squared = R.mean(A_squared, axis=1, keepdims=True)
                A_rms = R.sqrt(A_mean_squared)
                A_norm = A / A_rms
                return A_norm

        return Module

    @property
    def expected(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_flat_norm = Module.rms_norm_impl(A_flat, R.shape([batch_size, seq_len]))
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])
                return A_norm

            @R.function(private=True)
            def rms_norm_impl(
                A: R.Tensor(["A_dim0", "hidden_size"], "float16"),
                _: R.Shape(["batch_size", "seq_len"]),
            ) -> R.Tensor(["A_dim0", "hidden_size"], "float16"):
                A_squared = R.multiply(A, A)
                A_mean_squared = R.mean(A_squared, axis=1, keepdims=True)
                A_rms = R.sqrt(A_mean_squared)
                A_norm = A / A_rms
                return A_norm

        return Module

    @property
    def expected_after_removing_unused(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_flat_norm = Module.rms_norm_impl(A_flat)
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])
                return A_norm

            @R.function(private=True)
            def rms_norm_impl(
                A: R.Tensor(["A_dim0", "hidden_size"], "float16"),
            ) -> R.Tensor(["A_dim0", "hidden_size"], "float16"):
                A_squared = R.multiply(A, A)
                A_mean_squared = R.mean(A_squared, axis=1, keepdims=True)
                A_rms = R.sqrt(A_mean_squared)
                A_norm = A / A_rms
                return A_norm

        return Module


class TestNoMutationOfExternallyExposedSubroutine(Base):
    """No changes to public-facing functions

    Identical to `TestSimple`, except that the subroutine may be
    called directly by a user.  Therefore, its signature may not be
    altered.
    """

    @property
    def before(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_flat_norm = Module.rms_norm_impl(A_flat, R.shape([batch_size, seq_len]))
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])
                return A_norm

            @R.function
            def rms_norm_impl(
                A: R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"),
                _: R.Shape(["batch_size", "seq_len"]),
            ) -> R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"):
                A_squared = R.multiply(A, A)
                A_mean_squared = R.mean(A_squared, axis=1, keepdims=True)
                A_rms = R.sqrt(A_mean_squared)
                A_norm = A / A_rms
                return A_norm

        return Module

    expected = before
    expected_after_removing_unused = before


class TestRemoveMultipleVariables(Base):
    """Replace multiple expressions with tir.Var"""

    @property
    def before(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["n1", "n2", "n3", "n4", "n5"], "float16")
            ) -> R.Tensor(["n1", "n2", "n4", "n5"], "float16"):
                n1 = T.int64()
                n2 = T.int64()
                n3 = T.int64()
                n4 = T.int64()
                n5 = T.int64()

                A = R.reshape(A, [n1 * n2, n3, n4 * n5])
                A = Module.first_element(A, R.shape([n1, n2, n4, n5]))
                A = R.reshape(A, [n1, n2, n4, n5])
                return A

            @R.function(private=True)
            def first_element(
                A: R.Tensor(["n1 * n2", "n3", "n4 * n5"], "float16"),
                _: R.Shape(["n1", "n2", "n4", "n5"]),
            ) -> R.Tensor(["n1 * n2", "n4 * n5"], "float16"):
                A = R.strided_slice(A, axes=[1], begin=[0], end=[1])
                A = R.squeeze(A, axis=1)
                return A

        return Module

    @property
    def expected(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["n1", "n2", "n3", "n4", "n5"], "float16")
            ) -> R.Tensor(["n1", "n2", "n4", "n5"], "float16"):
                n1 = T.int64()
                n2 = T.int64()
                n3 = T.int64()
                n4 = T.int64()
                n5 = T.int64()

                A = R.reshape(A, [n1 * n2, n3, n4 * n5])
                A = Module.first_element(A, R.shape([n1, n2, n4, n5]))
                A = R.reshape(A, [n1, n2, n4, n5])
                return A

            @R.function(private=True)
            def first_element(
                A: R.Tensor(["n12", "n3", "n45"], "float16"),
                _: R.Shape(["n1", "n2", "n4", "n5"]),
            ) -> R.Tensor(["n12", "n45"], "float16"):
                A = R.strided_slice(A, axes=[1], begin=[0], end=[1])
                A = R.squeeze(A, axis=1)
                return A

        return Module

    @property
    def expected_after_removing_unused(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["n1", "n2", "n3", "n4", "n5"], "float16")
            ) -> R.Tensor(["n1", "n2", "n4", "n5"], "float16"):
                n1 = T.int64()
                n2 = T.int64()
                n3 = T.int64()
                n4 = T.int64()
                n5 = T.int64()

                A = R.reshape(A, [n1 * n2, n3, n4 * n5])
                A = Module.first_element(A)
                A = R.reshape(A, [n1, n2, n4, n5])
                return A

            @R.function(private=True)
            def first_element(
                A: R.Tensor(["n12", "n3", "n45"], "float16"),
            ) -> R.Tensor(["n12", "n45"], "float16"):
                A = R.strided_slice(A, axes=[1], begin=[0], end=[1])
                A = R.squeeze(A, axis=1)
                return A

        return Module


class TestNoReplacementIfVariableUsedInExpression(Base):
    """Do not replace PrimExpr if tir.Var is required

    Here, the `batch_size` and `seq_len` variables are used in the
    subroutine, as part of the `R.reshape` expression.  The
    `R.prim_value` arguments must be retained in order to define
    `batch_size` and `seq_len` in the subroutine.

    """

    @property
    def before(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_norm = Module.rms_norm_impl(
                    A_flat, R.prim_value(batch_size), R.prim_value(seq_len)
                )

                return A_norm

            @R.function(private=True)
            def rms_norm_impl(
                A: R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"),
                _1: R.Prim(value="batch_size"),
                _2: R.Prim(value="seq_len"),
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()

                A_squared = R.multiply(A, A)
                A_mean_squared = R.mean(A_squared, axis=1, keepdims=True)
                A_rms = R.sqrt(A_mean_squared)
                A_flat_norm = A / A_rms
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])
                return A_norm

        return Module

    expected = before
    expected_after_removing_unused = before


class TestNoReplacementIfVariableUsedInMatchCast(Base):
    """Do not replace PrimExpr if tir.Var is required

    Here, the `batch_size` and `seq_len` variables are used in the
    subroutine, as part of the `R.match_cast` binding.  The
    `R.prim_value` arguments must be retained in order to define
    `batch_size` and `seq_len` in the subroutine.

    """

    @property
    def before(self):
        @I.ir_module
        class Module:
            @R.function
            def main(
                A: R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16")
            ) -> R.Tensor(["batch_size", "seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()
                A_flat = R.reshape(A, [-1, hidden_size])
                A_flat_norm = Module.rms_norm_impl(
                    A_flat, R.prim_value(batch_size), R.prim_value(seq_len)
                )
                A_norm = R.reshape(A_flat_norm, [batch_size, seq_len, hidden_size])

                return A_norm

            @R.function(private=True)
            def rms_norm_impl(
                A: R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"),
                _1: R.Prim(value="batch_size"),
                _2: R.Prim(value="seq_len"),
            ) -> R.Tensor(["batch_size * seq_len", "hidden_size"], "float16"):
                batch_size = T.int64()
                seq_len = T.int64()
                hidden_size = T.int64()

                A_norm_ndim = R.call_pure_packed(
                    "some_packed_func_implementation", A, sinfo_args=[R.Tensor(ndim=3)]
                )
                A_norm = R.match_cast(A_norm_ndim, R.Tensor([batch_size, seq_len, hidden_size]))
                A_flat_norm = R.reshape(A_norm, [batch_size * seq_len, hidden_size])

                return A_flat_norm

        return Module

    expected = before
    expected_after_removing_unused = before


if __name__ == "__main__":
    tvm.testing.main()
