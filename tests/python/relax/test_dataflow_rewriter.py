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


import tvm.testing
from tvm.script import ir as I, relax as R, tir as T

import pytest


def test_rewrite_defined_by_ir_module():
    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @R.function
    def before(x: R.Tensor([32], "float32")):
        R.func_attr({"global_symbol": "main"})
        split = R.split(x, 2)
        lhs = split[0]
        rhs = split[1]
        out = lhs + rhs
        return out

    @R.function
    def expected(x: R.Tensor([32], "float32")):
        R.func_attr({"global_symbol": "main"})
        split = R.split(x, 2)
        lhs = split[0]
        rhs = split[1]
        out = R.call_pure_packed(
            "my_optimized_add_impl", lhs, rhs, sinfo_args=R.Tensor([16], "float32")
        )
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_missing_pattern_raises_error():
    """The rewriter must define a pattern to be matched"""

    with pytest.raises(KeyError, match="pattern"):

        @R.rewriter
        class Rewriter:
            @R.function
            def replacement():
                return R.tuple()


def test_incorrect_function_type_of_pattern_raises_error():
    """The rewriter's pattern must be a Relax function"""

    with pytest.raises(TypeError, match="pattern"):

        @R.rewriter
        class Rewriter:
            @T.prim_func
            def pattern():
                pass

            @R.function
            def replacement():
                return R.tuple()


def test_missing_replacement_raises_error():
    """The rewriter must define a replacement"""

    with pytest.raises(KeyError, match="replacement"):

        @R.rewriter
        class Rewriter:
            @R.function
            def pattern():
                return R.tuple()


def test_incorrect_function_type_of_replacement_raises_error():
    """The rewriter's replacement must be a Relax function"""

    with pytest.raises(TypeError, match="replacement"):

        @R.rewriter
        class Rewriter:
            @R.function
            def pattern():
                return R.tuple()

            @T.prim_func
            def replacement():
                pass


def test_mismatch_of_static_shapes_raises_error():
    """The pattern and replacement must accept the same shapes"""

    with pytest.raises(ValueError, match="must have the same signature"):

        @R.rewriter
        class Rewriter:
            @R.function
            def pattern(A: R.Tensor([32])):
                return A

            @R.function
            def replacement(A: R.Tensor([16])):
                return A


def test_rewriter_may_be_applied_to_ir_module():
    """A rewriter may mutate an IRModule

    The `PatternMatchingRewriter.__call__` implementation may accept
    either a single Relax function, or an entire IRModule.  If it is
    passed an IRModule, then all functions in the `IRModule` are
    updated.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @I.ir_module
    class Before:
        @R.function
        def func_a(x: R.Tensor([32], "float32")):
            split = R.split(x, 2)
            lhs = split[0]
            rhs = split[1]
            out = lhs + rhs
            return out

        @R.function
        def func_b(x: R.Tensor([16], "float32")):
            out = x + x
            return out

    @I.ir_module
    class Expected:
        @R.function
        def func_a(x: R.Tensor([32], "float32")):
            split = R.split(x, 2)
            lhs = split[0]
            rhs = split[1]
            out = R.call_pure_packed(
                "my_optimized_add_impl", lhs, rhs, sinfo_args=R.Tensor([16], "float32")
            )
            return out

        @R.function
        def func_b(x: R.Tensor([16], "float32")):
            out = R.call_pure_packed(
                "my_optimized_add_impl", x, x, sinfo_args=R.Tensor([16], "float32")
            )
            return out

    After = Rewriter(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_rewriter_may_be_used_as_ir_transform():
    """A rewriter may be used as a tvm.ir.transform.Pass"""

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([16], "float32")):
            y = x + x
            return y

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([16], "float32")):
            out = R.call_pure_packed(
                "my_optimized_add_impl", x, x, sinfo_args=R.Tensor([16], "float32")
            )
            return out

    After = tvm.ir.transform.Sequential([Rewriter])(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_same_pattern_applied_multiple_times():
    """The pattern-match may apply multiple times"""

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @R.function(private=True)
    def before(x: R.Tensor([16], "float32")):
        y = x + x
        z = y + y
        return z

    @R.function(private=True)
    def expected(x: R.Tensor([16], "float32")):
        y = R.call_pure_packed("my_optimized_add_impl", x, x, sinfo_args=R.Tensor([16], "float32"))
        z = R.call_pure_packed("my_optimized_add_impl", y, y, sinfo_args=R.Tensor([16], "float32"))
        return z

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_composition_of_rewrite_rules():
    """Rewrite rules may be composed together"""

    @R.rewriter
    class RewriteAdd:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = A + B
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @R.rewriter
    class RewriteMultiply:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = A * B
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            C = R.call_pure_packed(
                "my_optimized_mul_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @R.function(private=True)
    def before(
        A: R.Tensor([16], "float32"),
        B: R.Tensor([16], "float32"),
        C: R.Tensor([16], "float32"),
    ):
        D = A + B
        E = C * D
        return E

    @R.function(private=True)
    def expected(
        A: R.Tensor([16], "float32"),
        B: R.Tensor([16], "float32"),
        C: R.Tensor([16], "float32"),
    ):
        D = R.call_pure_packed("my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32"))
        E = R.call_pure_packed("my_optimized_mul_impl", C, D, sinfo_args=R.Tensor([16], "float32"))
        return E

    rewriter = RewriteAdd | RewriteMultiply

    after = rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_recursive_rewrite_rules():
    """Rewrite rules are applied until convergence

    In this test, both the `RewriteAdd` and `RewriteMultiply` patterns
    must be applied in order to produce the expected output.  However,
    the `RewriteMultiply` pattern relies on the expression produced by
    the `RewriteAdd` pass.

    """

    @R.rewriter
    class RewriteAdd:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A + A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return A * R.const(2.0, "float32")

    @R.rewriter
    class RewriteMultiply:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([], "float32")):
            C = A * B
            return C

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([], "float32")):
            C = R.call_pure_packed(
                "my_optimized_mul_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            return C

    @R.function(private=True)
    def before(A: R.Tensor([16], "float32")):
        B = A + A
        return B

    @R.function(private=True)
    def expected(A: R.Tensor([16], "float32")):
        B = R.call_pure_packed(
            "my_optimized_mul_impl",
            A,
            R.const(2.0, "float32"),
            sinfo_args=R.Tensor([16], "float32"),
        )
        return B

    rewriter = RewriteAdd | RewriteMultiply

    after = rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_of_arbitrary_dtype():
    """A pattern-match may apply to a tensor with unknown dtype

    In this test case, a pattern identifies `R.strided_slice` usage
    which returns the last slice of an array, and replaces it with a
    view into the input array.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor(["M", "N"])) -> R.Tensor(["N"]):
            M = T.int64()
            N = T.int64()
            last_slice_2d: R.Tensor([1, N]) = R.strided_slice(A, axes=[0], begin=[M - 1], end=[M])
            last_slice_1d: R.Tensor([N]) = R.squeeze(last_slice_2d, axis=0)
            return last_slice_1d

        @R.function
        def replacement(A: R.Tensor(["M", "N"])) -> R.Tensor(["N"]):
            M = T.int64()
            N = T.int64()

            # TODO(Lunderberg): Improve this syntax.  A Relax
            # PrimValue (e.g. `A.dtype.bits`) should be usable in any
            # Relax context that accepts a `PrimExpr`.  Currently,
            # this requires `R.match_cast` to produce a TIR symbolic
            # variable from the Relax PrimValue.
            bits_per_element = T.uint8()
            _ = R.match_cast(
                A.dtype.bits,
                R.Prim(value=bits_per_element),
            )
            lanes_per_element = T.uint16()
            _ = R.match_cast(
                A.dtype.lanes,
                R.Prim(value=lanes_per_element),
            )

            last_slice = R.memory.view(
                A,
                [N],
                relative_byte_offset=(M - 1)
                * N
                * T.ceildiv(
                    bits_per_element.astype("int64") * lanes_per_element.astype("int64"), 8
                ),
            )
            return last_slice

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([32, 16], "float16"),
            B: R.Tensor(["P", "Q"], "int4x8"),
            C: R.Tensor([16, 32]),
        ):
            P = T.int64()
            Q = T.int64()

            A_slice_2d = R.strided_slice(A, axes=[0], begin=[31], end=[32])
            A_slice_1d = R.squeeze(A_slice_2d, axis=0)

            B_slice_2d = R.strided_slice(B, axes=[0], begin=[P - 1], end=[P])
            B_slice_1d = R.squeeze(B_slice_2d, axis=0)

            C_slice_2d = R.strided_slice(C, axes=[0], begin=[15], end=[16])
            C_slice_1d = R.squeeze(C_slice_2d, axis=0)

            return (A_slice_1d, B_slice_1d, C_slice_1d)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([32, 16], "float16"),
            B: R.Tensor(["P", "Q"], "int4x8"),
            C: R.Tensor([16, 32]),
        ):
            P = T.int64()
            Q = T.int64()

            # The pattern matches any 2-d tensor, with any data type.
            # When the match's shape and dtype are both known,
            # normalization and canonicalization produces a statically
            # known value for `relative_byte_offset`.
            #
            # Relative offset is `(31 rows) *
            #                     (16 elements/row) *
            #                     (2 bytes/element)`
            A_slice_1d = R.memory.view(A, shape=[16], relative_byte_offset=992)

            # The pattern can also match a 2-d tensor with dynamic
            # shape.  The `relative_byte_offset` uses the known
            # datatype (4 bytes for each int4x8), but with dynamic
            # shape variables substituted in where required.
            #
            # Relative offset is `((P-1) rows) *
            #                     (Q elements/row) *
            #                     (4 bytes/element)`
            B_slice_1d = R.memory.view(B, shape=[Q], relative_byte_offset=(P - 1) * Q * 4)

            # The pattern can also match a 2-d tensor with static
            # shape, but unknown data type.  The
            # `relative_byte_offset` is determined based on the known
            # number of elements, and the dynamic size of each
            # element.
            #
            # Relative offset is `(15 rows) *
            #                     (32 elements/row) *
            #                     (ceildiv(bits*lanes,8) bytes/element)`
            C_bits_per_element = T.uint8()
            C_bits_prim_value = C.dtype.bits
            _ = R.match_cast(
                C_bits_prim_value,
                R.Prim(value=C_bits_per_element),
            )
            C_lanes_per_element = T.uint16()
            C_lanes_prim_value = C.dtype.lanes
            _ = R.match_cast(
                C_lanes_prim_value,
                R.Prim(value=C_lanes_per_element),
            )

            C_slice_1d = R.memory.view(
                C,
                shape=[32],
                relative_byte_offset=(
                    (C_bits_per_element.astype("int64") * C_lanes_per_element.astype("int64") + 7)
                    // 8
                )
                * 480,
            )

            return (A_slice_1d, B_slice_1d, C_slice_1d)

    after = Rewriter(Before)
    tvm.ir.assert_structural_equal(Expected, after)


def test_rewrite_may_introduce_private_relax_subroutines():
    """The replacement may contain subroutines"""

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A + A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return Rewriter.subroutine(A)

        @R.function(private=True)
        def subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = A + A
            C = B + B
            return C

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = Expected.subroutine(A)
            C = Expected.subroutine(B)
            return C

        @R.function(private=True)
        def subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

    After = Rewriter(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_rewrite_only_introduces_private_subroutines_when_required():
    """Only subroutines that are used will be added to the module

    Like `test_rewrite_may_introduce_private_relax_subroutines`, but
    the rewritten function only requires some of the subroutines
    provided by the rewriter.

    """

    @R.rewriter
    class RewriteAdd:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A + A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return RewriteAdd.subroutine_add(A)

        @R.function(private=True)
        def subroutine_add(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

    @R.rewriter
    class RewriteMul:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A * A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return R.call_tir(RewriteMul.subroutine_mul, [A], out_sinfo=R.Tensor([16], "float32"))

        @T.prim_func(private=True)
        def subroutine_mul(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * A[i]

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = A + A
            C = B + B
            return C

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = Expected.subroutine_add(A)
            C = Expected.subroutine_add(B)
            return C

        @R.function(private=True)
        def subroutine_add(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

    rewriter = RewriteAdd | RewriteMul

    After = rewriter(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_rewriter_may_not_introduce_public_subroutines():
    """The rewriter may only introduce private functions"""

    with pytest.raises(ValueError, match="is publicly exposed"):

        @R.rewriter
        class Rewriter:
            @R.function
            def pattern(A: R.Tensor([16], "float32")):
                return A + A

            @R.function
            def replacement(A: R.Tensor([16], "float32")):
                return Rewriter.subroutine(A)

            @R.function
            def subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
                return A * R.const(2.0, "float32")


def test_rewrite_branches_may_reuse_subroutine_name():
    """Each rewriter is independent, and may reuse subroutine names"""

    @R.rewriter
    class RewriteAdd:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A + A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return RewriteAdd.subroutine(A)

        @R.function(private=True)
        def subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

    @R.rewriter
    class RewriteMul:
        @R.function
        def pattern(A: R.Tensor([16], "float32")):
            return A * A

        @R.function
        def replacement(A: R.Tensor([16], "float32")):
            return R.call_tir(RewriteMul.subroutine, [A], out_sinfo=R.Tensor([16], "float32"))

        @T.prim_func(private=True)
        def subroutine(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * A[i]

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = A + A
            C = B * B
            return C

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = Expected.subroutine(A)
            C = R.call_tir(Expected.subroutine_1, [B], out_sinfo=R.Tensor([16], "float32"))
            return C

        @R.function(private=True)
        def subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            return A * R.const(2.0, "float32")

        @T.prim_func(private=True)
        def subroutine_1(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                B[i] = A[i] * A[i]

    rewriter = RewriteAdd | RewriteMul

    After = rewriter(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_rewrite_of_explicit_relax_tuple():
    """The rewriter function may return a tuple

    When it occurs explicitly within the Relax function, the tuple
    pattern matches against the Relax tuple, and the Relax tuple is
    replaced.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            proj_A = R.matmul(lhs_A, rhs)
            proj_B = R.matmul(lhs_B, rhs)
            proj_tuple = (proj_A, proj_B)
            return proj_tuple

        @R.function
        def replacement(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            lhs = R.concat([lhs_A, lhs_B])
            proj_concat = R.matmul(lhs, rhs)
            proj_tuple = R.split(proj_concat, 2)
            return proj_tuple

    @R.function(private=True)
    def before(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        proj_A = R.matmul(A, state)
        proj_B = R.matmul(B, state)
        proj_tuple = (proj_A, proj_B)
        out = proj_tuple[0] + proj_tuple[1]
        return out

    @R.function(private=True)
    def expected(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        concat_AB = R.concat([A, B])
        proj_concat = R.matmul(concat_AB, state)
        proj_tuple = R.split(proj_concat, 2)
        out = proj_tuple[0] + proj_tuple[1]
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_of_output_relax_tuple():
    """The rewriter may update a tuple being returned

    Unlike most relax expressions, tuples may appear as nested
    expressions.  Pattern-matching should be aware of this option.

    Like `test_rewrite_of_explicit_relax_tuple`, but the tuple appears
    as the return value in the function being modified.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            proj_A = R.matmul(lhs_A, rhs)
            proj_B = R.matmul(lhs_B, rhs)
            proj_tuple = (proj_A, proj_B)
            return proj_tuple

        @R.function
        def replacement(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            lhs = R.concat([lhs_A, lhs_B])
            proj_concat = R.matmul(lhs, rhs)
            proj_tuple = R.split(proj_concat, 2)
            return proj_tuple

    @R.function(private=True)
    def before(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        proj_A = R.matmul(A, state)
        proj_B = R.matmul(B, state)
        return (proj_A, proj_B)

    @R.function(private=True)
    def expected(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        concat_AB = R.concat([A, B])
        proj_concat = R.matmul(concat_AB, state)
        proj_tuple = R.split(proj_concat, 2)
        return proj_tuple

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_of_implicit_tuple():
    """The rewriter function may return a tuple

    The tuple being replaced does not need to explicitly exist within
    the updated Relax function.  So long as each element of the tuple
    pattern matches a Relax expression, the pattern match can apply.

    This rule ensures that pattern-matching is never broken when
    `CanonicalizeBindings` is applied.

    This test is identical to `test_rewrite_of_explicit_relax_tuple`,
    except that the function does not contain the round trip of
    packing `proj_A` and `proj_B` into a tuple, then immediately
    unpacking them from the tuple.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            proj_A = R.matmul(lhs_A, rhs)
            proj_B = R.matmul(lhs_B, rhs)
            proj_tuple = (proj_A, proj_B)
            return proj_tuple

        @R.function
        def replacement(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16], "float32"),
        ):
            lhs = R.concat([lhs_A, lhs_B])
            proj_concat = R.matmul(lhs, rhs)
            proj_tuple = R.split(proj_concat, 2)
            return proj_tuple

    @R.function(private=True)
    def before(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        proj_A = R.matmul(A, state)
        proj_B = R.matmul(B, state)
        out = proj_A + proj_B
        return out

    @R.function(private=True)
    def expected(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        concat_AB = R.concat([A, B])
        proj_concat = R.matmul(concat_AB, state)
        proj_tuple = R.split(proj_concat, 2)
        out = proj_tuple[0] + proj_tuple[1]
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_of_implicit_tuple_with_shared_wildcard():
    """Tuple elements may depend on the same input

    Here, both elements of the tuple depend on `y`.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            x: R.Tensor([16], "float32"),
            y: R.Tensor([16], "float32"),
            z: R.Tensor([16], "float32"),
        ):
            lhs = x + y
            rhs = y + z
            return (lhs, rhs)

        @R.function
        def replacement(
            x: R.Tensor([16], "float32"),
            y: R.Tensor([16], "float32"),
            z: R.Tensor([16], "float32"),
        ):
            return R.call_pure_packed(
                "optimized_impl",
                x,
                y,
                z,
                sinfo_args=R.Tuple(
                    [
                        R.Tensor([16], "float32"),
                        R.Tensor([16], "float32"),
                    ]
                ),
            )

    @R.function(private=True)
    def before(
        A: R.Tensor([16], "float32"),
        B: R.Tensor([16], "float32"),
        C: R.Tensor([16], "float32"),
    ):
        lhs = A + B
        rhs = B + C
        out = R.multiply(lhs, rhs)
        return out

    @R.function(private=True)
    def expected(
        A: R.Tensor([16], "float32"),
        B: R.Tensor([16], "float32"),
        C: R.Tensor([16], "float32"),
    ):
        lhs_rhs = R.call_pure_packed(
            "optimized_impl",
            A,
            B,
            C,
            sinfo_args=R.Tuple(
                [
                    R.Tensor([16], "float32"),
                    R.Tensor([16], "float32"),
                ]
            ),
        )
        out = R.multiply(lhs_rhs[0], lhs_rhs[1])
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_no_rewrite_of_implicit_tuple_when_shared_wildcard_is_mismatched():
    """Tuple elements must match simultaneously

    Each element of the tuple matches individually, but the two
    elements both depend on `B`.  Because the first tuple element
    would require `y = B`, while the second tuple element would
    require `y = C`, the match fails.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            x: R.Tensor([16], "float32"),
            y: R.Tensor([16], "float32"),
            z: R.Tensor([16], "float32"),
        ):
            lhs = x + y
            rhs = y + z
            return (lhs, rhs)

        @R.function
        def replacement(
            A: R.Tensor([16], "float32"),
            B: R.Tensor([16], "float32"),
            C: R.Tensor([16], "float32"),
        ):
            return R.call_pure_packed(
                "optimized_impl",
                A,
                B,
                C,
                sinfo_args=R.Tuple(
                    [
                        R.Tensor([16], "float32"),
                        R.Tensor([16], "float32"),
                    ]
                ),
            )

    @R.function(private=True)
    def before(
        A: R.Tensor([16], "float32"),
        B: R.Tensor([16], "float32"),
        C: R.Tensor([16], "float32"),
        D: R.Tensor([16], "float32"),
    ):
        lhs = A + B
        rhs = C + D
        out = R.multiply(lhs, rhs)
        return out

    expected = before

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_implicit_tuple_may_not_introduce_extra_compute():
    """Matching of implicit tuple may not cause extra compute

    Here, the `(proj_A, proj_B)` tuple could be an implcit tuple
    match, but that would repeat the computation of `proj_A`.  It
    would be computed once on its own, to be used for `proj_A_on_B`,
    and once for computing `(proj_A, proj_B)`.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16, 16], "float32"),
        ):
            proj_A = R.matmul(lhs_A, rhs)
            proj_B = R.matmul(lhs_B, rhs)
            proj_tuple = (proj_A, proj_B)
            return proj_tuple

        @R.function
        def replacement(
            lhs_A: R.Tensor([16, 16], "float32"),
            lhs_B: R.Tensor([16, 16], "float32"),
            rhs: R.Tensor([16, 16], "float32"),
        ):
            lhs = R.concat([lhs_A, lhs_B])
            proj_concat = R.matmul(lhs, rhs)
            proj_tuple = R.split(proj_concat, 2)
            return proj_tuple

    @R.function(private=True)
    def before(
        state: R.Tensor([16, 16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        # This function has no location at which a tuple
        # `(proj_A,proj_B)` could be constructed, then unpacked.

        proj_A = R.matmul(A, state)

        # A tuple `(proj_A, proj_B)` could not be constructed at this
        # location, because `proj_B` has not yet been computed.

        proj_A_on_B = R.matmul(proj_A, B)
        proj_B = R.matmul(proj_A_on_B, state)

        # A tuple `(proj_A, proj_B)` could be constructed here, but a
        # use-site of `proj_A` has already occurred.  Implicit
        # matching of a tuple is only allowed if it would replace
        # every use-site of a variable.

        out = proj_A + proj_B
        return out

    expected = before

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_of_implicit_tuple_with_three_elements():
    """Implicit tuples may contain three elements"""

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(qkv: R.Tensor([12288], "float32")):
            qkv_tuple = R.split(qkv, 3, axis=0)
            q = qkv_tuple[0]
            k = qkv_tuple[1]
            v = qkv_tuple[2]
            q_embed = R.call_pure_packed(
                "rotary_embedding", [q], sinfo_args=R.Tensor([4096], "float32")
            )
            k_embed = R.call_pure_packed(
                "rotary_embedding", [k], sinfo_args=R.Tensor([4096], "float32")
            )

            return (q_embed, k_embed, v)

        @R.function
        def replacement(qkv: R.Tensor([12288], "float32")):
            return R.call_pure_packed(
                "split_rotary_embedding",
                [qkv],
                sinfo_args=[
                    R.Tensor([4096], "float32"),
                    R.Tensor([4096], "float32"),
                    R.Tensor([4096], "float32"),
                ],
            )

    @R.function(private=True)
    def before(
        state: R.Tensor([4096], "float32"),
        proj_qkv: R.Tensor([12288, 4096], "float32"),
        kv_cache: R.Object,
    ):
        qkv = R.matmul(proj_qkv, state)
        qkv_tuple = R.split(qkv, 3, axis=0)
        q = qkv_tuple[0]
        k = qkv_tuple[1]
        v = qkv_tuple[2]
        q_embed = R.call_pure_packed(
            "rotary_embedding", [q], sinfo_args=R.Tensor([4096], "float32")
        )
        k_embed = R.call_pure_packed(
            "rotary_embedding", [k], sinfo_args=R.Tensor([4096], "float32")
        )

        attention = R.call_pure_packed(
            "compute_self_attention",
            [q_embed, k_embed, v, kv_cache],
            sinfo_args=R.Tensor([4096]),
        )

        return attention

    @R.function(private=True)
    def expected(
        state: R.Tensor([4096], "float32"),
        proj_qkv: R.Tensor([12288, 4096], "float32"),
        kv_cache: R.Object,
    ):
        qkv = R.matmul(proj_qkv, state)
        embedded_qkv_tuple = R.call_pure_packed(
            "split_rotary_embedding",
            [qkv],
            sinfo_args=[
                R.Tensor([4096], "float32"),
                R.Tensor([4096], "float32"),
                R.Tensor([4096], "float32"),
            ],
        )

        v = embedded_qkv_tuple[2]
        q_embed = embedded_qkv_tuple[0]
        k_embed = embedded_qkv_tuple[1]

        attention = R.call_pure_packed(
            "compute_self_attention",
            [q_embed, k_embed, v, kv_cache],
            sinfo_args=R.Tensor([4096]),
        )

        return attention

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_pattern_matching_may_not_reorder_across_impure_functions():
    """Matched pattern must be ordered with respect to impure functions

    To ensure that debug printouts, memory management, performance
    measurements, etc are not impacted by a pattern match, a pattern
    must be entirely before, or entirely after an impure function.  A
    pattern match in which some parts of the matched expression are
    performed before an impure function, while others are performed
    afterwards, is not allowed.

    In this test, the matmul and the add may not be fused, because the
    impure print statement occurs between them.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            state: R.Tensor([16], "float32"),
            weights: R.Tensor([16, 16], "float32"),
            bias: R.Tensor([16], "float32"),
        ):
            state = R.matmul(weights, state)
            state = R.add(bias, state)
            return state

        @R.function
        def replacement(
            state: R.Tensor([16], "float32"),
            weights: R.Tensor([16, 16], "float32"),
            bias: R.Tensor([16], "float32"),
        ):
            return R.call_pure_packed(
                "my_optimized_fma_impl",
                state,
                weights,
                bias,
                sinfo_args=R.Tensor([16], "float32"),
            )

    @R.function(private=True, pure=False)
    def before(
        state: R.Tensor([16], "float32"),
        weights: R.Tensor([16, 16], "float32"),
        bias: R.Tensor([16], "float32"),
    ):
        R.print(format="Start of function")
        state = R.matmul(weights, state)
        R.print(format="After matmul, before add")
        state = R.add(bias, state)
        R.print(format="End of function")
        return state

    expected = before

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_pattern_matching_may_occur_between_impure_functions():
    """Matched pattern may be adjacent to impure functions

    To ensure that debug printouts, memory management, performance
    measurements, etc are not impacted by a pattern match, a pattern
    must be entirely before, or entirely after an impure function.  A
    pattern match in which some parts of the matched expression are
    performed before an impure function, while others are performed
    afterwards, is not allowed.

    In this test, the matmul and the add may be fused, because the
    pattern occurs without an impure print statement in-between.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            state: R.Tensor([16], "float32"),
            weights: R.Tensor([16, 16], "float32"),
            bias: R.Tensor([16], "float32"),
        ):
            state = R.matmul(weights, state)
            state = R.add(bias, state)
            return state

        @R.function
        def replacement(
            state: R.Tensor([16], "float32"),
            weights: R.Tensor([16, 16], "float32"),
            bias: R.Tensor([16], "float32"),
        ):
            return R.call_pure_packed(
                "my_optimized_fma_impl",
                state,
                weights,
                bias,
                sinfo_args=R.Tensor([16], "float32"),
            )

    @R.function(private=True, pure=False)
    def before(
        state: R.Tensor([16], "float32"),
        weights: R.Tensor([16, 16], "float32"),
        bias: R.Tensor([16], "float32"),
    ):
        R.print(format="Start of function")
        state = R.matmul(weights, state)
        state = R.add(bias, state)
        R.print(format="End of function")
        return state

    @R.function(private=True, pure=False)
    def expected(
        state: R.Tensor([16], "float32"),
        weights: R.Tensor([16, 16], "float32"),
        bias: R.Tensor([16], "float32"),
    ):
        R.print(format="Start of function")
        state = R.call_pure_packed(
            "my_optimized_fma_impl",
            state,
            weights,
            bias,
            sinfo_args=R.Tensor([16], "float32"),
        )
        R.print(format="End of function")
        return state

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_rewrite_may_apply_within_conditional():
    """Rewrites may apply within to inner dataflow regions

    While dataflow regions may not contain conditionals, they may
    occur within the body of conditionals.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            return A + B

        @R.function
        def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
            return R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )

    @R.function(private=True)
    def before(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32"), cond: R.Prim("bool")):
        if cond:
            out = A + B
        else:
            C = A + B
            out = C + B
        return out

    @R.function(private=True)
    def expected(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32"), cond: R.Prim("bool")):
        if cond:
            out = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
        else:
            C = R.call_pure_packed(
                "my_optimized_add_impl", A, B, sinfo_args=R.Tensor([16], "float32")
            )
            out = R.call_pure_packed(
                "my_optimized_add_impl", C, B, sinfo_args=R.Tensor([16], "float32")
            )
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_match_dynamic_shape():
    """Pattern match/rewrites may be dynamic

    The tuple being replaced does not need to explicitly exist within
    the updated Relax function.  So long as each element of the tuple
    pattern matches a Relax expression, the pattern match can apply.

    This rule ensures that pattern-matching is never broken when
    `CanonicalizeBindings` is applied.

    This test is identical to `test_rewrite_of_explicit_relax_tuple`,
    except that the function does not contain the round trip of
    packing `proj_A` and `proj_B` into a tuple, then immediately
    unpacking them from the tuple.

    """

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            lhs_A: R.Tensor(["N1", "M"], "float32"),
            lhs_B: R.Tensor(["N2", "M"], "float32"),
            rhs: R.Tensor(["M"], "float32"),
        ):
            proj_A = R.matmul(lhs_A, rhs)
            proj_B = R.matmul(lhs_B, rhs)
            return (proj_A, proj_B)

        @R.function
        def replacement(
            lhs_A: R.Tensor(["N1", "M"], "float32"),
            lhs_B: R.Tensor(["N2", "M"], "float32"),
            rhs: R.Tensor(["M"], "float32"),
        ):
            N1 = T.int64()
            N2 = T.int64()

            lhs = R.concat([lhs_A, lhs_B])
            proj_concat = R.matmul(lhs, rhs)
            proj_A: R.Tensor([N1], "float32") = R.strided_slice(
                proj_concat, axes=[0], begin=[0], end=[N1]
            )
            proj_B: R.Tensor([N2], "float32") = R.strided_slice(
                proj_concat, axes=[0], begin=[N1], end=[N2 + N1]
            )
            return (proj_A, proj_B)

    @R.function(private=True)
    def before(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        proj_A = R.matmul(A, state)
        proj_B = R.matmul(B, state)
        out = proj_A + proj_B
        return out

    @R.function(private=True)
    def expected(
        state: R.Tensor([16], "float32"),
        A: R.Tensor([16, 16], "float32"),
        B: R.Tensor([16, 16], "float32"),
    ):
        concat_AB = R.concat([A, B])
        proj_concat = R.matmul(concat_AB, state)
        proj_A = R.strided_slice(proj_concat, axes=[0], begin=[0], end=[16])
        proj_B = R.strided_slice(proj_concat, axes=[0], begin=[16], end=[32])
        out = proj_A + proj_B
        return out

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_match_dynamic_pattern_against_dynamic_shape():
    """A dynamic pattern may match a static shape"""

    @R.rewriter
    class Rewriter:
        @R.function
        def pattern(
            A: R.Tensor(["M", "N"], "float32"),
            B: R.Tensor(["N", "N"], "float32"),
        ):
            return R.matmul(A, B)

        @R.function
        def replacement(
            A: R.Tensor(["M", "N"], "float32"),
            B: R.Tensor(["N", "N"], "float32"),
        ):
            M = T.int64()
            N = T.int64()
            return R.call_pure_packed(
                "my_optimized_square_matmul",
                A,
                B,
                sinfo_args=R.Tensor([M, N], "float32"),
            )

    @R.function(private=True)
    def before(
        A: R.Tensor(["N", "N*2"], "float32"),
        B: R.Tensor(["N*2", "N*2"], "float32"),
        C: R.Tensor(["N", "N"], "float32"),
    ):
        N = T.int64()
        D: R.Tensor([N, N * 2], "float32") = R.matmul(A, B)
        E: R.Tensor([N * 2, N], "float32") = R.permute_dims(D)
        F: R.Tensor([N * 2, N], "float32") = R.matmul(E, C)
        return F

    @R.function(private=True)
    def expected(
        A: R.Tensor(["N", "N*2"], "float32"),
        B: R.Tensor(["N*2", "N*2"], "float32"),
        C: R.Tensor(["N", "N"], "float32"),
    ):
        N = T.int64()

        D: R.Tensor([N, N * 2], "float32") = R.call_pure_packed(
            "my_optimized_square_matmul",
            A,
            B,
            sinfo_args=R.Tensor([N, N * 2], "float32"),
        )
        E: R.Tensor([N * 2, N], "float32") = R.permute_dims(D)
        F: R.Tensor([N * 2, N], "float32") = R.call_pure_packed(
            "my_optimized_square_matmul",
            E,
            C,
            sinfo_args=R.Tensor([N * 2, N], "float32"),
        )
        return F

    after = Rewriter(before)
    tvm.ir.assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm.testing.main()
