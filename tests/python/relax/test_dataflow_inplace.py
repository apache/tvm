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

from typing import List, Set, Tuple
import tvm
from tvm import relax, testing
from tvm.relax.transform import DataflowUseInplaceCalls
from tvm.relax.testing.transform import (
    dataflow_liveness_analysis,
    dataflow_alias_analysis,
    dataflow_inplace_analysis,
    dataflow_single_inplace_call,
)
from tvm.script.parser import ir as I, relax as R, tir as T

import numpy as np


def test_liveness_analysis():
    @I.ir_module
    class BasicLiveness:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1, dtype="int32")
                z = R.add(x, y)
                q = R.multiply(z, y)
                p = R.add(z, q)
                n = R.multiply(p, p)
                R.output(n, p)
            return n

    block = BasicLiveness["main"].body.blocks[0]
    live_ranges = dataflow_liveness_analysis(block)
    expected_ranges = {
        # x is live past the binding block
        "x": (-1, 5),
        "y": (0, 2),
        "z": (1, 3),
        "q": (2, 3),
        # exposed though ultimately not used
        "p": (3, 5),
        "n": (4, 5),
    }
    actual_ranges = {var.name_hint: live_range for var, live_range in live_ranges.items()}
    assert actual_ranges == expected_ranges


def test_alias_analysis_basic():
    @I.ir_module
    class BasicAliasAnalysis:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = x  # y is an alias of x
                z = R.add(y, y)  # fresh value
                n = z  # alias of z
                R.output(n)
            return n

    block = BasicAliasAnalysis["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, BasicAliasAnalysis["main"].params)
    expected = {
        "x": {0},
        "y": {0},
        "z": {1},
        "n": {1},
    }

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name_hint]
    assert tuple_map == {}


def test_alias_analysis_tuple():
    @I.ir_module
    class AliasesWithTuples:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1, dtype="int32")
                t = (x, y)
                a = t[0]
                b = t[1]
                c = t[0]
                d = t[1]
                u = t
                e = t[0]
                f = t[1]
                z = R.add(c, d)
                n = z
                R.output(n)
            return n

    block = AliasesWithTuples["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasesWithTuples["main"].params)
    expected = {
        "x": {0},
        "y": {1},
        "t": {2},
        "a": {0},
        "b": {1},
        "c": {0},
        "d": {1},
        "u": {2},
        "e": {0},
        "f": {1},
        "z": {3},
        "n": {3},
    }

    actual_alias_sets = {var.name_hint: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert 2 in tuple_map
    assert tuple_map[2] == [{0}, {1}]


def test_alias_split():
    @I.ir_module
    class AliasSplit:
        @R.function
        def main(x: R.Tensor((60,), "int32")) -> R.Tensor((15,), "int32"):
            with R.dataflow():
                t = R.split(x, 4)
                y = t[0]
                z = t[1]
                q = t[2]
                p = t[3]
                n = z
                R.output(n)
            return n

    block = AliasSplit["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasSplit["main"].params)
    expected = {
        "x": {0},
        "t": {1},
        "y": {2},
        "z": {3},
        "q": {4},
        "p": {5},
        "n": {3},
    }

    actual_alias_sets = {var.name_hint: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 1 in tuple_map
    assert tuple_map[1] == [{2}, {3}, {4}, {5}]


def test_alias_call_tir():
    # call TIR can yield either a single tensor or a tuple
    @I.ir_module
    class AliasCallTir:
        @T.prim_func
        def tir_id(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_id"})
            m = T.int32()
            n = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (m, n))

            for i, j in T.grid(m, n):
                with T.block("id"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @T.prim_func
        def tir_id2(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_id"})
            m = T.int32()
            n = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (m, n))
            C = T.match_buffer(z, (m, n))

            for i, j in T.grid(m, n):
                with T.block("id"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]
                    C[vi, vj] = A[vi, vj]

        @R.function
        def main(x: R.Tensor((10, 10), "int32")) -> R.Tensor((10, 10), "int32"):
            with R.dataflow():
                cls = AliasCallTir
                y = R.call_tir(cls.tir_id, (x,), out_sinfo=R.Tensor((10, 10), "int32"))
                t = R.call_tir(
                    cls.tir_id2,
                    (y,),
                    out_sinfo=[R.Tensor((10, 10), "int32"), R.Tensor((10, 10), "int32")],
                )
                z = y
                p = t[0]
                q = t[1]
                u = t
                m = u[0]
                n = u[1]
                v = n
                R.output(v)
            return v

    block = AliasCallTir["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasCallTir["main"].params)
    expected = {
        "x": {0},
        "y": {1},
        "t": {2},
        "z": {1},
        "p": {3},
        "q": {4},
        "u": {2},
        "m": {3},
        "n": {4},
        "v": {4},
    }

    actual_alias_sets = {var.name_hint: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 2 in tuple_map
    assert tuple_map[2] == [{3}, {4}]


def test_mystery_calls():
    @I.ir_module
    class AliasChaosCalls:
        @R.function
        def identity(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            return x

        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                cls = AliasChaosCalls
                y = cls.identity(x)
                z = cls.identity(y)
                m = R.const(1, dtype="int32")
                n = R.const(2, dtype="int32")
                t = (m, n)
                a = R.call_pure_packed(
                    "chaos", t, sinfo_args=R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32"))
                )
                b = a[0]
                c = a[1]
                R.output(c)
            return c

    block = AliasChaosCalls["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasChaosCalls["main"].params)
    expected = {
        "x": {0},
        "y": {0, 1},
        "z": {0, 1, 2},
        "m": {3},
        "n": {4},
        "t": {5},
        "a": {3, 4, 5, 6, 7, 8},  # either t or a fresh tuple
        "b": {3, 4, 5, 6, 7, 8},  # the tuple components can be aliased to any member...
        "c": {3, 4, 5, 6, 7, 8},  # the tuple components can be aliased to any member...
        # (in principle, we can use type information to narrow down the aliasing)
    }

    actual_alias_sets = {var.name_hint: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 2
    assert 5 in tuple_map
    assert tuple_map[5] == [{3}, {4}]
    assert 6 in tuple_map
    assert tuple_map[6] == [{3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8}]


def test_alias_external_value():
    @I.ir_module
    class AliasExternalValue:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.const(1, dtype="int32")  # not in DF block, treated as external
            t1 = (y, y)  # not in DF block, treated as external
            with R.dataflow():
                z = y  # mystery value
                a = R.const(2, dtype="int32")
                t2 = (z, a)
                b = t2[0]
                c = t1[1]  # tuple index into external value
                R.output(b)
            return b

    block = AliasExternalValue["main"].body.blocks[1]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasExternalValue["main"].params)
    expected = {
        "x": {0},
        "z": {-1},
        "a": {1},
        "t2": {2},
        "b": {-1},
        "c": {-1},
    }

    actual_alias_sets = {var.name_hint: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 2 in tuple_map
    assert tuple_map[2] == [{-1}, {1}]


def test_inplace_simple_case():
    @I.ir_module
    class InplaceBasic:
        @R.function
        def main(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")
        ) -> R.Tensor((2, 3), "int32"):
            with R.dataflow():
                z = R.add(x, y)  # cannot be done inplace: x and y are live later
                p = R.add(z, z)  # can be done inplace: z is not used later
                r = p  # alias of p
                m = R.multiply(p, p)  # p is not used later but r is, so can't do inplace
                n = R.add(m, r)  # can be done inplace: r is not used again
                ret = R.subtract(n, m)  # can be done inplace: neither is used again
                R.output(ret)
            return ret

    block = InplaceBasic["main"].body.blocks[0]
    size_match, exact_match = dataflow_inplace_analysis(
        block, InplaceBasic["main"].params, InplaceBasic
    )

    # order does not matter for the listing of candidates, so we have to implement as sets
    def assert_candidate_list(
        actual: List[Tuple[int, Set[int]]], expected: List[Tuple[int, Set[int]]]
    ) -> None:
        assert len(actual) == len(expected)
        for i in range(len(actual)):
            assert actual[i][0] == expected[i][0]
            assert len(expected[i][1]) == len(actual[i][1])
            for idx in actual[i][1]:
                assert idx in expected[i][1]

    assert_candidate_list(size_match, [(1, {0, 1}), (4, {1}), (5, {0, 1})])
    # TODO(@slyubomirsky): I couldn't think of an easy example where sizes don't match,
    # but broadcasting might cause it to happen
    assert_candidate_list(exact_match, [(1, {0, 1}), (4, {1}), (5, {0, 1})])


def test_inplace_single_call():
    @I.ir_module
    class TestModule:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            z = R.add(x, y)
            q = R.nn.silu(z)
            return q

    add_call = TestModule["main"].body.blocks[0].bindings[0].value
    new_add, new_mod = dataflow_single_inplace_call(TestModule, add_call, [0])

    @T.prim_func(private=True)
    def expected_add(
        A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(A[v_ax0, v_ax1])
                A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    tvm.ir.assert_structural_equal(new_mod["add_inplace"], expected_add)
    assert new_add.op.name == "relax.call_tir_inplace"
    assert new_add.args[0].name_hint == "add_inplace"
    for i, arg in enumerate(new_add.args[1].fields):
        arg == add_call.args[i]
    new_add.attrs.inplace_indices == [0]

    @T.prim_func(private=True)
    def expected_silu(A: T.Buffer((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        compute = T.alloc_buffer((T.int64(2), T.int64(3)))
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(A[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(A[v_ax0, v_ax1])
                A[v_ax0, v_ax1] = A[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

    silu_call = TestModule["main"].body.blocks[0].bindings[1].value
    new_silu, new_mod = dataflow_single_inplace_call(TestModule, silu_call, [0])

    tvm.ir.assert_structural_equal(new_mod["silu_inplace"], expected_silu)
    assert new_silu.op.name == "relax.call_tir_inplace"
    assert new_silu.args[0].name_hint == "silu_inplace"
    for i, arg in enumerate(new_silu.args[1].fields):
        arg == silu_call.args[i]
    new_silu.attrs.inplace_indices == [0]


def test_insert_inplace_calls():
    @I.ir_module
    class EndToEndTest:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                z = R.add(x, y)  # broadcast happens here
                # Cannot be done in-place because x is an argument.
                a = R.add(z, y)  # this one can be done in-place
                q = R.multiply(a, y)  # broadcast again, a is eligible
                r = R.subtract(y, y)  # cannot be done in-place because y is an argument
                s = R.subtract(r, r)  # No broadcast. Can be done in-place
                m = R.multiply(q, s)  # should give us all zeros
                R.output(m)
            return m

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def add_inplace(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[T.int64(0), v_ax1]

        @T.prim_func(private=True)
        def multiply_inplace(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[T.int64(0), v_ax1]

        @T.prim_func(private=True)
        def subtract_inplace(
            A: T.Buffer((T.int64(1), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(1), T.int64(3)):
                with T.block("T_subtract"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    T.writes(B[v_ax0, v_ax1])
                    B[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            cls = Expected
            with R.dataflow():
                z: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
                a: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.add_inplace,
                    (z, y),
                    inplace_indices=[0],
                    out_sinfo=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                q: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (a, y),
                    inplace_indices=[0],
                    out_sinfo=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                r: R.Tensor((1, 3), dtype="float32") = R.subtract(y, y)
                s: R.Tensor((1, 3), dtype="float32") = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (r, r),
                    inplace_indices=[1],
                    out_sinfo=[
                        R.Tensor((1, 3), dtype="float32"),
                    ],
                )
                m: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (q, s),
                    inplace_indices=[0],
                    out_sinfo=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                R.output(m)
            return m

    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(EndToEndTest)
    tvm.ir.assert_structural_equal(new_mod, Expected)

    x = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    y = tvm.nd.array(np.random.rand(1, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm.target.Target("llvm")
    ex = relax.build(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic():
    @I.ir_module
    class DynamicTestCase:
        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32"), y: R.Tensor(("a", "b"), dtype="float32")
        ) -> R.Tensor(("a", "b"), dtype="float32"):
            with R.dataflow():
                z = R.add(x, y)
                # Cannot be done in-place because x and y are arguments
                a = R.add(z, y)  # this one can be done in-place
                s = R.subtract(a, a)  # No broadcast. Can be done in-place
                R.output(s)
            return s

    # the result should be all zeroes
    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(DynamicTestCase)

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def add_inplace(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b = T.int64(), T.int64()
            A = T.match_buffer(var_A, (a, b))
            B = T.match_buffer(var_B, (a, b))
            for ax0, ax1 in T.grid(a, b):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

        @T.prim_func(private=True)
        def subtract_inplace(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b = T.int64(), T.int64()
            A = T.match_buffer(var_A, (a, b))
            B = T.match_buffer(var_B, (a, b))
            for ax0, ax1 in T.grid(a, b):
                with T.block("T_subtract"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    T.writes(B[v_ax0, v_ax1])
                    B[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32"), y: R.Tensor(("a", "b"), dtype="float32")
        ) -> R.Tensor(("a", "b"), dtype="float32"):
            a = T.int64()
            b = T.int64()
            cls = Expected
            with R.dataflow():
                z = R.add(x, y)
                a_1 = R.call_tir_inplace(
                    cls.add_inplace,
                    (z, y),
                    out_sinfo=R.Tensor((a, b), dtype="float32"),
                    inplace_indices=[0],
                )
                s = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (a_1, a_1),
                    out_sinfo=R.Tensor((a, b), dtype="float32"),
                    inplace_indices=[1],
                )
                R.output(s)
            return s

    tvm.ir.assert_structural_equal(new_mod, Expected, map_free_vars=True)
    x = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    y = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm.target.Target("llvm")
    ex = relax.build(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic_mismatch():
    # cannot statically prove the shapes to be equal so the module should be unchanged
    @I.ir_module
    class DynamicMistmatchTestCase:
        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32"), y: R.Tensor(("c", "d"), dtype="float32")
        ):
            with R.dataflow():
                z = R.add(x, y)
                # Cannot be done in-place because x and y are arguments
                a = R.add(z, y)  # cannot conclude that shapes match
                R.output(a)
            return a

    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(DynamicMistmatchTestCase)
    tvm.ir.assert_structural_equal(new_mod, DynamicMistmatchTestCase)


if __name__ == "__main__":
    testing.main()
