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
# ruff: noqa: F841


import numpy as np
import pytest
import torch

import tvm
from tvm import relax, testing
from tvm.relax import VMInstrumentReturnKind
from tvm.relax.testing.transform import (
    dataflow_alias_analysis,
    dataflow_inplace_analysis,
    dataflow_liveness_analysis,
    dataflow_single_inplace_call,
)
from tvm.relax.transform import DataflowUseInplaceCalls
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tirx as T


def test_liveness_analysis():
    @I.ir_module(s_tir=True)
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
    @I.ir_module(s_tir=True)
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
    @I.ir_module(s_tir=True)
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
    @I.ir_module(s_tir=True)
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
    @I.ir_module(s_tir=True)
    class AliasCallTir:
        @T.prim_func(s_tir=True)
        def tir_id(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_id"})
            m = T.int32()
            n = T.int32()
            A = T.match_buffer(x, (m, n), "int32")
            B = T.match_buffer(y, (m, n), "int32")

            for i, j in T.grid(m, n):
                with T.sblock("id"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @T.prim_func(s_tir=True)
        def tir_id2(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_id"})
            m = T.int32()
            n = T.int32()
            A = T.match_buffer(x, (m, n), "int32")
            B = T.match_buffer(y, (m, n), "int32")
            C = T.match_buffer(z, (m, n), "int32")

            for i, j in T.grid(m, n):
                with T.sblock("id"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]
                    C[vi, vj] = A[vi, vj]

        @R.function
        def main(x: R.Tensor((10, 10), "int32")) -> R.Tensor((10, 10), "int32"):
            with R.dataflow():
                cls = AliasCallTir
                y = R.call_tir(cls.tir_id, (x,), out_ty=R.Tensor((10, 10), "int32"))
                t = R.call_tir(
                    cls.tir_id2,
                    (y,),
                    out_ty=[R.Tensor((10, 10), "int32"), R.Tensor((10, 10), "int32")],
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
    @I.ir_module(s_tir=True)
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
                    "chaos", t, ty_args=R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32"))
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
    @I.ir_module(s_tir=True)
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
    @I.ir_module(s_tir=True)
    class InplaceBasic:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")) -> R.Tensor(
            (2, 3), "int32"
        ):
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
        actual: list[tuple[int, set[int]]], expected: list[tuple[int, set[int]]]
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
    @I.ir_module(s_tir=True)
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

    @T.prim_func(private=True, s_tir=True)
    def expected_add(
        A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with T.sblock("T_add"):
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

    @T.prim_func(private=True, s_tir=True)
    def expected_silu(A: T.Buffer((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"tirx.noalias": True})
        compute = T.sblock_alloc_buffer((T.int64(2), T.int64(3)))
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(A[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with T.sblock("T_multiply"):
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
    @I.ir_module(s_tir=True)
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

    @I.ir_module(s_tir=True)
    class Expected:
        @T.prim_func(private=True, s_tir=True)
        def add_inplace(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[T.int64(0), v_ax1]

        @T.prim_func(private=True, s_tir=True)
        def multiply_inplace(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[T.int64(0), v_ax1]

        @T.prim_func(private=True, s_tir=True)
        def subtract_inplace(
            A: T.Buffer((T.int64(1), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(1), T.int64(3)):
                with T.sblock("T_subtract"):
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
                    out_ty=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                q: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (a, y),
                    inplace_indices=[0],
                    out_ty=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                r: R.Tensor((1, 3), dtype="float32") = R.subtract(y, y)
                s: R.Tensor((1, 3), dtype="float32") = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (r, r),
                    inplace_indices=[1],
                    out_ty=[
                        R.Tensor((1, 3), dtype="float32"),
                    ],
                )
                m: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (q, s),
                    inplace_indices=[0],
                    out_ty=[
                        R.Tensor((2, 3), dtype="float32"),
                    ],
                )
                R.output(m)
            return m

    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(EndToEndTest)
    tvm.ir.assert_structural_equal(new_mod, Expected)

    x = tvm.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y = tvm.runtime.tensor(np.random.rand(1, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm.target.Target("llvm")
    ex = tvm.compile(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic():
    @I.ir_module(s_tir=True)
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

    @I.ir_module(s_tir=True)
    class Expected:
        @T.prim_func(private=True, s_tir=True)
        def add_inplace(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tirx.noalias": True})
            a, b = T.int64(), T.int64()
            A = T.match_buffer(var_A, (a, b))
            B = T.match_buffer(var_B, (a, b))
            for ax0, ax1 in T.grid(a, b):
                with T.sblock("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    T.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

        @T.prim_func(private=True, s_tir=True)
        def subtract_inplace(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tirx.noalias": True})
            a, b = T.int64(), T.int64()
            A = T.match_buffer(var_A, (a, b))
            B = T.match_buffer(var_B, (a, b))
            for ax0, ax1 in T.grid(a, b):
                with T.sblock("T_subtract"):
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
                    out_ty=R.Tensor((a, b), dtype="float32"),
                    inplace_indices=[0],
                )
                s = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (a_1, a_1),
                    out_ty=R.Tensor((a, b), dtype="float32"),
                    inplace_indices=[1],
                )
                R.output(s)
            return s

    tvm.ir.assert_structural_equal(new_mod, Expected, map_free_vars=True)
    x = tvm.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y = tvm.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm.target.Target("llvm")
    ex = tvm.compile(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic_mismatch():
    # cannot statically prove the shapes to be equal so the module should be unchanged
    @I.ir_module(s_tir=True)
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


class TestViewOpSharedStorageAndNoInplace:
    storage_ptr_x_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    storage_ptr_x_2d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    storage_ptr_x_squeeze = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    storage_ptr_x_ensure_zero_offset = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    @I.ir_module
    class _SharedStorageExpandDimsModule:
        @R.function
        def main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 1), dtype="float32") = R.expand_dims(x, axis=[1])
                lv1: R.Tensor((4, 1), dtype="float32") = R.expand_dims(x, axis=[1])
                gv: R.Tensor((4, 1), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageSqueezeModule:
        @R.function
        def main(x: R.Tensor((1, 4, 1), dtype="float32")) -> R.Tensor((4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 1), dtype="float32") = R.squeeze(x, axis=[0])
                lv1: R.Tensor((4, 1), dtype="float32") = R.squeeze(x, axis=[0])
                gv: R.Tensor((4, 1), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageReshapeModule:
        @R.function
        def main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 1), dtype="float32") = R.reshape(x, (4, 1))
                lv1: R.Tensor((4, 1), dtype="float32") = R.reshape(x, (4, 1))
                gv: R.Tensor((4, 1), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStoragePermuteDimsModule:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 1), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                lv1: R.Tensor((4, 1), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                gv: R.Tensor((4, 1), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageViewModule:
        @R.function
        def main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.memory.view(
                    x, R.shape([1, 4]), R.tuple(), R.tuple()
                )
                lv1: R.Tensor((1, 4), dtype="float32") = R.memory.view(
                    x, R.shape([1, 4]), R.tuple(), R.tuple()
                )
                gv: R.Tensor((1, 4), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageBatchFlattenModule:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.nn.batch_flatten(x)
                lv1: R.Tensor((1, 4), dtype="float32") = R.nn.batch_flatten(x)
                gv: R.Tensor((1, 4), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageFlattenModule:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="float32") = R.flatten(x)
                lv1: R.Tensor((4,), dtype="float32") = R.flatten(x)
                gv: R.Tensor((4,), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _SharedStorageEnsureZeroOffsetModule:
        @R.function
        def main(x: R.Tensor((4, 1), dtype="float32")) -> R.Tensor((4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 1), dtype="float32") = R.memory.ensure_zero_offset(x)
                lv1: R.Tensor((4, 1), dtype="float32") = R.memory.ensure_zero_offset(x)
                gv: R.Tensor((4, 1), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class _IndependentReluModule:
        """Just a testcase to verify that non-view ops do not share storage."""

        @R.function
        def main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="float32") = R.nn.relu(x)
                lv1: R.Tensor((4,), dtype="float32") = R.nn.relu(x)
                gv: R.Tensor((4,), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    @classmethod
    def _capture_op_tensors(cls, mod, input_nps, op_substr):
        """Capture TVM tensors passed to VM calls whose name contains op_substr."""
        captures = []

        def instrument(func, name, before_run, ret_value, *args):
            del func, ret_value
            if not before_run:
                return VMInstrumentReturnKind.NO_OP
            if op_substr not in name.lower():
                return VMInstrumentReturnKind.NO_OP
            tensor_args = [arg for arg in args if isinstance(arg, tvm.runtime.Tensor)]
            if not tensor_args:
                return VMInstrumentReturnKind.NO_OP
            captures.append({"call_name": name, "tensors": tensor_args})
            return VMInstrumentReturnKind.NO_OP

        if isinstance(input_nps, np.ndarray):
            input_nps = [input_nps]

        ex = relax.build(mod, tvm.target.Target("llvm"))
        vm = relax.VirtualMachine(ex, tvm.cpu())
        vm.set_instrument(instrument)
        vm["main"](*(tvm.runtime.tensor(arr, tvm.cpu()) for arr in input_nps))
        return captures

    @pytest.mark.parametrize(
        "mod,input_nps,op_substr,expect_same_storage",
        [
            pytest.param(
                _SharedStorageExpandDimsModule,
                [storage_ptr_x_1d],
                "add",
                True,
                id="shared_storage_expand_dims",
            ),
            pytest.param(
                _SharedStorageSqueezeModule,
                [storage_ptr_x_squeeze],
                "add",
                True,
                id="shared_storage_squeeze",
            ),
            pytest.param(
                _SharedStorageReshapeModule,
                [storage_ptr_x_1d],
                "add",
                True,
                id="shared_storage_reshape",
            ),
            pytest.param(
                _SharedStoragePermuteDimsModule,
                [storage_ptr_x_2d],
                "add",
                True,
                id="shared_storage_permute_dims",
            ),
            pytest.param(
                _SharedStorageFlattenModule,
                [storage_ptr_x_2d],
                "add",
                True,
                id="shared_storage_flatten",
            ),
            pytest.param(
                _SharedStorageBatchFlattenModule,
                [storage_ptr_x_2d],
                "add",
                True,
                id="shared_storage_batch_flatten",
            ),
            pytest.param(
                _SharedStorageViewModule,
                [storage_ptr_x_1d],
                "add",
                True,
                id="shared_storage_memory_view",
            ),
            pytest.param(
                _SharedStorageEnsureZeroOffsetModule,
                [storage_ptr_x_ensure_zero_offset],
                "add",
                True,
                id="shared_storage_ensure_zero_offset",
            ),
            pytest.param(
                _IndependentReluModule,
                [storage_ptr_x_1d],
                "add",
                False,
                id="independent_storage_relu",
            ),
        ],
    )
    def test_tensor_storage_ptr_extraction(self, mod, input_nps, op_substr, expect_same_storage):
        """Validate runtime storage overlap/sharing via VM instrumentation."""
        storage_shared = tvm.get_global_func("runtime.TVMTensorIsStorageShared")
        captures = self._capture_op_tensors(mod, input_nps, op_substr)
        assert len(captures), f"VM instrumentation did not see a {op_substr} call."
        assert len(captures) == 1, f"VM instrumentation should see exactly one {op_substr} call."
        cap = captures[0]
        assert len(cap["tensors"]) == 3, (
            f"VM instrumentation should see three {op_substr} tensor operands."
        )
        tensor_a, tensor_b = cap["tensors"][0], cap["tensors"][1]
        call_name = cap["call_name"]
        if expect_same_storage:
            assert storage_shared(tensor_a, tensor_b), (
                f"{mod.__name__}: operands should share the same storage (call {call_name!r})"
            )
        else:
            assert not storage_shared(tensor_a, tensor_b), (
                f"{mod.__name__}: operands must not share storage (call {call_name!r})"
            )

    @staticmethod
    def _emit_duplicate_view(op, x):
        if op == "relax.expand_dims":
            a = relax.op.expand_dims(x, axis=1)
            b = relax.op.expand_dims(x, axis=1)
        elif op == "relax.squeeze":
            a = relax.op.squeeze(x, axis=[0])
            b = relax.op.squeeze(x, axis=[0])
        elif op == "relax.reshape":
            a = relax.op.reshape(x, (4, 1))
            b = relax.op.reshape(x, (4, 1))
        elif op == "relax.permute_dims":
            a = relax.op.permute_dims(x, axes=[1, 0])
            b = relax.op.permute_dims(x, axes=[1, 0])
        elif op == "relax.memory.view":
            a = relax.op.memory.view(x, (4, 1))
            b = relax.op.memory.view(x, (4, 1))
        elif op == "relax.memory.ensure_zero_offset":
            a = relax.op.memory.ensure_zero_offset(x)
            b = relax.op.memory.ensure_zero_offset(x)
        elif op == "relax.flatten":
            a = relax.op.flatten(x)
            b = relax.op.flatten(x)
        elif op == "relax.nn.batch_flatten":
            a = relax.op.nn.batch_flatten(x)
            b = relax.op.nn.batch_flatten(x)
        else:
            raise ValueError(op)
        return a, b

    @staticmethod
    def _concat_axis_for_view_op(op):
        if op == "relax.flatten":
            return 0
        return 1

    @classmethod
    def _build_module(cls, op):
        if op == "relax.expand_dims":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.squeeze":
            x_ty = relax.TensorType((1, 4, 1), "float32")
        elif op == "relax.reshape":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.permute_dims":
            x_ty = relax.TensorType((1, 4), "float32")
        elif op == "relax.memory.view":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.memory.ensure_zero_offset":
            x_ty = relax.TensorType((4, 1), "float32")
        elif op in ("relax.flatten", "relax.nn.batch_flatten"):
            x_ty = relax.TensorType((1, 4), "float32")
        else:
            raise ValueError(op)

        bb = relax.BlockBuilder()
        x = relax.Var("x", x_ty)
        concat_axis = cls._concat_axis_for_view_op(op)
        with bb.function("main", [x]):
            with bb.dataflow():
                a_expr, b_expr = cls._emit_duplicate_view(op, x)
                a = bb.emit(a_expr)
                b = bb.emit(b_expr)
                prod = bb.emit(relax.op.multiply(a, b))
                out = bb.emit(relax.op.concat([prod, b], axis=concat_axis))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        return bb.finalize()

    @classmethod
    def _input_for_view_op(cls, op):
        if op == "relax.squeeze":
            return cls.storage_ptr_x_squeeze
        if op == "relax.memory.ensure_zero_offset":
            return cls.storage_ptr_x_ensure_zero_offset
        if op in ("relax.permute_dims", "relax.flatten", "relax.nn.batch_flatten"):
            return cls.storage_ptr_x_2d
        return cls.storage_ptr_x_1d

    @staticmethod
    def _torch_duplicate_view(x, op):
        if op == "relax.expand_dims":
            return x.unsqueeze(1)
        if op == "relax.squeeze":
            return x.squeeze(0)
        if op == "relax.reshape":
            return x.reshape(4, 1)
        if op == "relax.permute_dims":
            return x.permute(1, 0)
        if op == "relax.memory.view":
            return x.reshape(4, 1)
        if op == "relax.memory.ensure_zero_offset":
            return x
        if op == "relax.flatten":
            return x.flatten()
        if op == "relax.nn.batch_flatten":
            # TVM: ndim==2 input keeps shape (1, 4).
            return x
        raise ValueError(op)

    @classmethod
    def _expected_for_view_op(cls, op):
        x = torch.from_numpy(np.asarray(cls._input_for_view_op(op), dtype=np.float32))
        a = cls._torch_duplicate_view(x, op)
        b = cls._torch_duplicate_view(x, op)
        prod = a * b
        concat_axis = cls._concat_axis_for_view_op(op)
        return torch.cat([prod, b], dim=concat_axis).numpy()

    @pytest.mark.parametrize(
        "view_op",
        (
            # Keep this list in sync with IsViewMemoryOp() in
            # src/relax/transform/dataflow_inplace.cc
            "relax.expand_dims",
            "relax.squeeze",
            "relax.reshape",
            "relax.permute_dims",
            "relax.flatten",
            "relax.nn.batch_flatten",
            "relax.memory.view",
            "relax.memory.ensure_zero_offset",
        ),
    )
    def test_no_inplace_when_view_ops_share_input(self, view_op):
        mod = self._build_module(view_op)
        func = mod["main"]
        block = func.body.blocks[0]
        params = list(func.params)

        alias_sets, _ = dataflow_alias_analysis(block, params)
        view_vars = [
            binding.var
            for binding in block.bindings
            if (
                isinstance(binding.value, relax.Call)
                and isinstance(binding.value.op, tvm.ir.Op)
                and binding.value.op.name == view_op
            )
        ]
        a_var, b_var = view_vars[:2]
        assert alias_sets[a_var] & alias_sets[b_var], (
            f"{view_op}: duplicate views should share alias sets, but got "
            f"{alias_sets[a_var]} and {alias_sets[b_var]}"
        )

        _, exact_match = dataflow_inplace_analysis(block, params, mod)
        assert exact_match == [], f"{view_op}: expected no in-place opportunities"

        x_np = self._input_for_view_op(view_op).copy()
        mod_inplace = DataflowUseInplaceCalls()(mod)
        tvm.ir.assert_structural_equal(mod_inplace, mod)

        storage_shared = tvm.get_global_func("runtime.TVMTensorIsStorageShared")
        captures = self._capture_op_tensors(mod_inplace, x_np, "multiply")
        assert captures, f"{view_op}: VM instrumentation did not see a multiply call."
        cap = next(c for c in captures if len(c["tensors"]) >= 2)
        tensor_a, tensor_b = cap["tensors"][0], cap["tensors"][1]
        assert storage_shared(tensor_a, tensor_b), (
            f"{view_op}: multiply operands should share the same storage at runtime "
            f"(call {cap['call_name']!r})"
        )

        ex = relax.build(mod_inplace, tvm.target.Target("llvm"))
        vm = relax.VirtualMachine(ex, tvm.cpu())
        out = vm["main"](tvm.runtime.tensor(x_np, tvm.cpu()))
        np.testing.assert_allclose(out.numpy(), self._expected_for_view_op(view_op))


if __name__ == "__main__":
    testing.main()
