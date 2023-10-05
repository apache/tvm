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
from tvm.relax.analysis import dataflow_liveness_analysis, dataflow_alias_analysis
from tvm.script.parser import ir as I, relax as R, tir as T


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
                R.output(n)
            return n

    block = BasicLiveness["main"].body.blocks[0]
    live_ranges = dataflow_liveness_analysis(block)
    expected_ranges = {
        "x": (-1, 1),
        "y": (0, 2),
        "z": (1, 3),
        "q": (2, 3),
        "p": (3, 4),
        "n": (4, 5),
    }
    for var, live_range in live_ranges.items():
        assert live_range == expected_ranges[var.name_hint]


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

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name_hint]
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

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name_hint]
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

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name_hint]
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
        "a": {3, 4, 5, 6, 7, 8},
        "b": {-1},  # because a can be many things, b is unknown
        "c": {-1},  # because a can be many things, c is unknown
    }

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name_hint]
    assert len(tuple_map) == 2
    assert 5 in tuple_map
    assert tuple_map[5] == [{3}, {4}]
    assert 6 in tuple_map
    assert tuple_map[6] == [{7}, {8}]


if __name__ == "__main__":
    tvm.testing.main()
