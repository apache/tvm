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
from tvm.script import tir as T
from tvm import tir
from tvm.tir.schedule.testing import verify_trace_roundtrip


@T.prim_func
def innermost(A: T.Buffer[10, "float32"]):
    for i in range(10):
        with T.block("main"):
            A[i] = 0.0


@T.prim_func
def innermost_after(A: T.Buffer[10, "float32"]):
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("main_wrapper"):
            T.reads()
            T.writes()
            for i in T.serial(8):
                with T.block("main"):
                    T.reads()
                    T.writes(A[i])
                    A[i] = T.float32(0)
            for i_ in T.serial(2):
                i: T.int32 = i_ + 8
                with T.block("main_i_tail"):
                    T.reads()
                    T.writes(A[i])
                    A[i] = T.float32(0)


def test_partition_innermost():
    s = tir.Schedule(innermost, debug_mask="all")
    main = s.get_block("main")
    (i,) = s.get_loops(main)
    s.partition(i, tir.IntImm("int32", 8))
    print(s.mod["main"].script())
    print(innermost_after.script())
    tvm.ir.assert_structural_equal(innermost_after, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=innermost)


@T.prim_func
def outermost(A: T.Buffer[(12, 10), "float32"]):
    for i in range(12):
        for j in range(10):
            with T.block("main"):
                A[i, j] = 0.0


@T.prim_func
def outermost_after(A: T.Buffer[(12, 10), "float32"]):
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("main_wrapper"):
            T.reads()
            T.writes()
            for i, j in T.grid(8, 10):
                with T.block("main"):
                    T.reads()
                    T.writes(A[i, j])
                    A[i, j] = T.float32(0)
            for i_ in T.serial(4):
                i: T.int32 = i_ + 8
                for j in T.serial(10):
                    with T.block("main_i_tail"):
                        T.reads()
                        T.writes(A[i, j])
                        A[i, j] = T.float32(0)


def test_partition_outermost():
    s = tir.Schedule(outermost, debug_mask="all")
    main = s.get_block("main")
    i, j = s.get_loops(main)
    s.partition(i, tir.IntImm("int32", 8))
    print(s.mod["main"].script())
    print(outermost_after.script())
    tvm.ir.assert_structural_equal(outermost_after, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=outermost)
