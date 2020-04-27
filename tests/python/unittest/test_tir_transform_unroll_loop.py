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
from tvm import te
import os


def test_unroll_loop():
    ib = tvm.tir.ir_builder.create()
    dtype = 'int64'
    n = te.size_var('n')
    Ab = tvm.tir.decl_buffer((n, ), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(n, n + 2, name="i") as i:
        with ib.for_range(0, 8, name="i", for_type="unroll") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt))

    assert isinstance(stmt, tvm.tir.For)

    ret = tvm.tir.transform.UnrollLoop(16, 8, 0, True)(mod)["main"].body

    assert not isinstance(ret, tvm.tir.For)
    ret = tvm.tir.transform.UnrollLoop(15, 8, 0, True)(mod)["main"].body
    assert isinstance(ret, tvm.tir.For)
    ret = tvm.tir.transform.UnrollLoop(16, 8, 0, False)(mod)["main"].body
    assert isinstance(ret, tvm.tir.For)
    assert ret.for_type == tvm.tir.For.Unrolled

    ib = tvm.tir.ir_builder.create()
    ib.scope_attr(tvm.tir.const(0, "int32"), "pragma_auto_unroll_max_step", 16)
    ib.emit(stmt)
    wrapped = ib.get()
    wrapped = tvm.tir.SeqStmt([wrapped, stmt])
    assert isinstance(ret, tvm.tir.For)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], wrapped))
    ret = tvm.tir.transform.UnrollLoop(0, 8, 0, False)(mod)["main"].body

    assert isinstance(ret[0], tvm.tir.For)
    assert ret[0].for_type == tvm.tir.For.Unrolled
    assert isinstance(ret[1], tvm.tir.For)
    assert ret[1].for_type != tvm.tir.For.Unrolled

def test_unroll_fake_loop():
    ib = tvm.tir.ir_builder.create()
    dtype = 'int32'
    n = te.size_var('n')
    Ab = tvm.tir.decl_buffer((n, ), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(0, 1, name="i") as i:
        Aptr[i*2] = 3
        with ib.for_range(0, 10, name="j") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt))
    ret = tvm.tir.transform.UnrollLoop(8, 0, 1, False)(mod)["main"].body
    assert isinstance(ret[0], tvm.tir.Store)

def test_unroll_single_count_loops():
    n = te.size_var('n')
    A = te.placeholder((n,), name='A')
    B = te.compute((n,), lambda *i: A(*i), name='B')
    s = te.create_schedule(B.op)
    s = s.normalize()
    dom_map = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
    # all parameters to UnrolLoops are default values except for
    # auto_unroll_max_extent which has been set to 1 (default:0)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    ret = tvm.tir.transform.UnrollLoop(0, 8, 1, True)(mod)["main"].body

    assert ret == stmt

if __name__ == "__main__":
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_single_count_loops()
