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
import os


def test_unroll_loop():
    ib = tvm.ir_builder.create()
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(n, n + 2, name="i") as i:
        with ib.for_range(0, 8, name="i", for_type="unroll") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    assert isinstance(stmt, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, True)
    assert not isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 15, 8, 0, True)
    assert isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, False)
    assert isinstance(ret, tvm.stmt.For)
    assert ret.for_type == tvm.stmt.For.Unrolled

    ib = tvm.ir_builder.create()
    ib.scope_attr(tvm.const(0, "int32"), "pragma_auto_unroll_max_step", 16)
    ib.emit(stmt)
    wrapped = ib.get()
    wrapped = tvm.make.Block(wrapped, stmt)
    assert isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(wrapped, 0, 8, 0, False)
    assert isinstance(ret.first, tvm.stmt.For)
    assert ret.first.for_type == tvm.stmt.For.Unrolled
    assert isinstance(ret.rest, tvm.stmt.For)
    assert ret.rest.for_type != tvm.stmt.For.Unrolled

def test_unroll_fake_loop():
    ib = tvm.ir_builder.create()
    dtype = 'int32'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(0, 1, name="i") as i:
        Aptr[i*2] = 3
        with ib.for_range(0, 10, name="j") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    ret = tvm.ir_pass.UnrollLoop(stmt, 8, 0, 1, True)
    assert isinstance(ret.first, tvm.stmt.Store)

def test_unroll_single_count_loops():
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute((n,), lambda *i: A(*i), name='B')
    s = tvm.create_schedule(B.op)
    s = s.normalize()
    dom_map = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, dom_map)
    # all parameters to UnrolLoops are default values except for
    # auto_unroll_max_extent which has been set to 1 (default:0)
    after_unroll_stmt = tvm.ir_pass.UnrollLoop(stmt, 0, 8, 1, True)
    assert after_unroll_stmt == stmt

if __name__ == "__main__":
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_single_count_loops()
