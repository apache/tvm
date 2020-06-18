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
import numpy

def test_makeapi():
    """Not yet working, mock design"""
    n = te.size_var('n')
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = te.create_schedule(C.op)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name='B')
    Cb = tvm.tir.decl_buffer(C.shape, C.dtype, name='C')
    stmt = tvm.tir.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, C:Cb}, 64)

    num_unpacked_args = 2
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([n, Ab, Bb, Cb], stmt).with_attr({
            "global_symbol": "main",
            "target": tvm.target.create("llvm")
        }))
    f = tvm.tir.transform.MakePackedAPI(num_unpacked_args)(mod)["main"]
    assert(len(f.params) == 7)


if __name__ == "__main__":
    test_makeapi()
