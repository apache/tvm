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

def test_virtual_thread():
    m = tvm.var('m')
    A = tvm.placeholder((m, ), name='A')
    A1 = tvm.compute((m,), lambda i: A[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A1[i] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    vx = tvm.thread_axis("vthread", name="vx")
    xo, xi = s[A2].split(A2.op.axis[0], nparts=2)
    s[A2].bind(xo, vx)
    xo, xi = s[A2].split(xi, 8)
    s[A1].compute_at(s[A2], xo)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.InjectVirtualThread(stmt)
    print(stmt)

if __name__ == "__main__":
    test_virtual_thread()
