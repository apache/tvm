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

def test_thread_storage_sync():
    m = te.size_var('m')
    l = te.size_var('l')
    A = te.placeholder((m, l), name='A')

    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A2].bind(xo, te.thread_axis("blockIdx.x"))
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.tir.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.tir.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)

    cuda_target = tvm.target.create("cuda")

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([Ab, A2b], stmt).with_attr({
            "global_symbol": "test", "target": cuda_target}))

    fdevice = tvm.tir.transform.SplitHostDevice()(mod)["test_kernel0"]
    mod = tvm.IRModule.from_expr(fdevice)
    cuda_target = tvm.target.create("cuda")
    f = tvm.tir.transform.ThreadSync("shared")(mod)["test_kernel0"]
    body_list = tvm.tir.stmt_list(f.body.body.body.body)
    assert(body_list[1].value.name == "tvm_storage_sync")



if __name__ == "__main__":
    test_thread_storage_sync()
