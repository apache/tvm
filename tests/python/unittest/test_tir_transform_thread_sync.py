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
import tvm.testing


def run_passes(inputs, stmt):
    func = tvm.te.schedule.SchedulePostProcToPrimFunc(inputs, stmt, None)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    cuda_target = tvm.target.Target("cuda")

    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target})
    )(mod)

    mod = tvm.tir.transform.SplitHostDevice()(mod)
    return tvm.tir.transform.ThreadSync("shared")(mod)


@tvm.testing.requires_cuda
def test_thread_storage_sync():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")

    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A2].bind(xo, te.thread_axis("blockIdx.x"))
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = run_passes([A, A2], stmt)
    f = mod["test_kernel0"]
    body_list = tvm.tir.stmt_list(f.body.body.body)
    assert body_list[1].value.op.same_as(tvm.ir.Op.get("tir.tvm_storage_sync"))


@tvm.testing.requires_cuda
def test_sync_else_branch():
    def ir(A, B):
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)

        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", 1)

        local = ib.allocate(A.dtype, 8, name="buf_local", scope="local")
        shared = ib.allocate(A.dtype, 8, name="buf_shared", scope="shared")

        with ib.for_range(0, 8) as i:
            with ib.if_scope(Aptr[i] < 0):
                local[i] = Aptr[i]
            with ib.else_scope():
                shared[i] = Aptr[i]

        with ib.for_range(0, 8) as i:
            with ib.if_scope(Aptr[i] < 0):
                Bptr[i] = local[i]
            with ib.else_scope():
                Bptr[i] = shared[i]

        return ib.get()

    A = tvm.tir.decl_buffer((8,), "float32")
    B = tvm.tir.decl_buffer((8,), "float32")
    stmt = ir(A, B)
    mod = run_passes([A, B], stmt)
    assert "@tir.tvm_storage_sync" in str(mod)


if __name__ == "__main__":
    test_thread_storage_sync()
    test_sync_else_branch()
