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
from tvm import te
from tvm.topi.math import cast
import numpy as np


def test_bool_load():
    def do_copy(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")

        max_threads = 32
        ib.scope_attr(bx, "thread_extent", tvm.tir.indexdiv(n + max_threads - 1, max_threads))
        ib.scope_attr(tx, "thread_extent", max_threads)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < n):
            B[tid] = cast(A[tid], "int32")

        return ib.get()

    n = 1024
    A = te.placeholder((n,), name="A", dtype="bool")
    B = te.placeholder((n,), name="B", dtype="int32")

    target = "vulkan"

    if not tvm.testing.device_enabled(target):
        return

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_copy(ins[0], outs[0], n),
        name="bool_copy_ir",
        dtype="int32",
    )
    s = te.create_schedule(B.op)

    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B], target)

    ctx = tvm.context(target, 0)
    a_np = np.random.uniform(size=n) > 0.5
    b_np = np.zeros((n,), dtype="int32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    func(a, b)
    ref = a_np.astype(np.int32)
    tvm.testing.assert_allclose(b.asnumpy(), ref)


if __name__ == "__main__":
    test_bool_load()
