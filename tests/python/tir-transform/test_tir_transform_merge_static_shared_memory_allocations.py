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
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.driver.build_module import schedule_to_module
from tvm.topi.math import cast
from tvm.script import tir as T


def run_passes(sch, args):
    mod = schedule_to_module(sch, args)
    with tvm.transform.PassContext(config={"tir.merge_static_smem": True}):
        return tvm.transform.Sequential(
            [
                tvm.tir.transform.StorageFlatten(64),
                tvm.tir.transform.Simplify(),
                tvm.tir.transform.VectorizeLoop(),
                tvm.tir.transform.StorageRewrite(),
                tvm.tir.transform.MergeSharedMemoryAllocations(),
            ]
        )(mod)


def verify_single_allocation(stmt, alloc_size=None):
    num_alloc = [0]
    alloc_extents = []

    def verify(n):
        if (
            isinstance(n, tvm.tir.Allocate)
            and n.buffer_var.type_annotation.storage_scope == "shared"
        ):
            num_alloc[0] += 1
            alloc_extents.append(n.extents[0])

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1

    if alloc_size:
        assert alloc_extents[0] == alloc_size


@tvm.testing.requires_gpu
def test_matmul_shared():
    n = 1024
    block = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    B = te.placeholder((n, n), name="B", dtype="float16")

    def syncthread():
        return tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"]))

    def test_matmul_ir(A, B, C):
        ib = tvm.tir.ir_builder.create()

        tx = te.thread_axis("threadIdx.x")
        ty = te.thread_axis("threadIdx.y")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", block)
        ib.scope_attr(ty, "thread_extent", block)
        ib.scope_attr(bx, "thread_extent", n // block)
        ib.scope_attr(by, "thread_extent", n // block)

        A_sh = ib.allocate(A.dtype, (block, block), scope="shared", name="A_sh")  # fp16
        B_sh = ib.allocate(B.dtype, (block, block), scope="shared", name="B_sh")  # fp16
        # Create a shared memory for the accumulation.
        # This is for testing merging shared memory alloctions with different data type.
        # In practice, there is no need to allocate a shared memory for C.
        C_local = ib.allocate(C.dtype, (1,), scope="local", name="C_local")
        C_sh = ib.allocate(C.dtype, (block, block), scope="shared", name="C_sh")  # fp32

        A_ptr = ib.buffer_ptr(A)
        B_ptr = ib.buffer_ptr(B)
        C_ptr = ib.buffer_ptr(C)

        C_local[0] = 0.0

        with ib.for_range(0, n // block, name="i") as i:
            A_sh[ty, tx] = A_ptr[by * block + ty, i * block + tx]
            B_sh[ty, tx] = B_ptr[i * block + ty, bx * block + tx]
            ib.emit(syncthread())

            with ib.for_range(0, block, name="k") as k:
                C_local[0] += cast(A_sh[ty, k] * B_sh[k, tx], "float32")
            ib.emit(syncthread())

        C_sh[ty, tx] = C_local[0]
        C_ptr[by * block + ty, bx * block + tx] = C_sh[ty, tx]

        return ib.get()

    C = te.extern(
        A.shape,
        [A, B],
        lambda ins, outs: test_matmul_ir(ins[0], ins[1], outs[0]),
        name="matmul",
        dtype="float32",
    )
    s = te.create_schedule(C.op)
    mod = run_passes(s, [A, B, C])
    # C can be allocated at the start of A, so we only need to allocate 2 block * block memory with dtype = float16
    expected_alloc_size = block * block * 4
    verify_single_allocation(mod["main"].body, expected_alloc_size)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return

        fmatmul = tvm.build(s, [A, B, C], target)
        dev = tvm.device(target, 0)

        size = (n, n)
        a_np = np.random.uniform(size=size).astype(A.dtype)
        b_np = np.random.uniform(size=size).astype(B.dtype)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(size, dtype=C.dtype), dev)
        fmatmul(a, b, c)
        np_ref = np.dot(a_np.astype("float32"), b_np.astype("float32"))
        tvm.testing.assert_allclose(c.numpy(), np_ref, 1e-4, 1e-4)

    for target in ["cuda"]:
        check_target(target)


@tvm.testing.requires_gpu
def test_shared_more_dtype():
    """Test vectorized store into shared memory"""
    n = 512
    A = te.placeholder((n,), name="A", dtype="int8")
    B = te.placeholder((n,), name="B", dtype="int16")

    def test_device_ir(A, B, C):
        n = A.shape[0]
        ib = tvm.tir.ir_builder.create()

        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", n)

        A_sh = ib.allocate(A.dtype, (n,), scope="shared")  # i8
        B_sh = ib.allocate(B.dtype, (n,), scope="shared")  # i16
        C_sh = ib.allocate(C.dtype, (n,), scope="shared")  # i32

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)

        A_sh[tx] = Aptr[tx]
        B_sh[tx] = Bptr[tx]

        C_sh[tx] = cast(A_sh[tx], "int32") + cast(B_sh[tx], "int32")
        Cptr[tx] = C_sh[tx]
        return ib.get()

    C = te.extern(
        (n,),
        [A, B],
        lambda ins, outs: test_device_ir(ins[0], ins[1], outs[0]),
        name="vadd",
        dtype="int32",
    )
    s = te.create_schedule(C.op)

    mod = run_passes(s, [A, B, C])
    verify_single_allocation(mod["main"].body, n * 4)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return

        fadd = tvm.build(s, [A, B, C], target)
        dev = tvm.device(target, 0)

        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((n,), dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy().astype("float32") + b.numpy(), 1e-4, 1e-4)

    for target in ["cuda"]:
        check_target(target)


if __name__ == "__main__":
    tvm.testing.main()
