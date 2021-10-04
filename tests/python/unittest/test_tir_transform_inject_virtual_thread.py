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
import sys

import pytest

import tvm
import tvm.testing
from tvm import te

vthread_name = tvm.testing.parameter(
    "vthread",
    "cthread",
)
buffer_size = tvm.testing.parameter(4)
nthread = tvm.testing.parameter(2)


@tvm.testing.fixture
def vthread_mod(vthread_name, buffer_size, nthread):
    loop_extent = 100

    tx = te.thread_axis(vthread_name)
    ty = te.thread_axis(vthread_name)
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    with ib.for_range(0, loop_extent) as i:
        ib.scope_attr(tx, "virtual_thread", nthread)
        ib.scope_attr(ty, "virtual_thread", nthread)
        B = ib.allocate("float32", buffer_size, name="B", scope="shared")
        B[i] = A[i * nthread + tx]
        bbuffer = tvm.tir.decl_buffer((buffer_size,), dtype=B.dtype, data=B.asobject())
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "Run",
                bbuffer.access_ptr("r"),
                tvm.tir.call_intrin("int32", "tir.tvm_context_id"),
            )
        )
        C[i * nthread + tx] = B[i] + 1

    return tvm.IRModule.from_expr(tvm.tir.PrimFunc([], ib.get()))


def test_vthread(vthread_mod, vthread_name, buffer_size, nthread):
    mod = tvm.tir.transform.InjectVirtualThread()(vthread_mod)
    stmt = mod["main"]

    if vthread_name == "vthread":
        # All virtual thread axes that starts with "vthread" share the
        # same iteration, similar to threadIdx.x, so the number of
        # virtual threads is nthread.
        expected_buffer_size = buffer_size * nthread
    elif vthread_name == "cthread":
        # All other virtual thread axes are independent, so tx and ty
        # are independent and the total number of virtual threads is
        # nthread*nthread.
        expected_buffer_size = buffer_size * nthread * nthread
    else:
        raise ValueError(f"Unexpected vthread_name: {vthread_name}")

    assert stmt.body.body.extent.value == expected_buffer_size


@tvm.testing.fixture
def vthread_extern_mod(vthread_name, buffer_size, nthread):
    loop_extent = 100

    tx = te.thread_axis(vthread_name)
    ty = te.thread_axis(vthread_name)
    ib = tvm.tir.ir_builder.create()
    with ib.for_range(0, loop_extent) as i:
        ib.scope_attr(tx, "virtual_thread", nthread)
        ib.scope_attr(ty, "virtual_thread", nthread)
        A = ib.allocate("float32", buffer_size, name="A", scope="shared")
        B = ib.allocate("float32", buffer_size, name="B", scope="shared")
        C = ib.allocate("float32", buffer_size, name="C", scope="shared")
        cbuffer = tvm.tir.decl_buffer((buffer_size,), dtype=C.dtype, data=C.asobject())
        abuffer = tvm.tir.decl_buffer((buffer_size,), dtype=A.dtype, data=A.asobject())
        bbuffer = tvm.tir.decl_buffer((buffer_size,), dtype=B.dtype, data=B.asobject())
        A[tx] = tx + 1.0
        B[ty] = ty + 1.0
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "Run",
                abuffer.access_ptr("r"),
                bbuffer.access_ptr("r"),
                cbuffer.access_ptr("rw"),
            )
        )
    return tvm.IRModule.from_expr(tvm.tir.PrimFunc([], ib.get()))


def test_vthread_extern(vthread_extern_mod, vthread_name, buffer_size, nthread):
    mod = tvm.tir.transform.InjectVirtualThread()(vthread_extern_mod)
    stmt = mod["main"]

    if vthread_name == "vthread":
        # The shared A and B buffers are only exposed as read-only to
        # the external function, so they can still share the allocated
        # space.
        ro_buffer_size = buffer_size * nthread
        rw_buffer_size = buffer_size * nthread * nthread
    elif vthread_name == "cthread":
        ro_buffer_size = buffer_size * nthread * nthread
        rw_buffer_size = buffer_size * nthread * nthread
    else:
        raise ValueError(f"Unexpected vthread_name: {vthread_name}")

    A_alloc = stmt.body.body
    C_alloc = A_alloc.body.body
    assert A_alloc.extent.value == ro_buffer_size
    assert C_alloc.extent.value == rw_buffer_size


@tvm.testing.fixture
def vthread_if_then_else_mod(nthread):
    tx = te.thread_axis("vthread")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 100) as i:
        ib.scope_attr(tx, "virtual_thread", nthread)
        B = ib.allocate("float32", 128, name="B", scope="shared")
        with ib.if_scope(i == 0):
            B[i] = A[i * nthread + tx]
        with ib.else_scope():
            B[i] = A[i * nthread + tx] + 1
        with ib.if_scope(i == 0):
            B[i] = A[i * nthread + tx] + 2
    return tvm.IRModule.from_expr(tvm.tir.PrimFunc([], ib.get()))


def test_vthread_if_then_else(vthread_if_then_else_mod):
    stmt = tvm.tir.transform.InjectVirtualThread()(vthread_if_then_else_mod)["main"]

    assert stmt.body.body.body[0].else_case != None
    assert stmt.body.body.body[1].else_case == None


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
