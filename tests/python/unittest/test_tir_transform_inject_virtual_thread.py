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

def test_vthread():
    dtype = 'int64'
    n = 100
    m = 4
    nthread = 2
    def get_vthread(name):
        tx = te.thread_axis(name)
        ty = te.thread_axis(name)
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        C = ib.pointer("float32", name="C")
        with ib.for_range(0, n) as i:
            ib.scope_attr(tx, "virtual_thread", nthread)
            ib.scope_attr(ty, "virtual_thread", nthread)
            B = ib.allocate("float32", m, name="B", scope="shared")
            B[i] = A[i * nthread + tx]
            bbuffer = tvm.tir.decl_buffer((m,), dtype=B.dtype, data=B.asobject())
            ib.emit(tvm.tir.call_extern("int32", "Run",
                                    bbuffer.access_ptr("r"),
                                    tvm.tir.call_pure_intrin("int32", "tvm_context_id")))
            C[i * nthread + tx] = B[i] + 1
        return ib.get()

    stmt = tvm.tir.transform.InjectVirtualThread()(tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([], get_vthread("vthread"))))["main"].body

    assert stmt.body.body.extents[0].value == 2

    stmt = tvm.tir.transform.InjectVirtualThread()(tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([], get_vthread("cthread"))))["main"].body

    assert len(stmt.body.body.extents) == 3


def test_vthread_extern():
    dtype = 'int64'
    n = 100
    m = 4
    nthread = 2
    def get_vthread(name):
        tx = te.thread_axis(name)
        ty = te.thread_axis(name)
        ib = tvm.tir.ir_builder.create()
        with ib.for_range(0, n) as i:
            ib.scope_attr(tx, "virtual_thread", nthread)
            ib.scope_attr(ty, "virtual_thread", nthread)
            A = ib.allocate("float32", m, name="A", scope="shared")
            B = ib.allocate("float32", m, name="B", scope="shared")
            C = ib.allocate("float32", m, name="C", scope="shared")
            cbuffer = tvm.tir.decl_buffer((m,), dtype=C.dtype, data=C.asobject())
            abuffer = tvm.tir.decl_buffer((m,), dtype=A.dtype, data=A.asobject())
            bbuffer = tvm.tir.decl_buffer((m,), dtype=B.dtype, data=B.asobject())
            A[tx] = tx + 1.0
            B[ty] = ty + 1.0
            ib.emit(tvm.tir.call_extern("int32", "Run",
                                        abuffer.access_ptr("r"),
                                        bbuffer.access_ptr("r"),
                                        cbuffer.access_ptr("rw")))
        return ib.get()


    stmt = tvm.tir.transform.InjectVirtualThread()(tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([], get_vthread("cthread"))))["main"].body

    assert stmt.body.body.extents[0].value == 2
    assert stmt.body.body.body.body.body.body.extents[0].value == 2
    assert len(stmt.body.body.body.body.body.body.extents) == 3


def test_vthread_if_then_else():
    nthread = 2
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
    stmt = ib.get()

    stmt = tvm.tir.transform.InjectVirtualThread()(tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([], stmt)))["main"].body

    assert stmt.body.body.body[0].else_case != None
    assert stmt.body.body.body[1].else_case == None

if __name__ == "__main__":
    test_vthread_extern()
    test_vthread()
    test_vthread_if_then_else()
