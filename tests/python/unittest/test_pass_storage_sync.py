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

def test_storage_sync():
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
    f = tvm.tir.ir_pass.MakeAPI(stmt, "test", [Ab, A2b], 0, True)
    flist = tvm.tir.ir_pass.SplitHostDevice(f)
    f = flist[1]
    f = tvm.tir.ir_pass.ThreadSync(f, "shared")
    body_list = tvm.tir.stmt_list(f.body.body.body.body)
    assert(body_list[1].value.name == "tvm_storage_sync")


def test_coproc_sync():
    @tvm.register_func("tvm.info.mem.global.cache")
    def meminfo_cache():
        return tvm.ir.make_node(
            "MemoryInfo",
            unit_bits=8,
            max_simd_bits=32,
            max_num_bits=128,
            head_address=tvm.tir.call_extern("handle", "global_cache"))
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    cp = te.thread_axis((0, 1), "cop")
    A = ib.allocate("float32", 128, name="A", scope="global.cache")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 8, name="k") as k:
            with ib.for_range(0, 10, name="j") as j:
                ib.scope_attr(cp, "coproc_scope", 1)
                A[j] = A[j + k * 10] + 2
    stmt = ib.get()
    stmt = tvm.tir.ir_pass.CoProcSync(stmt)
    body = stmt.body.body.body
    blist = tvm.tir.stmt_list(body)
    assert(blist[1].value.name == "cop.coproc_read_barrier")
    assert(blist[1].value.args[3].value == 80)
    assert(blist[-2].value.name == "cop.coproc_sync")
    assert(blist[-1].value.name == "cop.coproc_write_barrier")
    assert(blist[-1].value.args[3].value == 10)


def test_coproc_sync2():
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    cp = te.thread_axis((0, 1), "cop")
    ty = te.thread_axis("cthread")
    A = ib.allocate("float32", 128, name="A")
    ib.scope_attr(ty, "virtual_thread", 2)
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        A[ty] = 0.0
    with ib.for_range(0, n, name="i") as i:
        with ib.new_scope():
            ib.scope_attr(cp, "coproc_scope", 1)
            A[ty] = 1.0
        with ib.new_scope():
            ib.scope_attr(cp, "coproc_scope", 2)
            A[ty] = 1.0
    stmt = ib.get()
    stmt = tvm.tir.ir_pass.CoProcSync(stmt)

def test_coproc_sync3():
    def __check_list(tvm_array, py_list):
        for ti, li in zip(tvm_array, py_list):
            if ti.value != li:
                return False
        return True

    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    cp = te.thread_axis((0, 1), "cop")
    A = ib.allocate("float32", 128, name="A", scope="global.cache")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, n, name="i") as j:
            with ib.new_scope():
                ib.scope_attr(cp, "coproc_scope", 1)
                A[i] = 1.0
            with ib.new_scope():
                ib.scope_attr(cp, "coproc_scope", 2)
                A[i] = 1.0
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 3)
        A[0] = 0.0

    stmt = ib.get()
    stmt = tvm.tir.ir_pass.CoProcSync(stmt)
    slist = tvm.tir.stmt_list(stmt[0].body.body)
    push_st = slist[2]
    slist = tvm.tir.stmt_list(slist[-1])
    pop_st = slist[0].body[0]

    assert(push_st.value.name == "cop.coproc_dep_push")
    assert(__check_list(push_st.value.args, [2,3]))
    assert(pop_st.value.name == "cop.coproc_dep_pop")
    assert(__check_list(pop_st.value.args, [2,3]))


if __name__ == "__main__":
    test_coproc_sync()
    test_storage_sync()
    test_coproc_sync2()
    test_coproc_sync3()
