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
from tvm.driver.build_module import schedule_to_module
from tvm.script import tir as T
from tvm.relay import GlobalVar


def test_flatten2():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")
    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="A")
    A2b = tvm.tir.decl_buffer(A2.shape, A2.dtype, name="A2")

    mod = schedule_to_module(s, [Ab, A2b], binds={A: Ab, A2: A2b})
    mod = tvm.tir.transform.StorageFlatten(64)(mod)


def test_flatten_prefetch():
    A = te.placeholder((25, 100, 4), name="A")
    _A = tvm.tir.decl_buffer(A.shape, A.dtype, name="A")
    i = te.size_var("i")
    j = te.size_var("j")
    region = [tvm.ir.Range.from_min_extent(i[0], i[1]) for i in [(i, 2), (j, 8), (0, 4)]]
    stmt = tvm.tir.Prefetch(_A, region)

    func = tvm.te.schedule.SchedulePostProcToPrimFunc([_A], stmt, {A: _A})

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.transform.Sequential(
        [tvm.tir.transform.StorageFlatten(64), tvm.tir.transform.Simplify()]
    )(mod)
    stmt = mod["main"].body
    assert stmt.extent.value == 2
    assert isinstance(stmt.body, tvm.tir.For)
    assert stmt.body.extent.value == 2

    def assert_flat_loads(stmt):
        if isinstance(stmt, tvm.tir.BufferLoad):
            assert len(stmt.indices) == 1, "All prefetch indices should be flattened"

    tvm.tir.stmt_functor.post_order_visit(stmt, assert_flat_loads)


def test_flatten_storage_align():
    m = 8
    l = 16
    A = te.placeholder((m, l), name="A")
    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    s[A1].storage_align(A1.op.axis[0], 2, 1)

    mod = schedule_to_module(s, [A, A2])
    mod = tvm.transform.Sequential(
        [tvm.tir.transform.StorageFlatten(64), tvm.tir.transform.Simplify()]
    )(mod)

    stmt = mod["main"].body
    assert stmt.extents[0].value == 17 * 8


def test_flatten_double_buffer():
    @tvm.script.ir_module
    class ModFromScript:
        @T.prim_func
        def main(A_param: T.handle, C_param: T.handle):
            A = T.match_buffer(A_param, (400,), "float32", strides=[1])
            C = T.match_buffer(C_param, (4,), "float32", strides=[1])
            T.func_attr({"from_legacy_te_schedule": True})
            threadIdx_x = T.env_thread("threadIdx.x")
            T.launch_thread(threadIdx_x, 1)
            for i in T.serial(0, 100):
                B = T.decl_buffer([4], "float32", scope="shared")
                with T.attr(B.data, "double_buffer_scope", 1):
                    for j in T.serial(0, 4):
                        B[j] = A[4 * i + j]

                for j in T.serial(0, 4):
                    C[j] = B[j] + 1.0

    mod = ModFromScript

    with tvm.transform.PassContext(config={"tir.InjectDoubleBuffer": {"split_loop": 2}}):
        mod = tvm.transform.Sequential(
            [
                tvm.tir.transform.StorageFlatten(64),
                tvm.tir.transform.InjectDoubleBuffer(),
                tvm.tir.transform.Simplify(),
            ]
        )(mod)

    stmt = mod["main"].body
    assert isinstance(stmt.body, tvm.tir.Allocate)
    assert list(stmt.body.extents) == [8]

    mod = tvm.tir.transform.ThreadSync("shared")(mod)
    f = mod["main"]

    count = [0]

    def count_sync(op):
        if isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.tvm_storage_sync")):
            count[0] += 1

    tvm.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


def test_flatten_let_buffer():
    @tvm.script.ir_module
    class module:
        @T.prim_func
        def main():
            T.func_attr({"from_legacy_te_schedule": True})

            # If a pointer defined using a LetStmt,
            A_data: T.handle("int32") = T.call_extern("dummy_extern_function", dtype="handle")

            # and a buffer is backed by that pointer,
            A = T.decl_buffer([1], dtype="float32", data=A_data)
            T.evaluate(A[0])

    # then the call to StorageFlatten would result in an exception
    # being thrown.
    tvm.tir.transform.StorageFlatten(64)(module)


@T.prim_func
def tir_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [2, 2])
    B = T.match_buffer(a, [2, 2])
    A[0, 1] = B[1, 1]


def test_flatten_tir():
    orig_mod = tvm.IRModule({GlobalVar("main"): tir_func})
    mod = tvm.tir.transform.StorageFlatten(64)(orig_mod)
    tvm.ir.assert_structural_equal(
        orig_mod, mod
    )  # StorageFlatten should do nothing to TIR functions


class TestPreserveDeclBuffer(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.StorageFlatten(64)

    def before():
        T.func_attr({"from_legacy_te_schedule": True})
        A = T.decl_buffer([16, 16], "float32")
        for i, j in T.grid(16, 16):
            A[i, j] = 0.0

    def expected():
        T.func_attr({"from_legacy_te_schedule": True})
        A = T.decl_buffer([256], "float32")
        for i, j in T.grid(16, 16):
            A[i * 16 + j] = 0.0


if __name__ == "__main__":
    tvm.testing.main()
