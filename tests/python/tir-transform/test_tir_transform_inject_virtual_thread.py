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
from tvm.script import tir as T, ir as I


def test_vthread():
    """Test virtual thread injection with vthread"""
    n = 100
    m = 4
    nthread = 2

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle("float32"), C: T.handle("float32")):
            A_buf = T.decl_buffer((n * nthread,), "float32", data=A)
            C_buf = T.decl_buffer((n * nthread,), "float32", data=C)
            for i in range(n):
                vt_x = T.launch_thread("vthread", nthread)
                vt_y = T.launch_thread("vthread", nthread)
                B_data = T.allocate([m], "float32", scope="shared")
                B = T.Buffer([m], "float32", data=B_data, scope="shared")
                B[i] = A_buf[i * nthread + vt_x]
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "Run",
                        B.access_ptr("r"),
                        T.call_intrin("int32", "tir.tvm_context_id"),
                    )
                )
                C_buf[i * nthread + vt_x] = B[i] + T.float32(1)

    # For vthread, expected allocation is m * nthread
    B_expected_alloc = m * nthread

    stmt = tvm.tir.transform.InjectVirtualThread()(Module)["main"]

    # Find allocate nodes
    allocates = []

    def find_allocates(node):
        if isinstance(node, tvm.tir.Allocate):
            allocates.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt.body, find_allocates)
    assert len(allocates) == 1
    assert list(allocates[0].extents) == [B_expected_alloc]


def test_vthread_extern():
    """Test virtual thread injection with extern call"""
    n = 100
    m = 4
    nthread = 2

    @I.ir_module
    class Module:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main"})
            for i in range(n):
                vt_x = T.launch_thread("vthread", nthread)
                vt_y = T.launch_thread("vthread", nthread)
                A_data = T.allocate([m], "float32", scope="shared")
                A = T.Buffer([m], "float32", data=A_data, scope="shared")
                B_data = T.allocate([m], "float32", scope="shared")
                B = T.Buffer([m], "float32", data=B_data, scope="shared")
                C_data = T.allocate([m], "float32", scope="shared")
                C = T.Buffer([m], "float32", data=C_data, scope="shared")
                A[vt_x] = T.Cast("float32", vt_x) + T.float32(1)
                B[vt_y] = T.Cast("float32", vt_y) + T.float32(1)
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "Run",
                        A.access_ptr("r"),
                        B.access_ptr("r"),
                        C.access_ptr("rw"),
                    )
                )

    # For vthread:
    # A, B expected allocation is m * nthread (used with single vthread each)
    A_expected_alloc = m * nthread
    # C expected allocation is m * nthread * nthread (used in extern with both vthreads)
    C_expected_alloc = m * nthread * nthread

    stmt = tvm.tir.transform.InjectVirtualThread()(Module)["main"]

    # Find allocate nodes
    allocates = []

    def find_allocates(node):
        if isinstance(node, tvm.tir.Allocate):
            allocates.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt.body, find_allocates)
    assert len(allocates) == 3
    # Check that we have the expected extents (order may vary)
    extents = sorted([int(a.extents[0]) for a in allocates])
    assert extents == sorted([A_expected_alloc, A_expected_alloc, C_expected_alloc])


def test_vthread_if_then_else():
    """Test virtual thread injection with if-then-else"""
    nthread = 2

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle("float32")):
            T.func_attr({"global_symbol": "main"})
            A_buf = T.decl_buffer((100 * nthread,), "float32", data=A)
            for i in range(100):
                vt = T.launch_thread("vthread", nthread)
                B_data = T.allocate([128], "float32", scope="shared")
                B = T.Buffer([128], "float32", data=B_data, scope="shared")
                if i == 0:
                    B[i] = A_buf[i * nthread + vt]
                else:
                    B[i] = A_buf[i * nthread + vt] + T.float32(1)
                if i == 0:
                    B[i] = A_buf[i * nthread + vt] + T.float32(2)

    stmt = tvm.tir.transform.InjectVirtualThread()(Module)["main"]

    # Find IfThenElse nodes
    if_nodes = []

    def find_ifs(node):
        if isinstance(node, tvm.tir.IfThenElse):
            if_nodes.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt.body, find_ifs)

    assert len(if_nodes) == 2
    # First if has else_case, second does not
    assert if_nodes[0].else_case is not None
    assert if_nodes[1].else_case is None


def test_vthread_simplified():
    """Indices resulting from vthread injection should simplified

    This ensures that downstream passes that check for Ramp nodes do
    not need to each simplify the indices.
    """

    @T.prim_func
    def before_func():
        vthread = T.env_thread("vthread")
        T.launch_thread(vthread, 4)
        B_data = T.allocate([4], "int32", scope="shared")
        B = T.Buffer([4], "int32", data=B_data, scope="shared")
        B[0:4] = T.broadcast(vthread, 4)

    @T.prim_func
    def expected_func():
        B_data = T.allocate([16], "int32", scope="shared")
        B = T.Buffer([16], "int32", data=B_data, scope="shared")
        # The indices for B should each be a single Ramp node, and
        # should not be the sum of a Ramp and Broadcast node.
        B[T.Mul(0, 4) : T.Mul(0, 4) + 4] = T.broadcast(0, 4)
        B[T.Mul(1, 4) : T.Mul(1, 4) + 4] = T.broadcast(1, 4)
        B[T.Mul(2, 4) : T.Mul(2, 4) + 4] = T.broadcast(2, 4)
        B[T.Mul(3, 4) : T.Mul(3, 4) + 4] = T.broadcast(3, 4)

    before_mod = tvm.IRModule.from_expr(before_func.with_attr("global_symbol", "main"))
    after_mod = tvm.tir.transform.InjectVirtualThread()(before_mod)
    after_func = after_mod["main"]

    tvm.ir.assert_structural_equal(after_func, expected_func.with_attr("global_symbol", "main"))


def test_vthread_vectorized():
    """Use of vthread is compatible with vector allocations"""

    @T.prim_func
    def before_func():
        vthread = T.env_thread("vthread")
        T.launch_thread(vthread, 4)
        B_data = T.allocate([4], "int32", "shared")
        B = T.Buffer([4], "int32", data=B_data, scope="shared")
        B[0:4] = T.broadcast(vthread, 4)

    @T.prim_func
    def expected_func():
        B_data = T.allocate([4], "int32x4", "shared")
        B = T.Buffer([4], "int32x4", data=B_data, scope="shared")
        B[T.Div(T.Mul(0, 4), 4)] = T.broadcast(0, 4)
        B[T.Div(T.Mul(1, 4), 4)] = T.broadcast(1, 4)
        B[T.Div(T.Mul(2, 4), 4)] = T.broadcast(2, 4)
        B[T.Div(T.Mul(3, 4), 4)] = T.broadcast(3, 4)

    before_mod = tvm.IRModule.from_expr(before_func.with_attr("global_symbol", "main"))
    intermediate_mod = tvm.tir.transform.InjectVirtualThread()(before_mod)
    after_mod = tvm.tir.transform.StorageRewrite()(intermediate_mod)
    after_func = after_mod["main"]

    tvm.ir.assert_structural_equal(after_func, expected_func.with_attr("global_symbol", "main"))


if __name__ == "__main__":
    tvm.testing.main()
