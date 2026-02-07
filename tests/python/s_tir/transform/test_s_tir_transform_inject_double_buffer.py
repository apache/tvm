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


def test_double_buffer():
    n = 100
    m = 4

    @I.ir_module
    class Module:
        @T.prim_func
        def db(A: T.handle("float32"), C: T.handle("float32")):
            A_buf = T.decl_buffer((n * m,), "float32", data=A)
            C_buf = T.decl_buffer((m,), "float32", data=C)
            tx = T.launch_thread("threadIdx.x", 1)
            for i in range(n):
                B_data = T.allocate([m], "float32", scope="shared")
                B = T.Buffer([m], "float32", data=B_data, scope="shared")
                with T.attr(B_data, "double_buffer_scope", 1):
                    for j in range(m):
                        B[j] = A_buf[i * 4 + j]
                for j in range(m):
                    C_buf[j] = B[j] + T.float32(1.0)

    mod = Module

    opt = tvm.transform.Sequential(
        [tvm.s_tir.transform.InjectDoubleBuffer(), tvm.tir.transform.Simplify()]
    )

    with tvm.transform.PassContext(config={"s_tir.InjectDoubleBuffer": {"split_loop": 2}}):
        mod = opt(mod)
    stmt = mod["db"].body

    # After transformation, the buffer allocation should be doubled
    allocate_node = None

    def visitor(op):
        nonlocal allocate_node
        if isinstance(op, tvm.tir.Allocate) and "B" in str(op.buffer_var):
            allocate_node = op

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    assert allocate_node is not None
    assert list(allocate_node.extents) == [m * 2]

    f = tvm.tir.transform.ThreadSync("shared")(mod)["db"]
    count = [0]

    def count_sync(op):
        if isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.tvm_storage_sync")):
            count[0] += 1

    tvm.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


def test_double_buffer_transform():
    transform = tvm.ir.transform.Sequential(
        [
            tvm.s_tir.transform.InjectDoubleBuffer(),
            tvm.tir.transform.Simplify(),
        ]
    )

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer([16, 32], "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                cache_data = T.allocate([32], "float32")
                cache = T.Buffer(32, "float32", data=cache_data)

                T.attr(cache_data, "double_buffer_scope", 1)

                for j in range(32):
                    cache[j] = A[i, j]

                B[i] = 0.0
                for j in range(32):
                    B[i] = B[i] + cache[j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer((16,), "float32")):
            cache_data = T.allocate([64], "float32", "global")
            cache = T.Buffer(64, data=cache_data)
            for j in range(32):
                cache[j] = A[0, j]

            B[0] = T.float32(0)
            for j in range(32):
                B[0] = B[0] + cache[j]

            for i_outer in range(15):
                T.attr(cache_data, "double_buffer_write", 1)
                for j in range(32):
                    cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
                B[i_outer + 1] = T.float32(0)
                for j in range(32):
                    B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_double_buffer_with_decl_buffer():
    """Like test_double_buffer_transform, but with a declared buffer object"""

    transform = tvm.ir.transform.Sequential(
        [
            tvm.s_tir.transform.InjectDoubleBuffer(),
            tvm.tir.transform.Simplify(),
        ]
    )

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                cache = T.decl_buffer(32, "float32")
                T.attr(cache.data, "double_buffer_scope", 1)

                for j in range(32):
                    cache[j] = A[i, j]

                B[i] = 0.0
                for j in range(32):
                    B[i] = B[i] + cache[j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
            cache = T.decl_buffer(64, "float32")
            for j in range(32):
                cache[j] = A[0, j]

            B[0] = T.float32(0)
            for j in range(32):
                B[0] = B[0] + cache[j]

            for i_outer in range(15):
                T.attr(cache.data, "double_buffer_write", 1)
                for j in range(32):
                    cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
                B[i_outer + 1] = T.float32(0)
                for j in range(32):
                    B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
