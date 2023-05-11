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
from tvm.script import tir as T


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.transform.Sequential(
        [
            tvm.tir.transform.FlattenBuffer(),
            tvm.tir.transform.Simplify(),
        ]
    )


class TestElementwise(BaseCompare):
    """2-d buffers are flattened to 1-d"""

    def before(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        for i in T.serial(0, 16):
            B_new = T.decl_buffer([1, 16], "float32")
            for j in T.serial(0, 16):
                B_new[0, j] = A[i, j] + 1.0
            for j in T.serial(0, 16):
                C[i, j] = B_new[0, j] * 2.0

    def expected(input_A: T.Buffer((16, 16), "float32"), input_C: T.Buffer((16, 16), "float32")):
        A = T.Buffer(256, dtype="float32", data=input_A.data)
        C = T.Buffer(256, dtype="float32", data=input_C.data)
        for i in T.serial(0, 16):
            B_new_data = T.allocate([16], "float32", scope="global")
            B_new = T.Buffer([16], "float32", scope="global", data=B_new_data)
            for j in T.serial(0, 16):
                B_new[j] = A[((i * 16) + j)] + 1.0
            for j in T.serial(0, 16):
                C[((i * 16) + j)] = B_new[j] * 2.0


class TestElementwiseWithoutDeclBuffer(BaseCompare):
    """2-d buffers are flattened to 1-d

    Like TestElementwise, but the TIR doesn't have the DeclBuffer
    node.  The T.Buffer declaration applies only during the
    parsing the TVMScript, and doesn't occur in the TIR itself.  In
    this case, the allocation should be assumed to be targeting flat
    memory, and should be flattened to a 1-d allocation.
    """

    def before(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        for i in T.serial(0, 16):
            B_new_data = T.allocate([1, 16], "float32", "global")
            B_new = T.Buffer([1, 16], "float32", data=B_new_data)
            for j in T.serial(0, 16):
                B_new[0, j] = A[i, j] + 1.0
            for j in T.serial(0, 16):
                C[i, j] = B_new[0, j] * 2.0

    def expected(input_A: T.Buffer((16, 16), "float32"), input_C: T.Buffer((16, 16), "float32")):
        A = T.Buffer(256, dtype="float32", data=input_A.data)
        C = T.Buffer(256, dtype="float32", data=input_C.data)
        for i in T.serial(0, 16):
            B_new_data = T.allocate([16], "float32", "global")
            B_new = T.Buffer(16, "float32", data=B_new_data)
            for j in T.serial(0, 16):
                B_new[j] = A[((i * 16) + j)] + 1.0
            for j in T.serial(0, 16):
                C[((i * 16) + j)] = B_new[j] * 2.0


class TestGPU(BaseCompare):
    """Buffer flattening may have indices based on GPU thread vars"""

    def before(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        i0 = T.env_thread("blockIdx.x")
        i1 = T.env_thread("threadIdx.x")
        i2 = T.env_thread("vthread")

        T.launch_thread(i0, 4)
        T.launch_thread(i1, 2)
        T.launch_thread(i2, 2)
        B = T.decl_buffer([1, 16], "float32", scope="local")
        for j in range(0, 16):
            B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
        for j in range(0, 16):
            C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0

    def expected(input_A: T.Buffer((16, 16), "float32"), input_C: T.Buffer((16, 16), "float32")):
        A = T.Buffer(256, dtype="float32", data=input_A.data)
        C = T.Buffer(256, dtype="float32", data=input_C.data)

        i0 = T.env_thread("blockIdx.x")
        i1 = T.env_thread("threadIdx.x")
        i2 = T.env_thread("vthread")

        T.launch_thread(i0, 4)
        T.launch_thread(i1, 2)
        T.launch_thread(i2, 2)
        B_data = T.allocate([16], "float32", scope="local")
        B = T.Buffer([16], "float32", scope="local", data=B_data)
        for j in range(0, 16):
            B[j] = A[i0 * 64 + i1 * 32 + i2 * 16 + j] + 1.0
        for j in range(0, 16):
            C[i0 * 64 + i1 * 32 + i2 * 16 + j] = B[j] * 2.0


class TestSymbolic(BaseCompare):
    """Dynamically-sized arrrays are flattened"""

    def before(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
        A = T.match_buffer(a, (n, m), "float32")
        C = T.match_buffer(c, (n, m), "float32")

        for i in range(0, n):
            B = T.decl_buffer([m], "float32")
            for j in range(0, m):
                B[j] = A[i, j] + 1.0
            for j in range(0, m):
                C[i, j] = B[j] * 2.0

    def expected(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
        input_A = T.match_buffer(a, (n, m), "float32")
        input_C = T.match_buffer(c, (n, m), "float32")
        A = T.Buffer(n * m, "float32", data=input_A.data)
        C = T.Buffer(n * m, "float32", data=input_C.data)

        for i in range(0, n):
            B_data = T.allocate([m], "float32", scope="global")
            B = T.Buffer([m], "float32", scope="global", data=B_data)
            for j in range(0, m):
                B[j] = A[i * m + j] + 1.0
            for j in range(0, m):
                C[i * m + j] = B[j] * 2.0


class TestFusedSymbolic(BaseCompare):
    """Dynamically-sized arrrays with fused iterator which can be flattened"""

    def before(a: T.handle, b: T.handle, n: T.int32) -> None:
        A = T.match_buffer(a, (32, n, n), "float32")
        B = T.match_buffer(b, (32, n, n), "float32")

        for i in range(0, n * n * 32):
            B[i // (n * n), (i % (n * n)) // n, i % n] = A[i // (n * n), (i % (n * n)) // n, i % n]

    def expected(a: T.handle, b: T.handle, n: T.int32) -> None:
        input_A = T.match_buffer(a, (32, n, n), "float32")
        input_B = T.match_buffer(b, (32, n, n), "float32")
        A = T.Buffer(n * n * 32, "float32", data=input_A.data)
        B = T.Buffer(n * n * 32, "float32", data=input_B.data)

        for i in range(0, n * n * 32):
            B[i] = A[i]


class TestFusedSymbolicWithPredicate(BaseCompare):
    """Dynamically-sized arrrays with fused iterator which can be flattened with extra predicate"""

    def before(a: T.handle, b: T.handle, n: T.int32) -> None:
        A = T.match_buffer(a, (32, n, n), "float32")
        B = T.match_buffer(b, (32, n, n), "float32")
        for bx, tx in T.grid((n * n + 1) // 2, 64):
            if bx * 64 + tx < n * n * 32:
                B[
                    (bx * 64 + tx) // (n * n), ((bx * 64 + tx) % (n * n)) // n, (bx * 64 + tx) % n
                ] = A[
                    (bx * 64 + tx) // (n * n), ((bx * 64 + tx) % (n * n)) // n, (bx * 64 + tx) % n
                ]

    def expected(a: T.handle, b: T.handle, n: T.int32) -> None:
        input_A = T.match_buffer(a, (32, n, n), "float32")
        input_B = T.match_buffer(b, (32, n, n), "float32")
        A = T.Buffer(n * n * 32, "float32", data=input_A.data)
        B = T.Buffer(n * n * 32, "float32", data=input_B.data)

        for bx, tx in T.grid((n * n + 1) // 2, 64):
            if bx * 64 + tx < n * n * 32:
                B[bx * 64 + tx] = A[bx * 64 + tx]


class TestMultiAlloc(BaseCompare):
    """If multiple allocations occur, all are flattened."""

    def before(A: T.Buffer((4, 32), "float32"), D: T.Buffer((4, 32), "float32")):
        for i, j in T.grid(4, 32):
            B = T.decl_buffer((4, 32), "float32", scope="global")
            C = T.decl_buffer((4, 32), "float32", scope="global")
            B[i, j] = A[i, j] + 1.0
            C[i, j] = A[i, j] + B[i, j]
            D[i, j] = C[i, j] * 2.0

    def expected(input_A: T.Buffer((4, 32), "float32"), input_D: T.Buffer((4, 32), "float32")):
        A = T.Buffer(128, "float32", data=input_A.data)
        D = T.Buffer(128, "float32", data=input_D.data)

        for i, j in T.grid(4, 32):
            B_data = T.allocate([128], "float32", scope="global")
            B = T.Buffer([128], "float32", scope="global", data=B_data)
            C_data = T.allocate([128], "float32", scope="global")
            C = T.Buffer([128], "float32", scope="global", data=C_data)
            B[i * 32 + j] = A[i * 32 + j] + 1.0
            C[i * 32 + j] = A[i * 32 + j] + B[i * 32 + j]
            D[i * 32 + j] = C[i * 32 + j] * 2.0


class TestStrided(BaseCompare):
    """Indices for flattened buffers use the specified striding."""

    def before(A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")):
        for i0 in T.serial(4):
            B = T.decl_buffer([4, 17], "float32")
            B_1 = T.Buffer([4, 16], dtype="float32", data=B.data, strides=[17, 1])
            for i1, j in T.grid(4, 16):
                B_1[i1, j] = A[i0 * 4 + i1, j] + 1.0
            for i1, j in T.grid(4, 16):
                C[i0 * 4 + i1, j] = B_1[i1, j] * 2.0

    def expected(input_A: T.Buffer((16, 16), "float32"), input_C: T.Buffer((16, 16), "float32")):
        A = T.Buffer(256, dtype="float32", data=input_A.data)
        C = T.Buffer(256, dtype="float32", data=input_C.data)
        for i0 in T.serial(0, 4):
            B_new_data = T.allocate([68], "float32", scope="global")
            B_new = T.Buffer([68], "float32", scope="global", data=B_new_data)
            for i1 in T.serial(0, 4):
                for j in T.serial(0, 16):
                    B_new[i1 * 17 + j] = A[i0 * 64 + i1 * 16 + j] + 1.0
            for i1 in T.serial(0, 4):
                for j in T.serial(0, 16):
                    C[i0 * 64 + i1 * 16 + j] = B_new[i1 * 17 + j] * 2.0


class TestBoolean(BaseCompare):
    """Boolean buffers should be replaced by a backing int8 array"""

    def before(A: T.Buffer(10, "bool"), B: T.Buffer(10, "bool")) -> None:
        for i0 in T.serial(10):
            B[i0] = A[i0]

    def expected(input_A: T.Buffer(10, "bool"), input_B: T.Buffer(10, "bool")) -> None:
        A = T.Buffer(10, dtype="int8", data=input_A.data)
        B = T.Buffer(10, dtype="int8", data=input_B.data)
        # body
        for i0 in T.serial(10):
            B[i0] = T.cast(T.cast(A[i0], "bool"), "int8")


class TestLowerTE(BaseCompare):
    """FlattenBuffer should do nothing on TE-based functions"""

    def before(self):
        x = te.placeholder((1,))
        y = te.compute((1,), lambda i: x[i] + 2)
        s = te.create_schedule(y.op)
        mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
        return mod["main"]

    expected = before


class TestFlattenInsideBlock(BaseCompare):
    """Flattening access inside a block flattens the accessed region."""

    def before():
        A = T.alloc_buffer([32, 32])
        for i, j in T.grid(32, 32):
            with T.block("block"):
                T.reads(A[i, j])
                T.evaluate(A[i, j])

    def expected():
        A = T.alloc_buffer([1024])
        for i, j in T.grid(32, 32):
            with T.block("block"):
                T.reads(A[i * 32 + j])
                T.evaluate(A[i * 32 + j])


class TestNoChangeTo2DPhysicalBuffer(BaseCompare):
    """Flattening preserves axis separators."""

    def before():
        A = T.alloc_buffer([32, 32], axis_separators=[1])
        for i, j in T.grid(32, 32):
            T.evaluate(A[i, j])

    expected = before


class TestFlattenAllocBufferWithAxisSeparators(BaseCompare):
    """Flattening preserves axis separators"""

    def before():
        A = T.alloc_buffer([2, 3, 5, 7, 11, 13], axis_separators=[3])
        for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 5, 7, 11, 13):
            T.evaluate(A[i0, i1, i2, i3, i4, i5])

    def expected():
        A = T.alloc_buffer([30, 1001], axis_separators=[1])
        for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 5, 7, 11, 13):
            T.evaluate(A[i0 * 15 + i1 * 5 + i2, i3 * 143 + i4 * 13 + i5])


class TestFlattenDeclBufferWithAxisSeparators(BaseCompare):
    """Flattening preserves axis separators

    Like TestFlattenAllocBufferWithAxisSeparators, but the allocations
    is done using Allocate/DeclBuffer, rather than through
    BlockNode::alloc_buffers.
    """

    def before():
        A = T.decl_buffer([2, 3, 5, 7, 11, 13], axis_separators=[3])
        for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 5, 7, 11, 13):
            T.evaluate(A[i0, i1, i2, i3, i4, i5])

    def expected():
        A_data = T.allocate([30, 1001], dtype="float32", scope="global")
        A = T.Buffer([30, 1001], dtype="float32", scope="global", axis_separators=[1], data=A_data)
        for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 5, 7, 11, 13):
            T.evaluate(A[i0 * 15 + i1 * 5 + i2, i3 * 143 + i4 * 13 + i5])


def test_lower_2d_physical_memory():
    """Axis separators should preserve 2-d buffers through lowering.

    A catch-all test to ensure that defining axis_separators is
    sufficient to maintain non-flat buffer descriptions through all
    lowering steps.
    """

    # This test doesn't use CompareBeforeAfter, because the after step
    # is not currently expressible in TVMScript.  This test can be
    # re-written after https://github.com/apache/tvm/pull/12412.

    @T.prim_func
    def func():
        buf = T.alloc_buffer(
            [1, 1],
            dtype="int32",
            scope="global",
            axis_separators=[1],
        )
        buf[0, 0] = 0

    lowered = tvm.lower(func)["main"]
    assert isinstance(lowered.body, tvm.tir.Allocate)
    assert list(lowered.body.extents) == [1, 1], (
        "Non-flat buffer allocations, "
        "marked by axis_separators, "
        "flattened to flat memory allocation."
    )


if __name__ == "__main__":
    tvm.testing.main()
