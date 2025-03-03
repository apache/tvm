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
from tvm.topi.math import cast
from tvm.script import tir as T


class TestMatmul(tvm.testing.CompareBeforeAfter):
    """Shared allocations should be merged, preserving DeclBuffer if present

    This test uses a matmul PrimFunc adapted from
    test_matmul_dyn_shared, using either `T.Buffer` (Allocate without
    DeclBuffer) or `T.decl_buffer` (Allocate followed by DeclBuffer)
    for the replaced allocations.
    """

    transform = tvm.tir.transform.MergeSharedMemoryAllocations()

    use_decl_buffer = tvm.testing.parameter(by_dict={"t_buffer": False, "decl_buffer": True})

    @tvm.testing.fixture
    def buffer_func(self, use_decl_buffer):
        if use_decl_buffer:
            return T.decl_buffer
        else:
            return T.Buffer

    @tvm.testing.fixture
    def before(self, buffer_func):
        @T.prim_func
        def func(
            A: T.Buffer((1024, 1024), "float16"),
            B: T.Buffer((1024, 1024), "float16"),
            matmul: T.Buffer((1024, 1024), "float32"),
        ):
            A_flat = T.Buffer(1048576, "float16", data=A.data)
            B_flat = T.Buffer(1048576, "float16", data=B.data)
            matmul_flat = T.Buffer(1048576, data=matmul.data)

            threadIdx_x = T.launch_thread("threadIdx.x", 16)
            C_local_data = T.allocate([1], "float32", "local")
            C_local = T.Buffer(1, data=C_local_data, scope="local")

            A_sh_data = T.allocate([256], "float16", "shared.dyn")
            A_sh = buffer_func(256, "float16", data=A_sh_data, scope="shared.dyn")
            B_sh_data = T.allocate([256], "float16", "shared.dyn")
            B_sh = buffer_func(256, "float16", data=B_sh_data, scope="shared.dyn")
            C_sh_data = T.allocate([256], "float32", "shared.dyn")
            C_sh = buffer_func(256, "float32", data=C_sh_data, scope="shared.dyn")

            threadIdx_y = T.launch_thread("threadIdx.y", 16)
            blockIdx_x = T.launch_thread("blockIdx.x", 64)
            blockIdx_y = T.launch_thread("blockIdx.y", 64)

            C_local[0] = T.float32(0)
            for i in range(64):
                A_sh[threadIdx_y * 16 + threadIdx_x] = A_flat[
                    blockIdx_y * 16384 + threadIdx_y * 1024 + i * 16 + threadIdx_x
                ]

                B_sh[threadIdx_y * 16 + threadIdx_x] = B_flat[
                    i * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
                ]
                T.tvm_storage_sync("shared")
                for k in range(16):
                    C_local[0] = C_local[0] + T.Cast(
                        "float32",
                        A_sh[threadIdx_y * 16 + k] * B_sh[k * 16 + threadIdx_x],
                    )
                T.tvm_storage_sync("shared")

            C_sh[threadIdx_y * 16 + threadIdx_x] = C_local[0]
            T.tvm_storage_sync("shared.dyn")

            matmul_flat[
                blockIdx_y * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
            ] = C_sh[threadIdx_y * 16 + threadIdx_x]

        return func

    @tvm.testing.fixture
    def expected(self, buffer_func):
        @T.prim_func
        def func(
            A: T.Buffer((1024, 1024), "float16"),
            B: T.Buffer((1024, 1024), "float16"),
            matmul: T.Buffer((1024, 1024), "float32"),
        ):
            A_flat = T.Buffer(1048576, "float16", data=A.data)
            B_flat = T.Buffer(1048576, "float16", data=B.data)
            matmul_flat = T.Buffer(1048576, data=matmul.data)

            threadIdx_x = T.launch_thread("threadIdx.x", 16)

            buf_dyn_shmem = T.allocate([1024], "uint8", "shared.dyn")

            C_local_data = T.allocate([1], "float32", "local")
            C_local = T.Buffer(1, data=C_local_data, scope="local")

            A_sh = buffer_func(256, "float16", data=buf_dyn_shmem, scope="shared.dyn")
            B_sh = buffer_func(256, "float16", data=buf_dyn_shmem, scope="shared.dyn")
            C_sh = buffer_func(256, "float32", data=buf_dyn_shmem, scope="shared.dyn")

            threadIdx_y = T.launch_thread("threadIdx.y", 16)
            blockIdx_x = T.launch_thread("blockIdx.x", 64)
            blockIdx_y = T.launch_thread("blockIdx.y", 64)

            C_local[0] = T.float32(0)
            for i in range(64):
                A_sh[threadIdx_y * 16 + threadIdx_x + 256] = A_flat[
                    blockIdx_y * 16384 + threadIdx_y * 1024 + i * 16 + threadIdx_x
                ]
                B_sh[threadIdx_y * 16 + threadIdx_x] = B_flat[
                    i * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
                ]
                T.tvm_storage_sync("shared")
                for k in range(16):
                    C_local[0] = C_local[0] + T.Cast(
                        "float32",
                        A_sh[threadIdx_y * 16 + k + 256] * B_sh[k * 16 + threadIdx_x],
                    )
                T.tvm_storage_sync("shared")

            C_sh[threadIdx_y * 16 + threadIdx_x] = C_local[0]
            T.tvm_storage_sync("shared.dyn")

            matmul_flat[
                blockIdx_y * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
            ] = C_sh[threadIdx_y * 16 + threadIdx_x]

        return func


class TestSimpleAllocNoReuse(tvm.testing.CompareBeforeAfter):
    """Test alloc and free within the same scope."""

    transform = tvm.tir.transform.MergeSharedMemoryAllocations()

    def before(self):
        @T.prim_func
        def func():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh_data = T.allocate([128], "float32", "shared.dyn")
            B_sh_data = T.allocate([128], "float32", "shared.dyn")
            A_sh = T.decl_buffer([128], data=A_sh_data, scope="shared.dyn")
            B_sh = T.decl_buffer([128], data=B_sh_data, scope="shared.dyn")
            B_sh[threadIdx_x] = A_sh[threadIdx_x]

        return func

    def expected(self):
        @T.prim_func
        def func():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            buf_dyn_shmem = T.allocate([1024], "uint8", "shared.dyn")
            A_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
            B_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
            B_sh[threadIdx_x + 128] = A_sh[threadIdx_x]

        return func


class TestSimpleAllocReuse(tvm.testing.CompareBeforeAfter):
    """Test alloc and free within the same scope with a reuse chance."""

    transform = tvm.tir.transform.MergeSharedMemoryAllocations()

    def before(self):
        @T.prim_func
        def func():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh_data = T.allocate([128], "float32", "shared.dyn")
            B_sh_data = T.allocate([128], "float32", "shared.dyn")
            A_sh = T.decl_buffer([128], data=A_sh_data, scope="shared.dyn")
            B_sh = T.decl_buffer([128], data=B_sh_data, scope="shared.dyn")
            A_sh[threadIdx_x] = 0
            B_sh[threadIdx_x] = 0

        return func

    def expected(self):
        @T.prim_func
        def func():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            buf_dyn_shmem = T.allocate([512], "uint8", "shared.dyn")
            A_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
            B_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
            A_sh[threadIdx_x] = 0
            B_sh[threadIdx_x] = 0

        return func


class TestAsyncCopy(tvm.testing.CompareBeforeAfter):
    """Test async copy in shared memory."""

    transform = tvm.tir.transform.MergeSharedMemoryAllocations()

    def before(self):
        @T.prim_func
        def func(A: T.buffer((128)), B: T.buffer((128))):
            A_sh_data = T.allocate([128], "float32", "shared.dyn")
            B_sh_data = T.allocate([128], "float32", "shared.dyn")
            A_sh = T.buffer([128], data=A_sh_data, scope="shared.dyn")
            B_sh = T.buffer([128], data=B_sh_data, scope="shared.dyn")
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            T.ptx_cp_async("float32", A_sh.data, threadIdx_x, A.data, threadIdx_x, 512)
            T.ptx_cp_async("float32", B_sh.data, threadIdx_x, B.data, threadIdx_x, 512)

        return func

    def expected(self):
        @T.prim_func
        def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            buf_dyn_shmem = T.allocate([1024], "uint8", "shared.dyn")
            T.ptx_cp_async("float32", buf_dyn_shmem, threadIdx_x * 4, A.data, threadIdx_x, 512)
            T.ptx_cp_async(
                "float32", buf_dyn_shmem, (128 + threadIdx_x) * 4, B.data, threadIdx_x, 512
            )

        return func


if __name__ == "__main__":
    tvm.testing.main()
