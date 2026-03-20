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
# ruff: noqa: F401, F821, F841
import tvm
import tvm.testing
from tvm import s_tir
from tvm.script import tirx as T


def run_passes(func: tvm.tirx.PrimFunc):
    mod = tvm.IRModule.from_expr(func)

    cuda_target = tvm.target.Target("cuda", host="llvm")

    mod = tvm.tirx.transform.Apply(
        lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target})
    )(mod)

    mod = tvm.tirx.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    return tvm.s_tir.transform.ThreadSync("shared")(mod)


@tvm.testing.requires_cuda
def test_sync_read_thread_id_independent_location():
    @T.prim_func(check_well_formed=False)
    def func(p0_arg: T.Buffer((1, 2, 1, 1), "float32"), p1: T.Buffer(2, "float32")) -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.Buffer([2], dtype="float32", data=p0_arg.data)
        result_local = T.sblock_alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.sblock_alloc_buffer([1], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 8)
        T.launch_thread(threadIdx_x, 4)
        result_local[0] = T.float32(0)
        if threadIdx_x < 1:
            temp_shared[0] = p0[0]
        result_local[0] = result_local[0] + temp_shared[0] * p1[0]
        if threadIdx_x < 1:
            temp_shared[0] = p0[1]
        result_local[0] = result_local[0] + temp_shared[0] * p1[1]

    mod = run_passes(func)
    assert "T.tvm_storage_sync" in str(mod)


def test_sync_shared_dyn():
    @T.prim_func(private=True)
    def func(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B = T.alloc_buffer((24,), "float32", scope="shared.dyn")
        C = T.alloc_buffer((1,), "float32", scope="local")
        D = T.alloc_buffer((16,), "float32", scope="shared.dyn")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        B_1 = T.decl_buffer((24,), data=B.data, scope="shared.dyn")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1 = T.decl_buffer((1,), data=C.data, scope="local")
        C_1[0] = B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1 = T.decl_buffer((16,), data=D.data, scope="shared.dyn")
        D_1[threadIdx_x] = C_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1[threadIdx_x]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B_1 = T.alloc_buffer((24,), "float32", scope="shared.dyn")
        C_1 = T.alloc_buffer((1,), "float32", scope="local")
        D_1 = T.alloc_buffer((16,), "float32", scope="shared.dyn")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        B_1_1 = T.decl_buffer((24,), data=B_1.data, scope="shared.dyn")
        A_1 = T.decl_buffer((16,), data=A.data)
        B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1_1 = T.decl_buffer((1,), data=C_1.data, scope="local")
        C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1_1 = T.decl_buffer((16,), data=D_1.data, scope="shared.dyn")
        T.tvm_storage_sync("shared.dyn")
        D_1_1[threadIdx_x] = C_1_1[0]
        E_1 = T.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = D_1_1[threadIdx_x]

    mod = tvm.IRModule({"main": func})
    mod = tvm.s_tir.transform.ThreadSync("shared.dyn")(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@tvm.testing.requires_cuda
def test_sync_bind():
    @T.prim_func(private=True)
    def func(A: T.Buffer((16 * 512), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared = T.alloc_buffer((512,), "float32", scope="shared")
        in_thread_A_temp = T.alloc_buffer((1,), "float32", scope="local")
        cross_thread_A_temp = T.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        A_shared_1 = T.decl_buffer((512,), data=A_shared.data, scope="shared")
        for ax0 in range(512):
            A_shared_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1 = T.decl_buffer((1,), data=in_thread_A_temp.data, scope="local")
        in_thread_A_temp_1[0] = T.float32(0)
        A_temp_1 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x])
        in_thread_A_temp_1[0] = A_temp_1
        A_temp_2 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 128])
        in_thread_A_temp_1[0] = A_temp_2
        A_temp_3 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 256])
        in_thread_A_temp_1[0] = A_temp_3
        A_temp_4 = T.bind(in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 384])
        in_thread_A_temp_1[0] = A_temp_4
        cross_thread_A_temp_1 = T.decl_buffer((1,), data=cross_thread_A_temp.data, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            T.tvm_thread_allreduce(
                T.uint32(1),
                in_thread_A_temp_1[0],
                T.bool(True),
                cross_thread_A_temp_1[0],
                threadIdx_x,
            )

    @T.prim_func(private=True)
    def expected(A: T.Buffer((8192,), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared_1 = T.alloc_buffer((512,), "float32", scope="shared")
        in_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        cross_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        A_shared_1_1 = T.decl_buffer((512,), data=A_shared_1.data, scope="shared")
        for ax0 in range(512):
            A_shared_1_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1_1 = T.decl_buffer((1,), data=in_thread_A_temp_1.data, scope="local")
        in_thread_A_temp_1_1[0] = T.float32(0)
        T.tvm_storage_sync("shared")
        A_temp_1 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x])
        in_thread_A_temp_1_1[0] = A_temp_1
        A_temp_2 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 128])
        in_thread_A_temp_1_1[0] = A_temp_2
        A_temp_3 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 256])
        in_thread_A_temp_1_1[0] = A_temp_3
        A_temp_4 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 384])
        in_thread_A_temp_1_1[0] = A_temp_4
        cross_thread_A_temp_1_1 = T.decl_buffer(
            (1,), data=cross_thread_A_temp_1.data, scope="local"
        )
        T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        )
        T.tvm_thread_allreduce(
            T.uint32(1),
            in_thread_A_temp_1_1[0],
            T.bool(True),
            cross_thread_A_temp_1_1[0],
            threadIdx_x,
        )

    mod = tvm.IRModule({"main": func})
    mod = tvm.s_tir.transform.ThreadSync("shared")(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    test_thread_storage_sync()
    test_sync_else_branch()
    test_sync_read_thread_id_independent_location()
    test_sync_shared_dyn()
    test_sync_bind()
