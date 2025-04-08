import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import ir as I
from tvm.script import tir as T


# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32"), D: T.Buffer((128, 128), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-unknown-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32}), "tir.is_entry_func": T.bool(True)})
    local_c = T.allocate([16384], "float32", "local")
    local_c_1 = T.Buffer((16384,), data=local_c, scope="local")
    with T.attr(T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32}), "target", 0):
        blockIdx_x = T.launch_thread("blockIdx.x", 64)
        local_c_local = T.allocate([16], "float32", "local")
        A_shared = T.allocate([512], "float32", "shared")
        B_shared = T.allocate([2048], "float32", "shared")
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        local_c_local_1 = T.Buffer((16,), data=local_c_local, scope="local")
        for k_0 in range(2):
            T.tvm_storage_sync("shared")
            A_shared_1 = T.Buffer((512,), data=A_shared, scope="shared")
            for ax0_ax1_fused in range(512):
                A_1 = T.Buffer((16384,), data=A.data)
                A_shared_1[ax0_ax1_fused] = A_1[blockIdx_x // 4 * 1024 + ax0_ax1_fused // 64 * 128 + k_0 * 64 + ax0_ax1_fused % 64]
            B_shared_1 = T.Buffer((2048,), data=B_shared, scope="shared")
            for ax0_ax1_fused in range(2048):
                B_1 = T.Buffer((16384,), data=B.data)
                B_shared_1[ax0_ax1_fused] = B_1[k_0 * 8192 + ax0_ax1_fused // 32 * 128 + blockIdx_x % 4 * 32 + ax0_ax1_fused % 32]
            T.tvm_storage_sync("shared")
            for k_1, j_3, k_2 in T.grid(16, 4, 4):
                cse_var_3: T.int32 = j_3 + 8
                cse_var_2: T.int32 = j_3 + 4
                cse_var_1: T.int32 = j_3 + 12
                if k_0 * 64 + k_1 * 4 + k_2 == 0:
                    local_c_local_1[j_3] = T.float32(0.0)
                    local_c_local_1[cse_var_2] = T.float32(0.0)
                    local_c_local_1[cse_var_3] = T.float32(0.0)
                    local_c_local_1[cse_var_1] = T.float32(0.0)
                local_c_local_1[j_3] = local_c_local_1[j_3] + A_shared_1[threadIdx_x // 4 * 64 + k_1 * 4 + k_2] * B_shared_1[k_1 * 128 + k_2 * 32 + threadIdx_x % 4 * 4 + j_3]
                local_c_local_1[cse_var_2] = local_c_local_1[cse_var_2] + A_shared_1[threadIdx_x // 4 * 64 + k_1 * 4 + k_2] * B_shared_1[k_1 * 128 + k_2 * 32 + threadIdx_x % 4 * 4 + j_3 + 16]
                local_c_local_1[cse_var_3] = local_c_local_1[cse_var_3] + A_shared_1[threadIdx_x // 4 * 64 + k_1 * 4 + k_2 + 256] * B_shared_1[k_1 * 128 + k_2 * 32 + threadIdx_x % 4 * 4 + j_3]
                local_c_local_1[cse_var_1] = local_c_local_1[cse_var_1] + A_shared_1[threadIdx_x // 4 * 64 + k_1 * 4 + k_2 + 256] * B_shared_1[k_1 * 128 + k_2 * 32 + threadIdx_x % 4 * 4 + j_3 + 16]
        for ax1 in range(4):
            local_c_1[blockIdx_x // 4 * 1024 + threadIdx_x // 4 * 128 + blockIdx_x % 4 * 32 + threadIdx_x % 4 * 4 + ax1] = local_c_local_1[ax1]
            local_c_1[blockIdx_x // 4 * 1024 + threadIdx_x // 4 * 128 + blockIdx_x % 4 * 32 + threadIdx_x % 4 * 4 + ax1 + 16] = local_c_local_1[ax1 + 4]
            local_c_1[blockIdx_x // 4 * 1024 + threadIdx_x // 4 * 128 + blockIdx_x % 4 * 32 + threadIdx_x % 4 * 4 + ax1 + 512] = local_c_local_1[ax1 + 8]
            local_c_1[blockIdx_x // 4 * 1024 + threadIdx_x // 4 * 128 + blockIdx_x % 4 * 32 + threadIdx_x % 4 * 4 + ax1 + 528] = local_c_local_1[ax1 + 12]
    T.attr(T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32}), "target", 0)
    blockIdx_x = T.launch_thread("blockIdx.x", 2)
    local_c_shared = T.allocate([8192], "float32", "shared")
    C_shared = T.allocate([16384], "float32", "shared")
    D_local = T.allocate([2048], "float32", "local")
    threadIdx_x = T.launch_thread("threadIdx.x", 4)
    local_c_shared_1 = T.Buffer((8192,), data=local_c_shared, scope="shared")
    for ax0_ax1_fused in range(8192):
        local_c_shared_1[ax0_ax1_fused] = local_c_1[blockIdx_x * 8192 + ax0_ax1_fused]
    C_shared_1 = T.Buffer((16384,), data=C_shared, scope="shared")
    for ax0_ax1_fused in range(16384):
        C_1 = T.Buffer((16384,), data=C.data)
        C_shared_1[ax0_ax1_fused] = C_1[ax0_ax1_fused]
    T.tvm_storage_sync("shared")
    D_local_1 = T.Buffer((16,), data=D_local, scope="local")
    for k_1, i_3, k_2, j_4 in T.grid(32, 2, 4, 2):
        if k_1 * 4 + k_2 == 0:
            for vthread_x_s in range(512):
                D_local_1[vthread_x_s * 4 + i_3 * 2 + j_4] = T.float32(0.0)
        for vthread_x_s in range(512):
            cse_var_4: T.int32 = vthread_x_s * 4 + i_3 * 2 + j_4
            D_local_1[cse_var_4] = D_local_1[cse_var_4] + local_c_shared_1[vthread_x_s // 32 * 512 + threadIdx_x // 2 * 256 + i_3 * 128 + k_1 * 4 + k_2] * C_shared_1[k_1 * 512 + k_2 * 128 + vthread_x_s % 32 * 4 + threadIdx_x % 2 * 2 + j_4]
    for ax0, ax1, vthread_x_s in T.grid(2, 2, 512):
        D_1 = T.Buffer((16384,), data=D.data)
        D_1[blockIdx_x * 8192 + vthread_x_s // 32 * 512 + threadIdx_x // 2 * 256 + ax0 * 128 + vthread_x_s % 32 * 4 + threadIdx_x % 2 * 2 + ax1] = D_local_1[vthread_x_s * 4 + ax0 * 2 + ax1]


if __name__ == "__main__":
    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.FuseThreadBindings()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod)