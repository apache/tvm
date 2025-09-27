"""
Tests for verifying shared memory limit using a deterministic TIR kernel.

To enable optional pipeline verification during normal builds (opt-in, off by default):

    with tvm.transform.PassContext(config={
        "tir.verify_gpu_code": True,
        # For Ampere (e.g., RTX A6000, SM 86), you may set a higher cap, e.g., 96 KB:
        # "tir.cuda.max_shared_memory_per_block": 96 * 1024,
        # By default, leave unset to use a conservative 48 KB.
    }):
        lib = tvm.tir.build(mod, target="cuda")

This test avoids schedule/lowering variability by using a direct kernel that allocates a
64 KB shared buffer and asserts the verifier fails when the cap is 48 KB.
"""

import pytest

import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def _pf_direct_kernel_shared_large(A: T.handle) -> None:
    T.func_attr({"global_symbol": "_pf_direct_kernel_shared_large", "tir.noalias": True})
    A_buf = T.match_buffer(A, (1,), dtype="float32")
    blockIdx_x = T.launch_thread("blockIdx.x", 1)
    threadIdx_x = T.launch_thread("threadIdx.x", 1)
    # 16384 float32 elements = 64 KB shared allocation
    sh = T.allocate([16384], "float32", "shared")
    s = T.float32(0)
    for t in T.serial(0, 16384):
        s = s + T.float32(1)
    A_buf[0] = s


def test_direct_kernel_shared_overflow_verify_false():
    mod = tvm.IRModule({"main": _pf_direct_kernel_shared_large})
    vbool = tvm.get_global_func("tir.analysis.verify_gpu_code")
    ok = vbool(
        mod["main"], {"max_shared_memory_per_block": 48 * 1024, "max_threads_per_block": 1024}
    )
    assert not ok

