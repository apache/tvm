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
from tvm import tirx as tir
from tvm.script import tirx as T
from tvm.tirx.cuda import op as cuda_op
from tvm.tirx.trn import op as trn_op


def _assert_print(obj, expected):
    # Standalone TIR nodes use the canonical tirx script prefix.
    out = obj.script(verbose_expr=True, extra_config={"tirx.prefix": "T"}).strip()
    assert out == expected.strip()


def test_printer_cuda_namespace_printf():
    node = tir.Evaluate(cuda_op.cuda_printf("x=%d", tir.IntImm("int32", 1)))
    _assert_print(node, 'T.cuda.printf("x=%d", 1)')


def test_printer_cuda_cluster_sync():
    node = tir.Evaluate(cuda_op.cuda_cluster_sync())
    _assert_print(node, "T.cuda.cluster_sync()")


def test_printer_cuda_namespace_mbarrier_wait():
    node = tir.Evaluate(cuda_op.cuda_mbarrier_wait(tir.IntImm("int32", 0), tir.IntImm("int32", 0)))
    _assert_print(node, "T.cuda.mbarrier_wait(0, 0)")


def test_printer_nvshmem_namespace():
    node = tir.Evaluate(cuda_op.nvshmem_fence())
    _assert_print(node, "T.nvshmem.fence()")


def test_printer_ptx_more():
    r = tir.Var("r", "handle")
    s = tir.Var("s", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    _assert_print(
        cuda_op.cuda_tcgen05_encode_matrix_descriptor(d, a, 1, 2, 0),
        "d = T.handle()\na = T.handle()\nT.cuda.tcgen05.encode_matrix_descriptor(d, a, 1, 2, 0)",
    )
    _assert_print(
        cuda_op.cuda_tcgen05_encode_instr_descriptor(
            d,
            d_dtype="f16",
            a_dtype="f16",
            b_dtype="f16",
            M=16,
            N=16,
            K=16,
            trans_a=True,
            trans_b=False,
            n_cta_groups=1,
            neg_a=False,
            neg_b=False,
            sat_d=False,
            is_sparse=False,
        ),
        'd = T.handle()\nT.cuda.tcgen05.encode_instr_descriptor(d, "f16", "f16", "f16", 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_print(
        cuda_op.cuda_tcgen05_encode_instr_descriptor_block_scaled(
            d,
            d_dtype="f16",
            a_dtype="f16",
            b_dtype="f16",
            sfa_dtype="f16",
            sfb_dtype="f16",
            sfa_tmem_addr=a,
            sfb_tmem_addr=b,
            M=16,
            N=16,
            K=16,
            trans_a=True,
            trans_b=False,
            is_sparse=True,
            n_cta_groups=1,
            neg_a=False,
            neg_b=False,
        ),
        "d = T.handle()\n"
        "a = T.handle()\n"
        "b = T.handle()\n"
        'T.cuda.tcgen05.encode_instr_descriptor_block_scaled(d, "f16", "f16", "f16", "f16", "f16", a, b, 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(True))',  # noqa: E501
    )


def test_printer_cuda_mbarrier_wait_var():
    bar = tir.Var("bar", "handle")
    _assert_print(
        cuda_op.cuda_mbarrier_wait(bar, 1), "bar = T.handle()\nT.cuda.mbarrier_wait(bar, 1)"
    )
    _assert_print(cuda_op.cuda_cluster_sync(), "T.cuda.cluster_sync()")


def test_printer_cuda_more():
    p = tir.Var("p", "handle")
    _assert_print(cuda_op.cuda_thread_fence(), "T.cuda.thread_fence()")
    _assert_print(cuda_op.cuda_warp_sync(), "T.cuda.warp_sync()")
    _assert_print(cuda_op.cuda_cta_sync(), "T.cuda.cta_sync()")
    _assert_print(cuda_op.cuda_grid_sync(), "T.cuda.grid_sync()")
    _assert_print(cuda_op.cuda_cluster_sync(), "T.cuda.cluster_sync()")
    _assert_print(cuda_op.cuda_syncthreads_and(1), "T.cuda.syncthreads_and(1)")
    _assert_print(cuda_op.cuda_syncthreads_or(1), "T.cuda.syncthreads_or(1)")
    _assert_print(cuda_op.cuda_nano_sleep(100), "T.cuda.nano_sleep(100)")
    _assert_print(
        cuda_op.cuda_atomic_add(p, tir.IntImm("int32", 1)),
        "p = T.handle()\nT.cuda.atomic_add(p, 1)",
    )
    _assert_print(cuda_op.cuda_atomic_cas(p, 1, 2), "p = T.handle()\nT.cuda.atomic_cas(p, 1, 2)")
    _assert_print(cuda_op.cuda_ldg(p, "float32"), 'p = T.handle()\nT.cuda.ldg(p, "float32")')
    _assert_print(
        cuda_op.cuda_func_call("f", 1, source_code=""), 'T.cuda.func_call("f", 1, source_code="")'
    )


def test_printer_cuda_low_level_warp_intrinsics_roundtrip():
    @T.prim_func(check_well_formed=False)
    def kernel():
        x = T.int32()
        mask = T.cuda.__activemask()
        T.evaluate(T.cuda.__shfl_sync(mask, x, 0, 32))
        T.evaluate(T.cuda.__shfl_up_sync(mask, x, 1, 32))
        T.evaluate(T.cuda.__shfl_down_sync(mask, x, 1, 32))
        T.evaluate(T.cuda.__shfl_xor_sync(mask, x, 1, 32))

    code = kernel.script()
    assert "T.cuda.__activemask()" in code
    assert "T.cuda.__shfl_sync(" in code
    assert "T.cuda.__shfl_up_sync(" in code
    assert "T.cuda.__shfl_down_sync(" in code
    assert "T.cuda.__shfl_xor_sync(" in code
    assert "T.tirx." not in code
    assert tvm.script.from_source(code).script() == code


def test_printer_webgpu_namespace_roundtrip():
    @T.prim_func(check_well_formed=False)
    def kernel():
        x = T.int32()
        T.evaluate(T.webgpu.subgroup_shuffle(x, 0))
        T.evaluate(T.webgpu.subgroup_shuffle_up(x, 1))
        T.evaluate(T.webgpu.subgroup_shuffle_down(x, 1))

    code = kernel.script()
    assert "T.webgpu.subgroup_shuffle(" in code
    assert "T.webgpu.subgroup_shuffle_up(" in code
    assert "T.webgpu.subgroup_shuffle_down(" in code
    assert "T.tirx." not in code
    assert tvm.script.from_source(code).script() == code


def test_printer_nvshmem_more():
    p = tir.Var("p", "handle")
    _assert_print(cuda_op.nvshmem_my_pe(), "T.nvshmem.my_pe()")
    _assert_print(cuda_op.nvshmem_n_pes(), "T.nvshmem.n_pes()")
    _assert_print(
        cuda_op.nvshmem_signal_op(p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.signal_op(p, 1, "set", 0)',
    )
    _assert_print(
        cuda_op.nvshmem_wait_until(p, "eq", 0),
        'p = T.handle()\nT.nvshmem.wait_until(p, "eq", 0, "uint64_t")',
    )
    _assert_print(cuda_op.nvshmem_quiet(), "T.nvshmem.quiet()")
    _assert_print(cuda_op.nvshmem_barrier_all(), "T.nvshmem.barrier_all()")
    _assert_print(
        cuda_op.nvshmem_getmem_nbi(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.getmem_nbi(p, p, 16, 0)",
    )
    _assert_print(
        cuda_op.nvshmem_getmem_nbi_warp(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.getmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_print(
        cuda_op.nvshmem_putmem_nbi_block(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi.block(p, p, 16, 0)",
    )
    _assert_print(
        cuda_op.nvshmem_putmem_nbi(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi(p, p, 16, 0)",
    )
    _assert_print(
        cuda_op.nvshmem_putmem_nbi_warp(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_print(
        cuda_op.nvshmem_putmem_signal_nbi(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi(p, p, 16, p, 1, "set", 0)',
    )
    _assert_print(
        cuda_op.nvshmem_putmem_signal_nbi_warp(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.warp(p, p, 16, p, 1, "set", 0)',
    )
    _assert_print(
        cuda_op.nvshmem_putmem_signal_nbi_block(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.block(p, p, 16, p, 1, "set", 0)',
    )


def test_printer_nki_namespace():
    A = tir.decl_buffer([1], dtype="float16", name="A")
    B = tir.decl_buffer([1], dtype="float16", name="B")
    a0 = A[0]
    b0 = B[0]
    _assert_print(
        trn_op.nki_load(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.load(A, B)',
    )
    _assert_print(
        trn_op.nki_store(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.store(A, B)',
    )
    _assert_print(
        trn_op.nki_tensor_copy(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.tensor_copy(A, B)',
    )
    _assert_print(
        trn_op.nki_matmul(a0, a0, b0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        "T.nki.matmul(A, A, B, T.bool(True))",
    )
    _assert_print(
        trn_op.nki_activation(a0, b0, "relu", 0.0, 1.0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.activation(A, B, "relu", T.float32(0.0), T.float32(1.0))',
    )
    _assert_print(
        trn_op.nki_memset(a0, 0),
        'A = T.Buffer((1,), "float16")\nT.nki.memset(A, 0)',
    )
    _assert_print(
        trn_op.nki_identity(a0, 1),
        'A = T.Buffer((1,), "float16")\nT.nki.identity(A, 1)',
    )
    _assert_print(
        trn_op.nki_reciprocal(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.reciprocal(A, B)',
    )
    _assert_print(
        trn_op.nki_tensorreduce(a0, b0, "sum", False, 0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.tensorreduce(A, B, "sum", T.bool(False), 0)',
    )
    _assert_print(
        trn_op.nki_tensortensor(a0, a0, b0, "add"),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.tensortensor(A, A, B, "add")',
    )
    _assert_print(
        trn_op.nki_tensorscalar(a0, a0, 1.0, "mul", False),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.tensorscalar(A, A, T.float32(1.0), "mul", T.bool(False))',
    )
    _assert_print(
        trn_op.nki_tensorscalar_reduce(a0, a0, 1.0, "mul", "sum", False),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.tensorscalar_reduce(A, A, T.float32(1.0), "mul", "sum", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_print(
        trn_op.nki_scalar_tensor_tensor(a0, a0, 1.0, a0, "add", "add"),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.scalar_tensor_tensor(A, A, T.float32(1.0), A, "add", "add", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_print(
        trn_op.nki_scalar_tensor_scalar(a0, a0, 1.0, 1.0, "add", "add"),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.scalar_tensor_scalar(A, A, T.float32(1.0), T.float32(1.0), "add", "add", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_print(
        trn_op.nki_activation_reduce(a0, a0, b0, "relu", "sum", 0.0, 1.0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.activation_reduce(A, A, B, "relu", "sum", T.float32(0.0), T.float32(1.0))',
    )
    _assert_print(
        trn_op.nki_affine_select(a0, a0, a0, 1.0),
        'A = T.Buffer((1,), "float16")\nT.nki.affine_select(A, A, A, T.float32(1.0))',
    )


def test_printer_ptx_mma_and_wgmma():
    r = tir.Var("r", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    tir.Var("b", "handle")
    _assert_print(
        cuda_op.cuda_wgmma_encode_matrix_descriptor(d, a, 1, 1, 0),
        "d = T.handle()\na = T.handle()\nT.cuda.wgmma.encode_matrix_descriptor(d, a, 1, 1, 0)",
    )
    _assert_print(cuda_op.cuda_wgmma_noop_barrier(0), "T.cuda.wgmma.noop_barrier(0)")
