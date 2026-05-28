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


from tvm import tirx as tir


def _assert_print(obj, expected):
    # Use Tx prefix so standalone TIR nodes (non-PrimFunc) print as Tx to match tirx namespace
    out = obj.script(verbose_expr=True, extra_config={"tirx.prefix": "Tx"}).strip()
    assert out == expected.strip()


def test_printer_cuda_namespace_printf():
    node = tir.Evaluate(tir.op.cuda_printf("x=%d", tir.IntImm("int32", 1)))
    _assert_print(node, 'Tx.cuda.printf("x=%d", 1)')


def test_printer_ptx_namespace_wgmma_commit_group():
    node = tir.Evaluate(tir.op.ptx_wgmma_commit_group())
    _assert_print(node, "Tx.ptx.wgmma.commit_group()")


def test_printer_cuda_cluster_sync():
    node = tir.Evaluate(tir.op.cuda_cluster_sync())
    _assert_print(node, "Tx.cuda.cluster_sync()")


def test_printer_ptx_namespace_cp_async_wait_group():
    node = tir.Evaluate(tir.op.ptx_cp_async_wait_group(tir.IntImm("int32", 0)))
    _assert_print(node, "Tx.ptx.cp_async.wait_group(0)")


def test_printer_nvshmem_namespace():
    node = tir.Evaluate(tir.op.nvshmem_fence())
    _assert_print(node, "Tx.nvshmem.fence()")


def test_printer_ptx_more():
    r = tir.Var("r", "handle")
    s = tir.Var("s", "handle")
    _assert_print(
        # New API: (trans, num, dtype, smem_ptr, *dst_handles).
        # .x1.b16 has 1 dst register, so 1 dst handle.
        tir.op.ptx_ldmatrix(True, 1, ".b16", s, r),
        's = Tx.handle()\nr = Tx.handle()\nTx.ptx.ldmatrix("void", Tx.bool(True), 1, ".b16", s, r)',
    )
    _assert_print(
        tir.op.ptx_stmatrix(s, r, num=1, trans=False),
        (
            "s = Tx.handle()\nr = Tx.handle()\nTx.ptx.stmatrix("
            '1, Tx.bool(False), "m8n8", "b16", "shared", s, r)'
        ),
    )
    _assert_print(tir.op.ptx_setmaxnreg(True, 64), "Tx.ptx.setmaxnreg(Tx.bool(True), 64)")
    _assert_print(tir.op.ptx_fetch_register(32, "laneid"), 'Tx.ptx.fetch_register(32, "laneid")')
    _assert_print(tir.op.ptx_wgmma_fence(), "Tx.ptx.wgmma.fence()")
    _assert_print(tir.op.ptx_wgmma_wait_group(0), "Tx.ptx.wgmma.wait_group(0)")
    _assert_print(tir.op.ptx_cp_async_commit_group(), "Tx.ptx.cp_async.commit_group()")
    _assert_print(tir.op.ptx_cp_async_bulk_commit_group(), "Tx.ptx.cp_async.bulk.commit_group()")
    _assert_print(
        tir.op.ptx_cp_async_bulk_wait_group(0, True),
        "Tx.ptx.cp_async.bulk.wait_group(0, Tx.bool(True))",
    )
    _assert_print(tir.op.ptx_cp_async_mbarrier_arrive(0), "Tx.ptx.cp_async.mbarrier.arrive(0)")
    _assert_print(tir.op.ptx_fence("acq_rel", "gpu"), 'Tx.ptx.fence("acq_rel", "gpu")')
    _assert_print(tir.op.ptx_fence("sc", "cta"), 'Tx.ptx.fence("sc", "cta")')
    _assert_print(
        tir.op.ptx_fence_proxy_async("shared::cta"), 'Tx.ptx.fence.proxy_async("shared::cta")'
    )
    _assert_print(tir.op.ptx_fence_proxy_async("global"), 'Tx.ptx.fence.proxy_async("global")')
    _assert_print(tir.op.ptx_fence_mbarrier_init(), "Tx.ptx.fence.mbarrier_init()")
    _assert_print(tir.op.ptx_elect_sync(), "Tx.ptx.elect_sync()")
    lane = tir.Var("lane", "int32")
    _assert_print(
        tir.op.selector(lane, tir.op.ptx_elect_sync()),
        "lane = Tx.int32()\nTx.selector(lane, Tx.ptx.elect_sync())",
    )
    _assert_print(
        tir.op.ptx_ld_global_acquire(r, s),
        "r = Tx.handle()\ns = Tx.handle()\nTx.ptx.ld_global_acquire(r, s)",
    )
    _assert_print(
        tir.op.ptx_map_shared_rank(r, 2), 'r = Tx.handle()\nTx.ptx.mapa(r, 2, "", "u64", "uint64")'
    )
    _assert_print(tir.op.ptx_bar_arrive(0, 128), "Tx.ptx.bar.arrive(0, 128)")
    _assert_print(tir.op.ptx_bar_sync(0, 128), "Tx.ptx.bar.sync(0, 128)")
    _assert_print(
        tir.op.ptx_tcgen05_alloc(s, 64, 1), "s = Tx.handle()\nTx.ptx.tcgen05.alloc(s, 64, 1)"
    )
    _assert_print(
        tir.op.ptx_tcgen05_dealloc(s, 64, 1), "s = Tx.handle()\nTx.ptx.tcgen05.dealloc(s, 64, 1)"
    )
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    _assert_print(
        tir.op.ptx_tcgen05_encode_matrix_descriptor(d, a, 1, 2, 0),
        "d = Tx.handle()\na = Tx.handle()\nTx.ptx.tcgen05.encode_matrix_descriptor(d, a, 1, 2, 0)",
    )
    _assert_print(
        tir.op.ptx_tcgen05_encode_instr_descriptor(
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
        'd = Tx.handle()\nTx.ptx.tcgen05.encode_instr_descriptor(d, "f16", "f16", "f16", 16, 16, 16, Tx.bool(True), Tx.bool(False), 1, Tx.bool(False), Tx.bool(False), Tx.bool(False), Tx.bool(False))',  # noqa: E501
    )
    _assert_print(
        tir.op.ptx_tcgen05_encode_instr_descriptor_block_scaled(
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
        "d = Tx.handle()\n"
        "a = Tx.handle()\n"
        "b = Tx.handle()\n"
        'Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(d, "f16", "f16", "f16", "f16", "f16", a, b, 16, 16, 16, Tx.bool(True), Tx.bool(False), 1, Tx.bool(False), Tx.bool(False), Tx.bool(True))',  # noqa: E501
    )
    _assert_print(
        tir.op.ptx_tcgen05_cp(a, d, shape="64x128b", cta_group=1, multicast="warpx2::02_13"),
        "a = Tx.handle()\n"
        "d = Tx.handle()\n"
        'Tx.ptx.tcgen05.cp(a, d, "64x128b", 1, "warpx2::02_13", "", 0, 0)',
    )
    _assert_print(tir.op.ptx_tcgen05_shift(a, 1), "a = Tx.handle()\nTx.ptx.tcgen05.shift(a, 1)")
    _assert_print(
        tir.op.ptx_tcgen05_ld(a, 0, shape="16x64b", num=1, row=0, col=0, pack=False),
        'a = Tx.handle()\nTx.ptx.tcgen05.ld(a, 0, 0, "16x64b", 1, Tx.bool(False), 0)',
    )
    _assert_print(
        tir.op.ptx_tcgen05_st(a, 0, shape="16x64b", num=1, row=0, col=0, unpack=False),
        'a = Tx.handle()\nTx.ptx.tcgen05.st(a, 0, 0, "16x64b", 1, Tx.bool(False), 0)',
    )
    _assert_print(tir.op.ptx_tcgen05_wait_ld(), "Tx.ptx.tcgen05.wait.ld()")
    _assert_print(tir.op.ptx_tcgen05_wait_st(), "Tx.ptx.tcgen05.wait.st()")
    _assert_print(
        tir.op.ptx_tcgen05_commit(a, 1, 0), "a = Tx.handle()\nTx.ptx.tcgen05.commit(a, 1, 0)"
    )
    _assert_print(
        tir.op.ptx_tcgen05_relinquish_alloc_permit(1), "Tx.ptx.tcgen05.relinquish_alloc_permit(1)"
    )


def test_printer_ptx_mbarrier():
    bar = tir.Var("bar", "handle")
    _assert_print(
        tir.op.ptx_mbarrier_init(bar, 32), "bar = Tx.handle()\nTx.ptx.mbarrier.init(bar, 32)"
    )
    _assert_print(tir.op.ptx_mbarrier_arrive(bar), "bar = Tx.handle()\nTx.ptx.mbarrier.arrive(bar)")
    _assert_print(
        tir.op.ptx_mbarrier_arrive_expect_tx(bar, 128),
        "bar = Tx.handle()\nTx.ptx.mbarrier.arrive.expect_tx(bar, 128)",
    )
    _assert_print(
        tir.op.ptx_mbarrier_try_wait(bar, 1), "bar = Tx.handle()\nTx.ptx.mbarrier.try_wait(bar, 1)"
    )
    _assert_print(tir.op.cuda_cluster_sync(), "Tx.cuda.cluster_sync()")


def test_printer_cuda_more():
    p = tir.Var("p", "handle")
    _assert_print(tir.op.cuda_thread_fence(), "Tx.cuda.thread_fence()")
    _assert_print(tir.op.cuda_warp_sync(), "Tx.cuda.warp_sync()")
    _assert_print(tir.op.cuda_cta_sync(), "Tx.cuda.cta_sync()")
    _assert_print(tir.op.cuda_grid_sync(), "Tx.cuda.grid_sync()")
    _assert_print(tir.op.cuda_cluster_sync(), "Tx.cuda.cluster_sync()")
    _assert_print(tir.op.cuda_syncthreads_and(1), "Tx.cuda.syncthreads_and(1)")
    _assert_print(tir.op.cuda_syncthreads_or(1), "Tx.cuda.syncthreads_or(1)")
    _assert_print(tir.op.cuda_nano_sleep(100), "Tx.cuda.nano_sleep(100)")
    _assert_print(
        tir.op.cuda_atomic_add(p, tir.IntImm("int32", 1)),
        "p = Tx.handle()\nTx.cuda.atomic_add(p, 1)",
    )
    _assert_print(tir.op.cuda_atomic_cas(p, 1, 2), "p = Tx.handle()\nTx.cuda.atomic_cas(p, 1, 2)")
    _assert_print(tir.op.cuda_ldg(p, "float32"), 'p = Tx.handle()\nTx.cuda.ldg(p, "float32")')
    _assert_print(
        tir.op.cuda_func_call("f", 1, source_code=""), 'Tx.cuda.func_call("f", 1, source_code="")'
    )


def test_printer_nvshmem_more():
    p = tir.Var("p", "handle")
    _assert_print(tir.op.nvshmem_my_pe(), "Tx.nvshmem.my_pe()")
    _assert_print(tir.op.nvshmem_n_pes(), "Tx.nvshmem.n_pes()")
    _assert_print(
        tir.op.nvshmem_signal_op(p, 1, "set", 0),
        'p = Tx.handle()\nTx.nvshmem.signal_op(p, 1, "set", 0)',
    )
    _assert_print(
        tir.op.nvshmem_wait_until(p, "eq", 0),
        'p = Tx.handle()\nTx.nvshmem.wait_until(p, "eq", 0, "uint64_t")',
    )
    _assert_print(tir.op.nvshmem_quiet(), "Tx.nvshmem.quiet()")
    _assert_print(tir.op.nvshmem_barrier_all(), "Tx.nvshmem.barrier_all()")
    _assert_print(
        tir.op.nvshmem_getmem_nbi(p, p, 16, 0),
        "p = Tx.handle()\nTx.nvshmem.getmem_nbi(p, p, 16, 0)",
    )
    _assert_print(
        tir.op.nvshmem_getmem_nbi_warp(p, p, 16, 0),
        "p = Tx.handle()\nTx.nvshmem.getmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi_block(p, p, 16, 0),
        "p = Tx.handle()\nTx.nvshmem.putmem_nbi.block(p, p, 16, 0)",
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi(p, p, 16, 0),
        "p = Tx.handle()\nTx.nvshmem.putmem_nbi(p, p, 16, 0)",
    )
    _assert_print(
        tir.op.nvshmem_putmem_nbi_warp(p, p, 16, 0),
        "p = Tx.handle()\nTx.nvshmem.putmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi(p, p, 16, p, 1, "set", 0),
        'p = Tx.handle()\nTx.nvshmem.putmem_signal_nbi(p, p, 16, p, 1, "set", 0)',
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi_warp(p, p, 16, p, 1, "set", 0),
        'p = Tx.handle()\nTx.nvshmem.putmem_signal_nbi.warp(p, p, 16, p, 1, "set", 0)',
    )
    _assert_print(
        tir.op.nvshmem_putmem_signal_nbi_block(p, p, 16, p, 1, "set", 0),
        'p = Tx.handle()\nTx.nvshmem.putmem_signal_nbi.block(p, p, 16, p, 1, "set", 0)',
    )


def test_printer_nki_namespace():
    A = tir.decl_buffer([1], dtype="float16", name="A")
    B = tir.decl_buffer([1], dtype="float16", name="B")
    a0 = A[0]
    b0 = B[0]
    _assert_print(
        tir.op.nki_load(a0, b0),
        'A = Tx.Buffer((1,), "float16")\nB = Tx.Buffer((1,), "float16")\nTx.nki.load(A, B)',
    )
    _assert_print(
        tir.op.nki_store(a0, b0),
        'A = Tx.Buffer((1,), "float16")\nB = Tx.Buffer((1,), "float16")\nTx.nki.store(A, B)',
    )
    _assert_print(
        tir.op.nki_tensor_copy(a0, b0),
        'A = Tx.Buffer((1,), "float16")\nB = Tx.Buffer((1,), "float16")\nTx.nki.tensor_copy(A, B)',
    )
    _assert_print(
        tir.op.nki_matmul(a0, a0, b0),
        'A = Tx.Buffer((1,), "float16")\n'
        'B = Tx.Buffer((1,), "float16")\n'
        "Tx.nki.matmul(A, A, B, Tx.bool(True))",
    )
    _assert_print(
        tir.op.nki_activation(a0, b0, "relu", 0.0, 1.0),
        'A = Tx.Buffer((1,), "float16")\n'
        'B = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.activation(A, B, "relu", Tx.float32(0.0), Tx.float32(1.0))',
    )
    _assert_print(
        tir.op.nki_memset(a0, 0),
        'A = Tx.Buffer((1,), "float16")\nTx.nki.memset(A, 0)',
    )
    _assert_print(
        tir.op.nki_identity(a0, 1),
        'A = Tx.Buffer((1,), "float16")\nTx.nki.identity(A, 1)',
    )
    _assert_print(
        tir.op.nki_reciprocal(a0, b0),
        'A = Tx.Buffer((1,), "float16")\nB = Tx.Buffer((1,), "float16")\nTx.nki.reciprocal(A, B)',
    )
    _assert_print(
        tir.op.nki_tensorreduce(a0, b0, "sum", False, 0),
        'A = Tx.Buffer((1,), "float16")\n'
        'B = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.tensorreduce(A, B, "sum", Tx.bool(False), 0)',
    )
    _assert_print(
        tir.op.nki_tensortensor(a0, a0, b0, "add"),
        'A = Tx.Buffer((1,), "float16")\n'
        'B = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.tensortensor(A, A, B, "add")',
    )
    _assert_print(
        tir.op.nki_tensorscalar(a0, a0, 1.0, "mul", False),
        'A = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.tensorscalar(A, A, Tx.float32(1.0), "mul", Tx.bool(False))',
    )
    _assert_print(
        tir.op.nki_tensorscalar_reduce(a0, a0, 1.0, "mul", "sum", False),
        'A = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.tensorscalar_reduce(A, A, Tx.float32(1.0), "mul", "sum", Tx.bool(False), Tx.bool(False))',  # noqa: E501
    )
    _assert_print(
        tir.op.nki_scalar_tensor_tensor(a0, a0, 1.0, a0, "add", "add"),
        'A = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.scalar_tensor_tensor(A, A, Tx.float32(1.0), A, "add", "add", Tx.bool(False), Tx.bool(False))',  # noqa: E501
    )
    _assert_print(
        tir.op.nki_scalar_tensor_scalar(a0, a0, 1.0, 1.0, "add", "add"),
        'A = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.scalar_tensor_scalar(A, A, Tx.float32(1.0), Tx.float32(1.0), "add", "add", Tx.bool(False), Tx.bool(False))',  # noqa: E501
    )
    _assert_print(
        tir.op.nki_activation_reduce(a0, a0, b0, "relu", "sum", 0.0, 1.0),
        'A = Tx.Buffer((1,), "float16")\n'
        'B = Tx.Buffer((1,), "float16")\n'
        'Tx.nki.activation_reduce(A, A, B, "relu", "sum", Tx.float32(0.0), Tx.float32(1.0))',
    )
    _assert_print(
        tir.op.nki_affine_select(a0, a0, a0, 1.0),
        'A = Tx.Buffer((1,), "float16")\nTx.nki.affine_select(A, A, A, Tx.float32(1.0))',
    )


def test_printer_ptx_mma_and_wgmma():
    r = tir.Var("r", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    tir.Var("b", "handle")
    _assert_print(
        tir.op.ptx_mma("m8n8k4", "row", "row", "fp16", "fp16", "fp16", "fp16", r, r, r, 0, False),
        'r = Tx.handle()\nTx.ptx.mma("void", "m8n8k4", "row", "row", "fp16", "fp16", "fp16", "fp16", r, r, r, 0, Tx.bool(False))',  # noqa: E501
    )
    _assert_print(
        tir.op.ptx_wgmma_encode_matrix_descriptor(d, a, 1, 1, 0),
        "d = Tx.handle()\na = Tx.handle()\nTx.ptx.wgmma.encode_matrix_descriptor(d, a, 1, 1, 0)",
    )
    _assert_print(tir.op.ptx_wgmma_noop_barrier(0), "Tx.ptx.wgmma.noop_barrier(0)")
    _assert_print(
        tir.op.ptx_wgmma_mma_async_ss(
            d,
            d,
            0,
            0,
            M=16,
            N=16,
            K=16,
            in_dtype="f16",
            out_dtype="f16",
            transA=True,
            transB=False,
            scaleA=1.0,
            scaleB=1.0,
            scaleD=True,
        ),
        'd = Tx.handle()\nTx.ptx.wgmma.mma_async.ss(16, 16, 16, "f16", "f16", Tx.bool(True), Tx.bool(False), Tx.float32(1.0), Tx.float32(1.0), Tx.bool(True), d, d, 0, 0)',  # noqa: E501
    )
    _assert_print(
        tir.op.ptx_wgmma_mma_async_rs(
            d,
            0,
            0,
            M=16,
            N=16,
            K=16,
            in_dtype="f16",
            out_dtype="f16",
            transA=True,
            transB=False,
            scaleA=1.0,
            scaleB=1.0,
            scaleD=True,
        ),
        'd = Tx.handle()\nTx.ptx.wgmma.mma_async.rs(16, 16, 16, "f16", "f16", Tx.bool(True), Tx.bool(False), Tx.float32(1.0), Tx.float32(1.0), Tx.bool(True), d, 0, 0)',  # noqa: E501
    )


def test_printer_ptx_cp_async_tensor():
    tmap = tir.Var("tm", "handle")
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_global_to_cluster(2, tmap, 0, tmap, 0, 1, "", 0, 1, ""),
        "tm = Tx.handle()\n"
        'Tx.ptx.cp_async.bulk.tensor.g2c(2, tm, 0, tm, 0, 1, Tx.uint64(0), 0, 0, 1, "")',
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(
            2, tmap, 0, tmap, 0, 1, "", 0, 1, ""
        ),
        "tm = Tx.handle()\n"
        "Tx.ptx.cp_async.bulk.tensor.g2c_tile_gather4"
        '(2, tm, 0, tm, 0, 1, Tx.uint64(0), 0, 0, 1, "")',
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(2, tmap, "", 0, 0, ""),
        "tm = Tx.handle()\n"
        'Tx.ptx.cp_async.bulk.tensor.g2c_prefetch(2, tm, Tx.uint64(0), 0, 0, 0, "")',
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_shared_to_global(2, 0, tmap, "", 0, 0, ""),
        'tm = Tx.handle()\nTx.ptx.cp_async.bulk.tensor.s2g(2, 0, tm, Tx.uint64(0), 0, 0, 0, "")',
    )
    _assert_print(
        tir.op.ptx_cp_async_bulk_tensor_shared_to_global_reduce(2, 0, tmap, "", "add", 0, 0, ""),
        "tm = Tx.handle()\n"
        "Tx.ptx.cp_async.bulk.tensor.s2g_reduce"
        '(2, 0, tm, Tx.uint64(0), 0, "add", 0, 0, "")',
    )


def test_printer_ptx_cp_async_call():
    sh = tir.Var("sh", "handle")
    gl = tir.Var("gl", "handle")
    _assert_print(
        tir.op.ptx_cp_async(
            sh, gl, 16, cache_hint="", prefetch_size=-1, predicate=-1, fill_mode=""
        ),
        "sh = Tx.handle()\ngl = Tx.handle()\n"
        'Tx.ptx.cp_async("void", sh, gl, 16, Tx.uint64(0), 0, -1, -1, "")',
    )
