# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm import relay


def intrin_mfma_load_matrix(shape, matrix, thread=None, strides=None):
    M, N, K = shape
    if matrix == "A":
        row, col = M, K
        src_strides, dst_strides = strides, None
    elif matrix == "B":
        row, col = K, N
        src_strides, dst_strides = strides, None
    output_shape = (row, col)

    A = te.placeholder(output_shape, name=matrix, dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="global", offset_factor=1, strides=src_strides)

    C = te.compute(output_shape, lambda i, j: A[i, j], name="C")

    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="local", offset_factor=1, strides=dst_strides)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]

        tx = thread
        if tx == None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)

        blk_td = tx % 16
        offset = tx // 16
        # TODO(csullivan): Using offset works, but using tx directly does not, fix this
        if matrix == "A":
            for blk_id in range(0, 4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_td, blk_id * 4 + offset], "float16")))
        elif matrix == "B":
            for blk_id in range(0, 4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_id * 4 + offset, blk_td], "float16")))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_mfma_store_matrix(shape, thread=None, strides=None):
    M, N, K = shape
    A = te.placeholder((M, N), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="local", offset_factor=1)
    C = te.compute((M, N), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", offset_factor=1, strides=strides)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        tx = thread
        if tx == None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)
        vec_width = 4
        blk_id = tx // 16
        blk_td = tx % 16

        # TODO(csullivan): Consider TVM change to BufferVar.__getitem__
        # to convert int to const for quality of life when using vector types.
        ib.emit(BC.vstore([blk_td, blk_id * vec_width], BA.vload([0, 0], "float32x4")))

        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_mfma_gemm(shape, dtype, input_scope):
    M, N, K = shape

    # TODO(csullivan):Replace below with a function to get the correct
    # llvm function based on shape and type
    assert M == 16
    assert N == 16
    assert K == 16
    mfma_instr_name = "llvm.amdgcn.mfma.f32.16x16x16f16"
    llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(mfma_instr_name)
    assert llvm_id != 0

    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype("float") * B[k, j].astype("float"), axis=[k]),
        name="C",
    )

    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", scope=input_scope, offset_factor=1)
    Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", scope=input_scope, offset_factor=1)
    Cb = tvm.tir.decl_buffer(C.shape, C.dtype, name="Cb", scope="local")

    def intrin_func(ins, outs):
        Ab, Bb = ins
        (Cb,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(Cb.vstore([0, 0], tvm.tir.const(0, "float32x4")))
            return ib.get()

        def update():
            # Each thread is responsible for 4 values along the reduction axis for a single (m, n) pixel
            ib = tvm.tir.ir_builder.create()
            a_vec = Ab.vload([0, 0], "float16x4")
            b_vec = Bb.vload([0, 0], "float16x4")
            c_vec = Cb.vload([0, 0], "float32x4")
            args_6 = tvm.tir.const(6, "uint32")
            # Transpose inputs (equivalent to switching the order in this packed layout)
            # in order to ensure row-major order on the output with coalesced vector writes.
            gemm = tvm.tir.call_llvm_pure_intrin(
                "float32x4", mfma_instr_name, args_6, b_vec, a_vec, c_vec, 0, 0, 0
            )
            ib.emit(Cb.vstore([0, 0], gemm))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})


def schedule(s, A, B, C):
    warp_size = 64
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 16
    warp_col_tiles = 16
    chunk = 4

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")

    AF = s.cache_read(A, "local", [C])
    BF = s.cache_read(B, "local", [C])
    CF = s.cache_write(C, "local")

    b, i, j = s[C].op.axis
    i, ii = s[C].split(i, factor=warp_row_tiles)
    block_i, i = s[C].split(i, factor=block_row_warps)
    j, jj = s[C].split(j, factor=warp_col_tiles)
    block_j, j = s[C].split(j, factor=block_col_warps)
    s[C].reorder(block_i, block_j, i, j, ii, jj)
    s[C].bind(b, block_z)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)

    s[CF].compute_at(s[C], j)
    b, _i, _j = s[CF].op.axis
    (_k,) = s[CF].op.reduce_axis
    _ko, _ki = s[CF].split(_k, factor=warp_col_tiles)
    s[CF].reorder(_ko, _i, _j, _ki)

    s[AF].compute_at(s[CF], _ko)
    ba, _m, _ka = AF.op.axis
    _kao, _kai = s[AF].split(_ka, factor=warp_col_tiles)
    s[AF].reorder(ba, _kao, _m, _kai)

    s[BF].compute_at(s[CF], _ko)
    bb, _kb, _n = BF.op.axis
    _kbo, _kbi = s[BF].split(_kb, factor=warp_row_tiles)
    s[BF].reorder(bb, _kbo, _n, _kbi)

    s[AF].tensorize(_m, intrin_mfma_load_matrix((16, 16, 16), "A", strides=(A.shape[2], 1)))
    s[BF].tensorize(_n, intrin_mfma_load_matrix((16, 16, 16), "B", strides=(B.shape[2], 1)))
    s[CF].tensorize(_i, intrin_mfma_gemm(shape=(16, 16, 16), dtype="float16", input_scope="local"))
    s[C].tensorize(ii, intrin_mfma_store_matrix(shape=(16, 16, 16), strides=(B.shape[2], 1)))

    return s


def test_amd_xdlops_tensorization_explicit_schedule(target="rocm -mcpu=gfx908"):
    batch_size, m, n, k = 32, 32, 32, 32
    assert m % 16 == 0
    assert n % 16 == 0
    assert k % 16 == 0
    mm, nn, kk = m, n, k
    A = te.placeholder((batch_size, mm, kk), name="A", dtype="float16")
    B = te.placeholder((batch_size, kk, nn), name="B", dtype="float16")
    k1 = te.reduce_axis((0, kk), name="k1")
    C = te.compute(
        (batch_size, mm, nn),
        lambda b, i, j: te.sum(
            A[b, i, k1].astype("float") * B[b, k1, j].astype("float"), axis=[k1]
        ),
        name="Fragment_C",
    )
    s = te.create_schedule(C.op)
    s = schedule(s, A, B, C)

    func = tvm.build(s, [A, B, C], target, name="intrinsic")
    dev = tvm.rocm(0)
    a_np = np.random.uniform(size=(batch_size, mm, kk)).astype(A.dtype)
    b_np = np.random.uniform(size=(batch_size, kk, nn)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((batch_size, mm, nn), dtype=C.dtype), dev)
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=3)

    func(a, b, c)
    c_np = c.asnumpy()
    np_result = np.matmul(a_np.astype(C.dtype), b_np.astype(C.dtype))
    np.testing.assert_allclose(c_np, np_result, rtol=1e-4, atol=1e-4)


def run_relay_func(func, inputs, target):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(func, target)

    from tvm.contrib import graph_executor

    dev = tvm.rocm(0)
    m = graph_executor.create(graph, lib, dev)
    for (name, data) in inputs.items():
        m.set_input(name, tvm.nd.array(data))

    m.run()
    tvm_output = m.get_output(0)
    return tvm_output.asnumpy()


def test_amd_xdlops_tensorization_batch_matmul(target):
    batch_size, m, n, k = 32, 32, 32, 32
    A = relay.var("A", shape=(batch_size, m, k), dtype="float32")
    B = relay.var("B", shape=(batch_size, n, k), dtype="float32")
    C = relay.nn.batch_matmul(A, B)
    f = relay.Function(relay.analysis.free_vars(C), C)
    a_np = np.random.uniform(size=(batch_size, m, k)).astype("float32")
    b_np = np.random.uniform(size=(batch_size, n, k)).astype("float32")

    c_np = run_relay_func(f, {"A": a_np, "B": b_np}, target)
    np_result = np.matmul(a_np.astype("float16"), b_np.transpose((0, 2, 1)).astype("float16"))
    np.testing.assert_allclose(c_np, np_result, rtol=1e-3, atol=1e-3)


def test_amd_xdlops_tensorization_dense(target):
    batch, inp, out = 32, 32, 32
    A = relay.var("A", shape=(batch, inp), dtype="float32")
    B = relay.var("B", shape=(out, inp), dtype="float32")
    C = relay.nn.dense(A, B)
    f = relay.Function(relay.analysis.free_vars(C), C)
    a_np = np.random.uniform(size=(batch, inp)).astype("float32")
    b_np = np.random.uniform(size=(out, inp)).astype("float32")

    c_np = run_relay_func(f, {"A": a_np, "B": b_np}, target)
    np_result = np.matmul(a_np.astype("float16"), b_np.transpose((1, 0)).astype("float16"))
    np.testing.assert_allclose(c_np, np_result, rtol=1e-3, atol=1e-3)


def test_amd_xdlops():
    target = "rocm -mcpu=gfx908"
    if not tvm.testing.device_enabled(target):
        print("skip because %s is not enabled.." % target)
        return
    test_amd_xdlops_tensorization_explicit_schedule(target=target)
    test_amd_xdlops_tensorization_batch_matmul(target=target)
    test_amd_xdlops_tensorization_dense(target=target)


if __name__ == "__main__":
    test_amd_xdlops()
