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
from tvm import te
import numpy as np
from tvm.topi.testing import conv2d_nhwc_python
import tvm.testing

VERIFY = True


def intrin_wmma_load_matrix(shape, scope):
    n, m, l = shape
    if scope == "wmma.matrix_a":
        row, col = n, l
    elif scope == "wmma.matrix_b":
        row, col = l, m
    A = te.placeholder((row, col), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=row * col
    )
    C = te.compute((row, col), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=row * col
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                m,
                l,
                BC.elem_offset // (row * col),
                BA.access_ptr("r"),
                col,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(shape):
    n, m, l = shape
    A = te.placeholder((n, l), name="A", dtype="float16")
    B = te.placeholder((l, m), name="B", dtype="float16")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute(
        (n, m),
        lambda ii, jj: te.sum(A[ii, k].astype("float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=n * l
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=l * m
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=n * m,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    n,
                    m,
                    l,
                    BC.elem_offset // (n * m),
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // (n * m),
                    BA.data,
                    BA.elem_offset // (n * l),
                    BB.data,
                    BB.elem_offset // (l * m),
                    BC.data,
                    BC.elem_offset // (n * m),
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(shape):
    n, m, l = shape
    A = te.placeholder((n, m), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=n * m
    )
    C = te.compute((n, m), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=n * m
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                m,
                l,
                BA.elem_offset // (n * m),
                BC.access_ptr("w"),
                m,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


@tvm.testing.requires_tensorcore
def test_tensor_core_batch_matmal():
    batch_size = 4
    n = 512
    m, l = n, n
    assert n % 32 == 0
    assert m % 8 == 0
    assert l % 16 == 0
    nn, mm, ll = n // 32, m // 8, l // 16
    A = te.placeholder((batch_size, nn, ll, 32, 16), name="A", dtype="float16")
    B = te.placeholder((batch_size, ll, mm, 16, 8), name="B", dtype="float16")
    k1 = te.reduce_axis((0, ll), name="k1")
    k2 = te.reduce_axis((0, 16), name="k2")
    C = te.compute(
        (batch_size, nn, mm, 32, 8),
        lambda b, i, j, ii, jj: te.sum(
            A[b, i, k1, ii, k2].astype("float") * B[b, k1, j, k2, jj].astype("float"), axis=[k1, k2]
        ),
        name="Fragment_C",
    )
    s = te.create_schedule(C.op)

    warp_size = 32
    kernel_size = 16
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 4
    warp_col_tiles = 2
    chunk = 4

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")

    b, i, j, kernel_i, kernel_j = s[C].op.axis
    i, ii = s[C].split(i, factor=warp_row_tiles)
    block_i, i = s[C].split(i, factor=block_row_warps)
    j, jj = s[C].split(j, factor=warp_col_tiles)
    block_j, j = s[C].split(j, factor=block_col_warps)
    s[C].reorder(block_i, block_j, i, j, ii, jj, kernel_i, kernel_j)
    s[C].bind(b, block_z)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(i, thread_y)
    s[C].bind(j, thread_z)

    s[CF].compute_at(s[C], j)
    b, warp_i, warp_j, _i, _j = s[CF].op.axis
    k, _k = CF.op.reduce_axis
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _i, _j, _k)

    s[AF].compute_at(s[CF], ki)
    s[BF].compute_at(s[CF], ki)

    s[AS].compute_at(s[CF], ko)
    b, xo, yo, xi, yi = AS.op.axis
    tx, xo = s[AS].split(xo, nparts=block_row_warps)
    ty, yo = s[AS].split(yo, nparts=block_col_warps)
    t = s[AS].fuse(xi, yi)
    to, ti = s[AS].split(t, nparts=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)

    s[BS].compute_at(s[CF], ko)
    b, xo, yo, xi, yi = BS.op.axis
    tx, xo = s[BS].split(xo, nparts=block_row_warps)
    ty, yo = s[BS].split(yo, nparts=block_col_warps)
    t = s[BS].fuse(xi, yi)
    to, ti = s[BS].split(t, nparts=warp_size)
    s[BS].bind(tx, thread_y)
    s[BS].bind(ty, thread_z)
    s[BS].bind(to, thread_x)

    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix((32, 8, 16), "wmma.matrix_a"))
    s[BF].tensorize(BF.op.axis[-2], intrin_wmma_load_matrix((32, 8, 16), "wmma.matrix_b"))
    s[C].tensorize(kernel_i, intrin_wmma_store_matrix((32, 8, 16)))
    s[CF].tensorize(_i, intrin_wmma_gemm((32, 8, 16)))

    func = tvm.build(s, [A, B, C], "cuda")

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=(batch_size, nn, ll, 32, 16)).astype(A.dtype)
    b_np = np.random.uniform(size=(batch_size, ll, mm, 16, 8)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((batch_size, nn, mm, 32, 8), dtype=C.dtype), dev)
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=3)
    print("gemm with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

    if VERIFY:
        func(a, b, c)
        a_np = a_np.transpose((0, 1, 3, 2, 4)).reshape(batch_size, n, n)
        b_np = b_np.transpose((0, 1, 3, 2, 4)).reshape(batch_size, n, n)
        c_np = c.numpy().transpose((0, 1, 3, 2, 4)).reshape(batch_size, n, n)
        np.testing.assert_allclose(
            c_np, np.matmul(a_np.astype(C.dtype), b_np.astype(C.dtype)), rtol=1e-4, atol=1e-4
        )


@tvm.testing.requires_tensorcore
def test_tensor_core_batch_conv():
    # The sizes of inputs and filters
    batch_size = 32
    height = 14
    width = 14
    in_channels = 32
    out_channels = 64
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    block_size = 16

    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 4
    warp_col_tiles = 2
    warp_size = 32
    chunk = 2

    # Input feature map: (N, H, W, IC, n, ic)
    data_shape = (
        batch_size // block_size,
        height,
        width,
        in_channels // block_size,
        block_size,
        block_size,
    )
    # Kernel: (H, W, IC, OC, ic, oc)
    kernel_shape = (
        kernel_h,
        kernel_w,
        in_channels // block_size,
        out_channels // block_size,
        block_size,
        block_size,
    )

    # Output feature map: (N, H, W, OC, n, oc)
    output_shape = (
        batch_size // block_size,
        height,
        width,
        out_channels // block_size,
        block_size,
        block_size,
    )

    assert batch_size % block_size == 0
    assert in_channels % block_size == 0
    assert out_channels % block_size == 0

    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    ic = te.reduce_axis((0, in_channels // block_size), name="ic")
    ii = te.reduce_axis((0, block_size), name="ii")

    # Algorithm
    A = te.placeholder(data_shape, name="A", dtype="float16")
    W = te.placeholder(kernel_shape, name="W", dtype="float16")
    Apad = te.compute(
        (
            batch_size // block_size,
            height + 2 * pad_h,
            width + 2 * pad_w,
            in_channels // block_size,
            block_size,
            block_size,
        ),
        lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
            tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
            A[n, h - pad_h, w - pad_w, i, nn, ii],
            tvm.tir.const(0.0, "float16"),
        ),
        name="Apad",
    )
    Conv = te.compute(
        output_shape,
        lambda n, h, w, o, nn, oo: te.sum(
            Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float32")
            * W[kh, kw, ic, o, ii, oo].astype("float32"),
            axis=[ic, kh, kw, ii],
        ),
        name="Conv",
    )

    s = te.create_schedule(Conv.op)
    s[Apad].compute_inline()

    AS = s.cache_read(Apad, "shared", [Conv])
    WS = s.cache_read(W, "shared", [Conv])
    AF = s.cache_read(AS, "wmma.matrix_a", [Conv])
    WF = s.cache_read(WS, "wmma.matrix_b", [Conv])
    ConvF = s.cache_write(Conv, "wmma.accumulator")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    nc, hc, wc, oc, nnc, ooc = Conv.op.axis
    block_k = s[Conv].fuse(hc, wc)
    s[Conv].bind(block_k, block_z)
    nc, nci = s[Conv].split(nc, factor=warp_row_tiles)
    block_i, nc = s[Conv].split(nc, factor=block_row_warps)
    oc, oci = s[Conv].split(oc, factor=warp_col_tiles)
    block_j, oc = s[Conv].split(oc, factor=block_col_warps)
    s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    s[Conv].bind(block_i, block_x)
    s[Conv].bind(block_j, block_y)
    s[Conv].bind(nc, thread_y)
    s[Conv].bind(oc, thread_z)

    s[ConvF].compute_at(s[Conv], oc)
    n, h, w, o, nnf, oof = ConvF.op.axis
    ko, ki = s[ConvF].split(ic, factor=chunk)
    s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

    s[AF].compute_at(s[ConvF], kw)
    s[WF].compute_at(s[ConvF], kw)

    s[WS].compute_at(s[ConvF], kh)
    s[AS].compute_at(s[ConvF], kh)

    n, h, w, i, nn, ii = AS.op.axis
    tx, xo = s[AS].split(n, nparts=block_row_warps)
    ty, yo = s[AS].split(xo, nparts=block_col_warps)
    t = s[AS].fuse(nn, ii)
    to, ti = s[AS].split(t, factor=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(ti, thread_x)

    kh, kw, ic, o, ii, oo = WS.op.axis
    tx, xo = s[WS].split(o, nparts=block_row_warps)
    ty, yo = s[WS].split(xo, nparts=block_col_warps)
    t = s[WS].fuse(ii, oo)
    to, ti = s[WS].split(t, nparts=warp_size)
    s[WS].bind(tx, thread_y)
    s[WS].bind(ty, thread_z)
    s[WS].bind(to, thread_x)
    s[WS].vectorize(ti)

    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix((16, 16, 16), "wmma.matrix_a"))
    s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix((16, 16, 16), "wmma.matrix_b"))
    s[Conv].tensorize(nnc, intrin_wmma_store_matrix((16, 16, 16)))
    s[ConvF].tensorize(nnf, intrin_wmma_gemm((16, 16, 16)))

    func = tvm.build(s, [A, W, Conv], "cuda")

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=data_shape).astype(A.dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(W.dtype)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    c = tvm.nd.array(np.zeros(output_shape, dtype=Conv.dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=3)
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))

    if VERIFY:
        func(a, w, c)
        a_np = a_np.transpose(0, 4, 1, 2, 3, 5).reshape(batch_size, height, width, in_channels)
        w_np = w_np.transpose(0, 1, 2, 4, 3, 5).reshape(
            kernel_h, kernel_w, in_channels, out_channels
        )
        c_np = (
            c.numpy().transpose((0, 4, 1, 2, 3, 5)).reshape(batch_size, height, width, out_channels)
        )
        c_std = conv2d_nhwc_python(
            a_np.astype(Conv.dtype), w_np.astype(Conv.dtype), (stride_h, stride_w), (pad_h, pad_w)
        ).astype(Conv.dtype)
        np.testing.assert_allclose(c_np, c_std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_tensor_core_batch_matmal()
    test_tensor_core_batch_conv()
