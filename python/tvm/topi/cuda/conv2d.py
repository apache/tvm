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
# pylint: disable=invalid-name, unused-argument
"""Compute definition for conv2d with cuda backend"""
import tvm
from tvm.tir.schedule import BlockRV, Schedule
from tvm._ffi import register_func
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import OtherOptionEntity
from tvm.contrib import cudnn
from tvm.tir.tensor_intrin.cuda import (
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_SHARED_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
)


from .. import nn, generic
from ..nn.utils import get_pad_tuple, get_output_shape
from ..nn.pad import pad
from ..utils import get_const_tuple, traverse_inline
from .conv2d_direct import schedule_direct_cuda
from ..transform import reshape


@autotvm.register_topi_compute("conv2d_nchw.cuda")
def conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d with NCHW layout"""
    return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nchw.cuda")
def schedule_conv2d_nchw(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_nchw":
            schedule_direct_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_cudnn.cuda")
def conv2d_cudnn(
    cfg, data, kernel, strides, padding, dilation, groups=1, layout="NCHW", out_dtype="float32"
):
    """Compute conv2d using CuDNN library"""
    if layout == "NCHW":
        tensor_format = 0  # CUDNN_TENSOR_NCHW
        N, _, H, W = get_const_tuple(data.shape)
    elif layout == "NHWC":
        tensor_format = 1  # CUDNN_TENSOR_NHWC
        N, H, W, _ = get_const_tuple(data.shape)
    else:
        raise ValueError(f"Unsupported layout {layout} in cudnn")
    CO, CI, KH, KW = get_const_tuple(kernel.shape)

    # handle dilation
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (KH - 1) * dilation_h + 1
    KW_dilated = (KW - 1) * dilation_h + 1

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    if (pt != pb) or (pl != pr):
        raise ValueError("Cudnn doesn't support asymmetric padding.")

    OH = (H + pt + pb - KH) // stride_h + 1
    OW = (W + pl + pr - KW) // stride_w + 1

    if isinstance(N, int):
        cfg.add_flop(
            groups
            * 2
            * N
            * OH
            * OW
            * CO
            * CI
            * ((KH - 1) * dilation_h + 1)
            * ((KW - 1) * dilation_w + 1)
        )

    if data.dtype == "int8" or kernel.dtype == "int8":
        if layout == "NCHW":
            raise ValueError("NCHW layout do not support int8 in cudnn")
        dtype = "int32"
    else:
        dtype = data.dtype

    cfg.define_knob("algo", range(cudnn.algo_to_index("fwd", "CUDNN_CONVOLUTION_FWD_ALGO_COUNT")))
    if cfg.is_fallback:
        if cudnn.exists():
            # Let CUDNN choose the best algo, based on benchmarks run
            # on the local machine.  In the future, this should be
            # based on parameters stored in the Target.
            cfg["algo"] = OtherOptionEntity(-1)
        else:
            cfg["algo"] = OtherOptionEntity(0)

    return cudnn.conv_forward(
        data,
        kernel,
        [pt, pl],  # cudnn padding pt, pl on both sides of input
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        algo=cfg["algo"].val,
        conv_dtype=dtype,
        groups=groups,
    )


@autotvm.register_topi_schedule("conv2d_cudnn.cuda")
def schedule_conv2d_cudnn(cfg, outs):
    """Create the schedule for conv2d_cudnn"""
    return generic.schedule_extern(outs)


def conv2d_backward_weight_cudnn(
    dy, x, kernel_size, padding, stride, dilation, groups, layout, output_dtype
):
    """Compute conv2d wgrad using CuDNN library"""
    assert layout in ["NCHW", "NHWC"]

    if dy.dtype == "float16":
        # cuDNN does not seem to support other combination.
        assert output_dtype == "float16", "Only supports fp16 output for cuDNN fp16 wgrad."

    conv_dtype = "float32"  # Accumulation is always fp32
    return cudnn.conv_backward_filter(
        dy,
        x,
        kernel_size,
        padding,
        stride,
        dilation,
        conv_mode=1,
        tensor_format=0 if layout == "NCHW" else 1,
        conv_dtype=conv_dtype,
        groups=groups,
    )


@autotvm.register_topi_compute("conv2d_nchw_mma.cuda")
def conv2d_nchw_mma(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d nchw using im2col"""
    assert data.dtype == "float16"
    out_channels, in_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert dilation_h == 1 and dilation_w == 1

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    batch_size, _, P, Q = get_output_shape(
        data, kernel, stride_h, stride_w, dilation_h, dilation_w, padding
    )
    assert batch_size == 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]

    if all([v == 0 for v in pad_before]) and all([v == 0 for v in pad_after]):
        pad_data = data
    else:
        pad_data = pad(data, pad_before, pad_after, name="pad_data")

    M = out_channels
    K = in_channels * kernel_h * kernel_w
    N = batch_size * P * Q

    if kernel_h * kernel_w == 1:
        ck = te.reduce_axis((0, K), name="k")
        A = reshape(kernel, (M, K))
        B = reshape(pad_data, (K, N))
        C = te.compute(
            (batch_size, out_channels, P, Q),
            lambda b, o, h, w: te.sum(
                A[o, ck].astype(out_dtype) * B[ck, h * Q + w].astype(out_dtype),
                axis=[ck],
            ),
            name="conv2d_nchw_mma",
            attrs={
                "schedule_rule": "conv2d_nchw_mma",
            },
        )
    else:
        # Convert the kernel of (O,I,H,W) to (N,K) format i.e (OC,IC*KH*KW)
        A = te.compute(
            (M, K),
            lambda x, y: kernel[
                x, (y // (kernel_h * kernel_w)), (y // kernel_w) % kernel_h, y % kernel_w
            ],
            name="T_reshape",
        )

        # Convert the data of (N,C,H,W) to (K,M) format i.e (IC*KH*KW,OH*OW)
        B = te.compute(
            (K, N),
            lambda y, x: pad_data[
                0,
                y // (kernel_h * kernel_w),
                stride_h * (x // Q) + ((y // kernel_w) % kernel_h),
                stride_w * (x % Q) + y % kernel_w,
            ],
            name="T_reshape_1",
        )

        # Apply GEMM operation. The result will be of (N,O,H,W) format
        ck = te.reduce_axis((0, K), name="k")
        C = te.compute(
            (batch_size, out_channels, P, Q),
            lambda b, o, h, w: te.sum(
                A[o, ck].astype(out_dtype) * B[ck, h * Q + w].astype(out_dtype),
                axis=[ck],
            ),
            name="conv2d_nchw_mma",
            attrs={
                "schedule_rule": "conv2d_nchw_mma",
            },
        )
    return C


def schedule_rule_conv2d_nchw_mma(sch: Schedule, block: BlockRV):
    """Create the schedule for conv2d nchw im2col"""
    k_inner = 16

    target = tvm.target.Target.current(allow_none=False)

    i_factors = []
    j_factors = []
    k_factors = []

    # comment out the following line enables sampling
    # i_factors, j_factors, k_factors = [16, 8, 2, 1, 1], [1, 32, 2, 4, 1], [64, 4, 1]
    do_sample = False
    if len(i_factors) == 0 and len(j_factors) == 0 and len(k_factors) == 0:
        do_sample = True

    shared_scope = "shared"
    # TODO(Mei Ye): Default warp size is 64 for Navi3, use a non-default warp size of 32
    # can get much better performance.
    warp_size = target.thread_warp_size
    vector_size = 4
    b_transposed = False
    block = sch.get_block("conv2d_nchw_mma")
    write_buf = sch.get_sref(block).stmt.writes
    output_type = write_buf[0].buffer.dtype

    if output_type == "float32":
        out_bytes_per_ele = 4
    else:
        out_bytes_per_ele = 2

    read_buf = sch.get_sref(sch.get_block("T_reshape_1")).stmt.reads
    in_channels = read_buf[0].buffer.shape[-3]

    data_type = read_buf[0].buffer.dtype
    bytes_per_ele = 4

    if data_type == "float16":
        bytes_per_ele = 2
    else:
        raise ValueError("Unsupported data type:", data_type)

    read_buf = sch.get_sref(sch.get_block("T_reshape")).stmt.reads
    kernel_height = read_buf[0].buffer.shape[-2]
    kernel_width = read_buf[0].buffer.shape[-1]
    k_dim = in_channels * kernel_height * kernel_width

    loops = sch.get_loops(block)
    i3_extent = sch.get_sref(loops[-2]).stmt.extent
    i1_extent = sch.get_sref(loops[-4]).stmt.extent

    sch.transform_block_layout(
        block, lambda i0, i1, i2, i3, i4: ((i0 * i1_extent + i1), (i2 * i3_extent + i3), i4)
    )
    block1 = sch.reindex(block=block, buffer=("write", 0))

    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, k_inner])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)
    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    max_local = 65536
    tile_row = 16
    tile_col = 16
    max_num_threads = target.max_num_threads
    max_shared_mem = target.max_shared_memory_per_block
    max_block_size_x = target.max_block_size_x
    max_block_size_y = target.max_block_size_y

    if do_sample:
        while True:
            # sample i-axis
            factors = sch.sample_perfect_tile(i, n=5)
            i_factors = [sch.get(e) for e in factors]

            # Local memory (register) constraint
            j3_j4_max = (
                (max_local - i_factors[3] * i_factors[4] * tile_row * tile_col)
                // (tile_row * tile_col)
                // (1 + i_factors[3] * i_factors[4])
            )

            # max_num_threads constraint
            j2_max = max_num_threads // warp_size // i_factors[2]

            # max_block_size constraint
            j0_max = max_block_size_x // i_factors[0]
            j1_max = max_block_size_y // i_factors[1]

            factors = sch.sample_perfect_tile(j, n=5)
            j_factors = [sch.get(e) for e in factors]

            if (
                j_factors[0] > j0_max
                or j_factors[1] > j1_max
                or j_factors[2] > j2_max
                or j_factors[3] * j_factors[4] > j3_j4_max
            ):
                continue

            # Calculate the shared mem required for the staging buffer.
            # In the compact_buffer pass, the size of the final buffer
            # will be determined based on below calculation
            # Buffer[(i2 - 1) * (tile_row * i3 * i4) + tile_row, (j2 - 1)
            # * (tile_col * j3 * j4) + tile_col]
            x_dim = (i_factors[2] - 1) * (tile_row * i_factors[3] * i_factors[4]) + tile_row
            y_dim = (j_factors[2] - 1) * (tile_col * j_factors[3] * j_factors[4]) + tile_col
            max_out_shared_mem = x_dim * y_dim * out_bytes_per_ele

            # calculate the remaining shared memory after allocating staging buffer
            rem_shared_mem = max_shared_mem - max_out_shared_mem
            if rem_shared_mem <= 0:
                continue

            # max_shared_memory_per_block constraint
            matrix_A_and_B_shared_mem = (
                i_factors[2] * i_factors[3] * i_factors[4] * tile_row
                + j_factors[2] * j_factors[3] * j_factors[4] * tile_col
            ) * bytes_per_ele

            k0_min = (k_dim * matrix_A_and_B_shared_mem + rem_shared_mem - 1) // rem_shared_mem
            k1_k2_max = tile_row * tile_col // k0_min

            factors = sch.sample_perfect_tile(k, n=3)
            k_factors = [sch.get(e) for e in factors]

            if k_factors[0] >= k0_min and k_factors[1] * k_factors[2] <= k1_k2_max:
                break

    num_ty = i_factors[2] * j_factors[2]
    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(i2, j2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, shared_scope)
        sch.compute_at(block_read, k0)
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        if idx == 0:
            # Calculate the size of data that will be copied into the shared buffer for buffer 0
            copy_size = i_factors[3] * i_factors[4] * 16 * k_factors[1] * k_factors[2] * k_inner
        else:
            # Calculate the size of data that will be copied into the shared buffer for buffer 1
            copy_size = j_factors[3] * j_factors[4] * 16 * k_factors[1] * k_factors[2] * k_inner

        # Instead of interleaving, copy the data buffer into the shared buffer in contiguous blocks.
        # By doing this, the compiler will generate ds_load_b128 instructions instead of the default
        # ds_load_u16/ds_load_u64, improving performance.
        factors = num_ty * warp_size * vector_size
        loop_size = copy_size // factors

        if loop_size > 1:
            _, f_1, f_2, f_3, f_4 = sch.split(
                fused, factors=[None, num_ty, warp_size, loop_size, vector_size]
            )
            sch.vectorize(f_4)
        else:
            # Once the copying has been divided into threads in x
            # and y dimensions, see if the remaining buffer can be vectorized.
            v_size = copy_size // (num_ty * warp_size)
            if v_size >= vector_size:
                _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
                sch.vectorize(f_3)
            else:
                # Do not vectorize because the amount of space left over
                # after threading the copy is less than vector_size.
                _, f_1, f_2 = sch.split(fused, factors=[None, num_ty, warp_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")

        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    c_warp_scope = "wmma.accumulator"
    a_warp_scope = "wmma.matrix_a"
    b_warp_scope = "wmma.matrix_b"

    A_warp = sch.cache_read(block_outer, 0, a_warp_scope)
    B_warp = sch.cache_read(block_outer, 1, b_warp_scope)

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    C_warp = sch.cache_write(block_outer, 0, c_warp_scope)
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_block("conv2d_nchw_mma_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)

    sch.reverse_compute_at(block1, jo)
    fused = sch.fuse(*sch.get_loops(block1)[-2:])
    _, f_2, f_3 = sch.split(fused, factors=[None, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.vectorize(f_3)

    sch.set_scope(
        sch.get_block("conv2d_nchw_mma_reindex_wmma.accumulator"),
        buffer_index=0,
        storage_scope="shared",
    )

    sch.tensorize(loop_a, WMMA_LOAD_16x16x16_F16_A_INTRIN)
    sch.tensorize(loop_b, WMMA_LOAD_16x16x16_F16_B_INTRIN)

    intrin = WMMA_SYNC_16x16x16_f16f16f32_INTRIN
    if output_type == "float16":
        intrin = WMMA_SYNC_16x16x16_f16f16f16_INTRIN
    sch.tensorize(sch.get_loops(block_inner)[-3], intrin)

    intrin = WMMA_FILL_16x16x16_F32_INTRIN
    if output_type == "float16":
        intrin = WMMA_FILL_16x16x16_F16_INTRIN
    sch.tensorize(sch.get_loops(block_init_c)[-2], intrin)

    intrin = WMMA_STORE_16x16x16_F32_SHARED_INTRIN
    if output_type == "float16":
        intrin = WMMA_STORE_16x16x16_F16_SHARED_INTRIN
    sch.tensorize(sch.get_loops(C_warp)[-2], intrin)

    # print(sch.mod.script())

    return [sch]


register_func("meta_schedule.vulkan.conv2d_nchw_mma", schedule_rule_conv2d_nchw_mma)
