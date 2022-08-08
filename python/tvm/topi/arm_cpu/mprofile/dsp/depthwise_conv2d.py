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

"""Direct implementation of conv2d."""

from tvm import autotvm
from tvm.autotvm.task import deserialize_args
from tvm import te
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm.tir.expr import Mul

# For depthwise_conv2d, kernels are normally given in HWOI format,
# which when input_channels = output channels, we will call HWC.
# This is bad, as we want "related" parts of the kernel to be next
# to each other, so we can use __SMLAD later.
#
# Consider a 3x3 int8 kernel with no bias vector, with eight
# channels. Let us specify entries in the kernel as H_W_C - i.e.
# where 0_2_3 represents the rightmost position in the first row
# of channel 4/8 (4 because of zero indexing). Each [ ] represents
# a 32-bit integer. We currently store the kernel as:
#
# 0 ................................31
# [ 0_0_0 || 0_0_1 || 0_0_2 || 0_0_3 ] [ 0_0_4 || 0_0_5 || 0_0_6 || 0_0_7 ]
# [ 0_1_0 || 0_1_1 || 0_1_2 || 0_1_3 ] [ 0_1_4 || 0_1_5 || 0_1_6 || 0_1_7 ]
# [ 0_2_0 || 0_2_1 || 0_2_2 || 0_2_3 ] [ 0_2_4 || 0_2_5 || 0_2_6 || 0_2_7 ]
# [ 1_0_0 || 1_0_1 || 1_0_2 || 1_0_3 ] [ 1_0_4 || 1_0_5 || 1_0_6 || 1_0_7 ]
# [ 1_1_0 || 1_1_1 || 1_1_2 || 1_1_3 ] [ 1_1_4 || 1_1_5 || 1_1_6 || 1_1_7 ]
# [ 1_2_0 || 1_2_1 || 1_2_2 || 1_2_3 ] [ 1_2_4 || 1_2_5 || 1_2_6 || 1_2_7 ]
# [ 2_0_0 || 2_0_1 || 2_0_2 || 2_0_3 ] [ 2_0_4 || 2_0_5 || 2_0_6 || 2_0_7 ]
# [ 2_1_0 || 2_1_1 || 2_1_2 || 2_1_3 ] [ 2_1_4 || 2_1_5 || 2_1_6 || 2_1_7 ]
# [ 2_2_0 || 2_2_1 || 2_2_2 || 2_2_3 ] [ 2_2_4 || 2_2_5 || 2_2_6 || 2_2_7 ]
#
# Let 0x00 be all zeros. We rearrange into:
#
# 0 ................................31
# [ 0_0_0 || 0_0_1 || 0_1_0 || 0_1_1 ] [ 0_0_2 || 0_0_3 || 0_1_2 || 0_1_3 ]
# [ 0_2_0 || 0_2_1 || 1_0_0 || 1_0_1 ] [ 0_2_2 || 0_2_3 || 1_0_2 || 1_0_3 ]
# [ 1_1_0 || 1_1_1 || 1_2_0 || 1_2_1 ] [ 1_1_2 || 1_1_3 || 1_2_2 || 1_2_3 ]
# [ 2_0_0 || 2_0_1 || 2_1_0 || 2_1_1 ] [ 2_0_2 || 2_0_3 || 2_1_2 || 2_1_3 ]
# [ 2_2_0 || 2_2_1 || 0x000 || 0x000 ] [ 2_2_2 || 2_2_3 || 0x000 || 0x000 ]
# [ 0_0_4 || 0_0_5 || 0_1_4 || 0_1_5 ] [ 0_0_6 || 0_0_7 || 0_1_6 || 0_1_7 ]
# [ 0_2_4 || 0_2_5 || 1_0_4 || 1_0_5 ] [ 0_2_6 || 0_2_7 || 1_0_6 || 1_0_7 ]
# [ 1_1_4 || 1_1_5 || 1_2_4 || 1_2_5 ] [ 1_1_6 || 1_1_7 || 1_2_6 || 1_2_7 ]
# [ 2_0_4 || 2_0_5 || 2_1_4 || 2_1_5 ] [ 2_0_6 || 2_0_7 || 2_1_6 || 2_1_7 ]
# [ 2_2_4 || 2_2_5 || 0x000 || 0x000 ] [ 2_2_6 || 2_2_7 || 0x000 || 0x000 ]
#
# This saves us six operations comapred to the original ordering, as we
# do not need halfword packing instructions.
#
# This kernel re-arranging function will be used for 3x3 kernels (as that
# is all this DSP implementation currently supports) but would work with
# any M*N kernel such that M*N is odd.


def _rearrange_kernel(kernel):
    # Kernel must be HWC format.
    K_H, K_W, C, _ = get_const_tuple(kernel.shape)
    assert C % 4 == 0

    # TODO remove this restriction
    assert (K_W * K_H) % 2 == 1

    def fcompute(c_o, pos, c_i):
        channel = (2 * (pos % 2)) + (c_i % 2) + (4 * c_o)
        true_pos_index = 2 * (pos // 2) + (c_i // 2)

        return tir.if_then_else(
            true_pos_index < (K_H * K_W),
            kernel[true_pos_index // K_W, true_pos_index % K_W, channel, 0],
            tir.const(0, "int8"),
        )

    return te.compute(
        (C // 4, K_H * K_W + 1, 4), lambda co, pos, ci: fcompute(co, pos, ci), name="packed_kernel"
    )


def depthwise_conv2d_nhwc_dsp(*args, **kwargs):
    """Defines the v7e-m DSP instructions of depthwise_conv2d."""
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    layout = args[-2]
    cfg = autotvm.get_config()
    args = [cfg] + args
    assert layout == "NHWC"
    conv = depthwise_conv2d_nhwc_dsp_compute(*args)
    sched = depthwise_conv2d_nhwc_dsp_schedule(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


depthwise_conv2d_nhwc_dsp.template_key = "dsp"
depthwise_conv2d_nhwc_dsp.default_data_layout = "NHWC"
depthwise_conv2d_nhwc_dsp.default_kernel_layout = "HWOI"


def depthwise_conv2d_nhwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute function for v7e-m DSP instructions of DepthwiseConv2D. Has a lot of requirements
    for use - not not all apply, the fallback implementation will be used instead."""
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        STRIDE_H = STRIDE_W = strides
    else:
        STRIDE_H, STRIDE_W = strides

    # We do not support dilation currently. It would be possible, but it would require
    # modifying the way the kernel is packed. Gnarly.
    if isinstance(dilation, int):
        DILATION_H = DILATION_W = dilation
    else:
        DILATION_H, DILATION_H = dilation
    assert DILATION_H == DILATION_H == 1

    B, H, W, C = data.shape
    K_H, K_W, _, _ = kernel.shape

    # We require that the number of channels be divisible by 4. This restriction could
    # be removed with strip mining if people cared.
    assert C % 4 == 0

    # We don't support different numbers of input and output channels.
    assert C == kernel.shape[2]
    assert kernel.shape[3] == 1

    # The int16 case could also be optimized here, but would require writing a whole new
    # micro kernel. Probably not worth it.
    assert out_dtype == "int8"

    # This can pretty easily be generalized in the future. Likely worth doing, and this
    # function was written to make doing so easy. Should only require adding more calls
    # to QUAD_CHANNEL_REARRANGE_SUM.
    assert K_W == K_H == 3

    # We do not currently support custom padding. Would be pretty easy to implement.
    assert padding == "SAME" or padding == "VALID"

    # Padding the data requires COPYING THE ENTIRE INPUT TENSOR, which
    # is slow and bad. We should really implement a strip mining
    # routine to avoid this, but TVM has terrible support for that.

    if padding == "SAME":
        # This assumption makes the logic easier. Could be removed with work.
        assert H % STRIDE_H == W % STRIDE_W == 0

        OUT_H = H // STRIDE_H
        OUT_W = W // STRIDE_W

        # Padding this way is weird and quirky, and we only do it to match TFLite.
        pad_top = 1 if STRIDE_H == 1 else 0
        pad_left = 1 if STRIDE_W == 1 else 0

        data_padded = pad(
            data, [0, pad_top, pad_left, 0], [0, K_H // 2, K_W // 2, 0], name="padded_data"
        )

    elif PADDING_STRATEGY == "VALID":
        assert H > K_H and W > K_W
        OUT_H = (H - K_H) // STRIDE_H + 1
        OUT_W = (W - K_W) // STRIDE_W + 1
        data_padded = data

    else:
        raise RuntimeError()
    _, P_H, P_W, _ = data_padded.shape

    packed_kernel = _rearrange_kernel(kernel)
    kh_i = te.reduce_axis((0, K_H), name="kh_i")
    kw_i = te.reduce_axis((0, K_W), name="kw_i")
    return te.compute(
        (B, OUT_H, OUT_W, C),
        lambda h, i, j, k: te.sum(
            DATA_PAD[h, (i * STRIDE_H) + kh_i, (j * STRIDE_W) + kw_i, k]
            * PACKED_KER[
                k // 4,
                (2 * ((3 * kh_i + kw_i) // 2)) + ((k % 4) // 2),
                (2 * ((kh_i + kw_i) % 2)) + (k % 2),
            ],
            axis=(kh_i, kw_i),
        ),
        name="depthwise_conv2d",
        tag=f"depthwise_conv2d_nhwc_{P_H}_{P_W}_dsp",
    )


def depthwise_conv2d_nhwc_dsp_schedule(cfg, outs):

    """Schedule function for v7e-m DSP instructions of conv2d."""
    schedule = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "depthwise_conv2d_nhwc" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        padded_data = conv_out.op.input_tensors[0]
        packed_kernel = conv_out.op.input_tensors[1]
        kernel = packed_kernel.op.input_tensors[0]

        B, P_H, P_W, C = padded_data.shape
        K_H, K_W, _, _ = kernel.shape
        suffix = "".join(random.choices(string.ascii_uppercase, k=8))

        b_ax, y_ax, x_ax, c_ax = schedule[output].op.axis
        ky_ax, kx_ax = schedule[output].op.reduce_axis
        c_ax_o, c_ax_i = schedule[output].split(c_ax, factor=4)
        schedule[output].reorder(b_ax, c_ax_o, y_ax, x_ax, ky_ax, kx_ax, c_ax_i)

        quad_channel_convolve = intrin_quad_channel_convolve_3x3(P_H, P_W, C, K_H, K_W, suffix)
        s[CONVOLVED].tensorize(ky_ax, gemv)
        sched[output].pragma(
            b_ax, "import_c", quad_channel_convolve_3x3_impl(P_H, P_W, C, K_H, K_W, suffix)
        )

    traverse_inline(sched, outs[-1].op, _callback)
    return sched


def intrin_quad_channel_convolve_3x3(P_H, P_W, C, K_H, K_W, suffix):
    a = te.placeholder((K_H, K_W, 4), name="a", dtype="int8")
    b = te.placeholder((K_H * K_W + 1, 4), name="b", dtype="int8")
    kh_i = te.reduce_axis((0, K_H), name="kh_i")
    kw_i = te.reduce_axis((0, K_W), name="kw_i")

    c = te.compute(
        (4,),
        lambda k: te.sum(
            a[kh_i, kw_i, k]
            * b[
                (2 * ((3 * kh_i + kw_i) // 2)) + ((k % 4) // 2),
                (2 * ((kh_i + kw_i) % 2)) + (k % 2),
            ],
            axis=(kh_i, kw_i),
        ),
        name="c",
    )

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[P_W * C, C, 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[4, 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                f"kernel_convolve_noedge_{P_H}_{P_W}_{C}_{K_H}_{K_W}_{suffix}",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def quad_channel_convolve_3x3_impl(P_H, P_W, C, K_H, K_W, suffix):
    return (
        textwrap.dedent(
            f"""

        #include <stdint.h>

        // __SXTB16(_ROR(X, Y)) is combined into one assembly instruction

        #define QUAD_CHANNEL_TENSOR_REARRANGE_SUM_DSP( \
            arranged_kernel, \
            tensor_v0_c3210, tensor_v1_c3210, \
            sum0, sum1, sum2, sum3) {{ \
          \
          uint32_t tensor_v0_c20 = __SXTB16(tensor_v0_c3210); \
          uint32_t tensor_v0_c31 = __SXTB16(__ROR(tensor_v0_c3210, 8)); \
          uint32_t tensor_v1_c20 = __SXTB16(tensor_v1_c3210); \
          uint32_t tensor_v1_c31 = __SXTB16(__ROR(tensor_v1_c3210, 8)); \
          \
          uint32_t kernel_v1c1_v1c0_v0c1_v0c0 = *arranged_kernel++; \
          uint32_t kernel_v1c3_v1c2_v0c3_v0c2 = *arranged_kernel++; \
          \
          uint32_t kernel_v10_c0 = __SXTB16(kernel_v1c1_v1c0_v0c1_v0c0); \
          uint32_t kernel_v10_c1 = __SXTB16(__ROR(kernel_v1c1_v1c0_v0c1_v0c0, 8)); \
          uint32_t kernel_v10_c2 = __SXTB16(kernel_v1c3_v1c2_v0c3_v0c2); \
          uint32_t kernel_v10_c3 = __SXTB16(__ROR(kernel_v1c3_v1c2_v0c3_v0c2, 8)); \
          \
          uint32_t tensor_v10_c0 = __PKHBT(tensor_v0_c20, tensor_v1_c20, 16); \
          uint32_t tensor_v10_c1 = __PKHBT(tensor_v0_c31, tensor_v1_c31, 16); \
          uint32_t tensor_v10_c2 = __PKHTB(tensor_v1_c20, tensor_v0_c20, 16); \
          uint32_t tensor_v10_c3 = __PKHTB(tensor_v1_c31, tensor_v0_c31, 16); \
          \
          sum_c0 = __SMLAD(tensor_v10_c0, kernel_v10_c0, sum_c0); \
          sum_c1 = __SMLAD(tensor_v10_c1, kernel_v10_c1, sum_c1); \
          sum_c2 = __SMLAD(tensor_v10_c2, kernel_v10_c2, sum_c2); \
          sum_c3 = __SMLAD(tensor_v10_c3, kernel_v10_c3, sum_c3); \
        }}


        /* Here, we want to take the LOWER BYTES of 32 bit numbers v3 - v0
        * and concatenate them as "v3 || v2 || v1 || v0". In C++, this is:

        return ((sum_c0 & 0x000000FF)) +
               ((sum_c1 & 0x000000FF) << 8) +
               ((sum_c2 & 0x000000FF) << 16) +
               ((sum_c3 & 0x000000FF) << 24);

        * Naively, this takes 4x ands, 3x adds, 3x shifts. With optimization flags,
        * clang optimizes this down to eight operations:

            mov     r12, #255
            and     r0, r0, #255
            orr     r12, r12, #65280
            and     r1, r12, r1, lsl #8
            orr     r0, r1, r0
            and     r1, r2, #255
            orr     r0, r0, r1, lsl #16
            orr     r0, r0, r3, lsl #24

        * But being clever engineers, we can do it in four instructions. I think,
        * but have been unable to prove, that fewer is impossible. */

        #define WRITE_QUAD_BYTE_JOIN_DSP(out, v3, v2, v1, v0) {{ \
          uint32_t v3_00_v1_00 = PKHBT(v1 << 8, v3, 24); \
          uint32_t gg_v2_gg_v0 = PKHBT(v0, v2, 16); \
          out[0] = UXTAB16(v3_00_v1_00, gg_v2_gg_v0); \
        }}

        /* We do four channels at once to get this speed boost. */
        extern "C" int kernel_convolve_noedge_{P_H}_{P_W}_{C}_{K_H}_{K_W}_{suffix}(
            uint32_t *out,
            uint32_t *tensor,
            uint32_t *packed_kernel) {{

          uint32_t sum_c0 = 0;
          uint32_t sum_c1 = 0;
          uint32_t sum_c2 = 0;
          uint32_t sum_c3 = 0;

          QUAD_CHANNEL_TENSOR_REARRANGE_SUM(
            packed_kernel, *tensor, *(tensor + {C // 4}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          QUAD_CHANNEL_TENSOR_REARRANGE_SUM(
            packed_kernel, *(tensor + {(2) * C // 4}), *(tensor + {P_W * (C // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          QUAD_CHANNEL_TENSOR_REARRANGE_SUM(
            packed_kernel, *(tensor + {(P_W + 1) * (C // 4)}), *(tensor + {(P_W + 2) * (C // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          QUAD_CHANNEL_TENSOR_REARRANGE_SUM(
            packed_kernel, *(tensor + {(2 * P_W) * (C // 4)}), *(tensor + {(2 * P_W + 1) * (C // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          QUAD_CHANNEL_TENSOR_REARRANGE_SUM(
            packed_kernel, *(tensor + {(2 * P_W + 2) * (C // 4)}), 0,
            sum_c0, sum_c1, sum_c2, sum_c3)

          WRITE_QUAD_BYTE_JOIN_DSP(out, sum_c3, sum_c2, sum_c1, sum_c0);
        }}
        """
        ),
    )
