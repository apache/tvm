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
# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Bitserial conv2d schedule on arm cpu"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from ..nn.pad import pad
from ..nn.bitserial_conv2d import bitserial_conv2d_legalize
from ..nn.bitserial_util import bitpack, binary_op_multiplier
from ..nn.utils import get_pad_tuple
from ..utils import get_const_int, get_const_tuple, traverse_inline


def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, use_bitpack=True):
    if use_bitpack:
        kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type="uint8")
    else:
        kernel_q = kernel
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO // VC, KH, KW, KB, VC, CI)
    return te.compute(
        kvshape,
        lambda co, dh, dw, b, vc, ci: kernel_q[dh][dw][b][ci][co * VC + vc],
        name="kernel_vec",
    )


@autotvm.register_topi_compute("bitserial_conv2d_nhwc.arm_cpu")
def bitserial_conv2d_nhwc(
    cfg,
    data,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    pack_dtype,
    out_dtype,
    unipolar,
):
    """Compute convolution with pack on spatial axes."""
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    assert pack_dtype == "uint8", "only support packing into uint8 bits"
    assert out_dtype == "int16", "only support output type of int16"

    N, H, W, CI = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        CI_packed = CI // 8
    else:
        KH, KW, KB, CI_packed, CO = get_const_tuple(kernel.shape)

    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH - 1, KW - 1

    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (1, OH, OW, CO)

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    # Pad input channels of weights and data when it is not a multiple of 8
    if CI_packed % 8 != 0:
        CI_PAD = CI_packed % 8
        CI_packed += CI_PAD
    else:
        CI_PAD = 0

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI_packed), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(activation_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split("tile_co", co, num_outputs=2, filter=lambda x: x.size[-1] == 8)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2, filter=lambda x: x.size[-1] >= 2)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda x: x.size[-1] >= 2)
    ci_o, ci_i = cfg.define_split(
        "tile_ci", ci, num_outputs=2, filter=lambda x: x.size[-1] == 8 or x.size[-1] == 16
    )
    re_axes = cfg.define_reorder(
        "reorder_0",
        [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
        policy="candidate",
        candidate=[
            [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
            [n, oh, ow, co, vh, vw, kw, kh, ci_o, kb, ib, vc, ci_i],
        ],
    )
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type="uint8")

    kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC, len(kernel.shape) == 4)
    idxm = tvm.tir.indexmod
    if idxm(kernel_vec.shape[-1], 8) != 0 and CI_PAD != 0:
        kernel_vec = pad(kernel_vec, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, CI_PAD])

    N, H, W, IB, CI = data_q.shape
    OCO, KH, KW, KB, VC, CI = kernel_vec.shape

    dvshape = (
        N,
        PAD_H // (VH * HSTR),
        PAD_W // (VW * WSTR),
        VH * HSTR + HCAT,
        VW * WSTR + WCAT,
        IB,
        CI,
    )
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)

    if TPAD != 0 and RPAD != 0:
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, CI_PAD), name="data_pad")
    elif CI_PAD != 0:
        data_pad = pad(data_q, (0, 0, 0, 0, 0), (0, 0, 0, 0, CI_PAD), name="data_pad")
    else:
        data_pad = data_q

    data_vec = te.compute(
        dvshape,
        lambda n, h, w, vh, vw, b, ci: data_pad[n][h * VH * HSTR + vh][w * VW * WSTR + vw][b][ci],
        name="data_vec",
    )
    ci = te.reduce_axis((0, CI), name="ci")
    dh = te.reduce_axis((0, KH), name="dh")
    dw = te.reduce_axis((0, KW), name="dw")
    ib = te.reduce_axis((0, IB), name="ib")
    kb = te.reduce_axis((0, KB), name="kb")

    def _bipolar_conv(n, h, w, co, vh, vw, vc):
        return te.sum(
            (
                tvm.tir.popcount(
                    kernel_vec[co, dh, dw, kb, vc, ci].astype("uint16")
                    & data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ib, ci].astype("uint16")
                )
                << (kb + ib).astype("uint16")
            ),
            axis=[dh, dw, kb, ib, ci],
        )

    def _unipolar_conv(n, h, w, co, vh, vw, vc):
        return te.sum(
            (
                (
                    tvm.tir.popcount(
                        kernel_vec[co, dh, dw, kb, vc, ci].astype("int16")
                        & data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ib, ci].astype("int16")
                    )
                    - tvm.tir.popcount(
                        ~kernel_vec[co, dh, dw, kb, vc, ci].astype("int16")
                        & data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ib, ci]
                    ).astype("int16")
                )
                << (kb + ib).astype("int16")
            ),
            axis=[dh, dw, kb, ib, ci],
        )

    if unipolar:
        conv_vec = te.compute(ovshape, _unipolar_conv, name="conv_vec", tag="unipolar")
    else:
        conv_vec = te.compute(ovshape, _bipolar_conv, name="conv_vec", tag="bipolar")

    conv = te.compute(
        oshape,
        lambda n, h, w, co: conv_vec[
            n, idxd(h, VH), idxd(w, VW), idxd(co, VC), idxm(h, VH), idxm(w, VW), idxm(co, VC)
        ].astype(out_dtype),
        name="conv",
        tag="spatial_bitserial_conv_nhwc",
    )

    return conv


def _intrin_popcount(m, k_i, w_b, x_b, unipolar):
    pack_dtype = "uint8"
    w = te.placeholder((w_b, m, k_i), dtype=pack_dtype, name="w")
    x = te.placeholder(
        (
            x_b,
            k_i,
        ),
        dtype=pack_dtype,
        name="x",
    )
    k = te.reduce_axis((0, k_i), name="k")
    bw = te.reduce_axis((0, w_b), name="bw")
    bx = te.reduce_axis((0, x_b), name="bx")
    if unipolar:
        dtype = "int16"
        z = te.compute(
            (m,),
            lambda i: te.sum(
                (
                    tvm.tir.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                    - tvm.tir.popcount(~w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                )
                << (bw + bx).astype(dtype),
                axis=[bw, bx, k],
            ),
            name="z",
        )
    else:
        dtype = "uint16"
        z = te.compute(
            (m,),
            lambda i: te.sum(
                tvm.tir.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                << (bw + bx).astype(dtype),
                axis=[bw, bx, k],
            ),
            name="z",
        )
    Wb = tvm.tir.decl_buffer(
        w.shape, w.dtype, name="W", offset_factor=k_i, strides=[te.var("ldw"), te.var("ldw"), 1]
    )  # stride can be inferred
    Xb = tvm.tir.decl_buffer(
        x.shape, x.dtype, name="X", offset_factor=k_i, strides=[te.var("ldw"), 1]
    )
    Zb = tvm.tir.decl_buffer(z.shape, z.dtype, name="Z", offset_factor=1, strides=[1])

    def _intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]

        args_2 = tvm.tir.const(2, "uint32")

        if unipolar:
            vpadd = "llvm.arm.neon.vpadd.v8i8"
            vpadalu = "llvm.arm.neon.vpadals.v16i8.v8i16"
            full_dtype = "int8x16"
            half_dtype = "int8x8"
            return_dtype = "int16x8"
        else:
            vpadd = "llvm.arm.neon.vpadd.v8u8"
            vpadalu = "llvm.arm.neon.vpadalu.v16u8.v8u16"
            full_dtype = "uint8x16"
            half_dtype = "uint8x8"
            return_dtype = "uint16x8"

        def _instr(index):
            irb = tvm.tir.ir_builder.create()
            if index == 1:  # reduce reset
                irb.emit(zz.vstore(0, tvm.tir.const(0, return_dtype)))
                return irb.get()
            # body and reduce update
            cnts8 = [None] * 8
            cnts4 = [None] * 4
            cnts2 = [None] * 2
            for bw in range(w_b):
                for bx in range(x_b):
                    if k_i == 16:
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], "uint8x16").astype(full_dtype)
                            x_ = xx.vload([bx, 0], "uint8x16").astype(full_dtype)
                            if unipolar:
                                cnts = tvm.tir.popcount(w_ & x_) - tvm.tir.popcount(~w_ & x_)
                            else:
                                cnts = tvm.tir.popcount(w_ & x_)
                            upper_half = tvm.tir.call_intrin(half_dtype, "tir.vectorhigh", cnts)
                            lower_half = tvm.tir.call_intrin(half_dtype, "tir.vectorlow", cnts)
                            cnts8[i] = upper_half + lower_half
                        for i in range(m // 2):
                            cnts4[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts8[i * 2], cnts8[i * 2 + 1]
                            )
                        for i in range(m // 4):
                            cnts2[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts4[i * 2], cnts4[i * 2 + 1]
                            )
                        cnts = tvm.tir.call_intrin(
                            full_dtype, "tir.vectorcombine", cnts2[0], cnts2[1]
                        )
                        shifted_cnts = cnts << tvm.tir.const(bw + bx, pack_dtype)
                        out = tvm.tir.call_llvm_pure_intrin(
                            return_dtype, vpadalu, args_2, zz.vload(0, return_dtype), shifted_cnts
                        )
                    else:  # ki == 8
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], "uint8x8").astype(half_dtype)
                            x_ = xx.vload([bx, 0], "uint8x8").astype(half_dtype)
                            if unipolar:
                                cnts8[i] = tvm.tir.popcount(w_ & x_) - tvm.tir.popcount(~w_ & x_)
                            else:
                                cnts8[i] = tvm.tir.popcount(w_ & x_)
                        for i in range(m // 2):
                            cnts4[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts8[i * 2], cnts8[i * 2 + 1]
                            )
                        for i in range(m // 4):
                            cnts2[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts4[i * 2], cnts4[i * 2 + 1]
                            )
                        cnts = tvm.tir.call_intrin(
                            full_dtype, "tir.vectorcombine", cnts2[0], cnts2[1]
                        )
                        shifted_cnts = cnts << tvm.tir.const(bw + bx, pack_dtype)
                        out = tvm.tir.call_llvm_pure_intrin(
                            return_dtype, vpadalu, args_2, zz.vload(0, return_dtype), shifted_cnts
                        )
                    irb.emit(zz.vstore(0, out))
            return irb.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds={w: Wb, x: Xb, z: Zb}, default_buffer_params=buffer_params
    )


# ARM specific schedule that using custom microkernel
def _schedule_spatial_conv2d_nhwc(
    cfg, s, data_pad, data_vec, kernel_vec, conv_out, output, last, unipolar
):
    _, _, _, _, _, IB, CI = data_vec.shape
    _, KH, KW, KB, _, _ = kernel_vec.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule data padding and  packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    s[data_vec].parallel(oh)

    #### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    s[kernel_vec].parallel(oco)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    kh, kw, kb, ib, ci = s[conv_out].op.reduce_axis

    ci_o, ci_i = cfg["tile_ci"].apply(s, conv_out, ci)
    re_axes = cfg["reorder_0"].apply(
        s, conv_out, [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i]
    )

    # Use microkernel
    kfactor = cfg["tile_ci"].size[1]
    if kfactor % 8 == 0:
        pc = _intrin_popcount(VC, kfactor, KB, IB, unipolar)
        s[conv_out].tensorize(kb, pc)

    n, h, w, co = s[last].op.axis
    co, vc = cfg["tile_co"].apply(s, last, co)
    oh, vh = cfg["tile_oh"].apply(s, last, h)
    ow, vw = cfg["tile_ow"].apply(s, last, w)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)
    s[last].vectorize(vc)
    if last != output:
        s[output].compute_inline()

    s[conv_out].compute_at(s[last], co)
    s[last].parallel(oh)
    return s


@autotvm.register_topi_schedule("bitserial_conv2d_nhwc.arm_cpu")
def schedule_bitserial_conv2d_nhwc(cfg, outs):
    """Arm cpu schedule for bitserial conv2d"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "spatial_bitserial_conv_nhwc" in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[1]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, te.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data.op.input_tensors[0]
            unipolar = "unipolar" in conv_out.op.tag
            _schedule_spatial_conv2d_nhwc(
                cfg, s, data_pad, data_vec, kernel_vec, conv_out, output, outs[0], unipolar
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


@bitserial_conv2d_legalize.register("arm_cpu")
def _bitserial_conv2d_legalize(attrs, inputs, arg_types):
    """Legalizes Bitserial Conv2D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """

    # Fix different kernel layouts where possible.
    if attrs["data_layout"] == "NHWC":
        data, kernel = inputs
        if len(kernel.data.shape) == 4:
            # HWIO layout is expected for NHWC input.
            if attrs["kernel_layout"] == "HWOI":
                # Handle HWOI layout. This is common in TF depthwise conv2d graph.
                kernel = relay.transpose(kernel, axes=(0, 1, 3, 2))
            elif attrs["kernel_layout"] == "OIHW":
                kernel = relay.transpose(kernel, axes=(2, 3, 1, 0))
            ## Set new attrs for the tranposed conv.
            new_attrs = {k: attrs[k] for k in attrs.keys()}
            new_attrs["kernel_layout"] = "HWIO"

            conv = relay.nn.bitserial_conv2d(data, kernel, **new_attrs)
            return conv
    return None
