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
"""
Gemmini related intrinsics
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from __future__ import absolute_import as _abs

from typing import List, Tuple
import tvm
from tvm import te


def gemm(
    env,
    dim_i: int,
    dim_k: int,
    dim_j: int,
    stride: int = 1,
    is_depthwise_conv2d: bool = True,
    mode: int = 1,
    accum_patch: tvm.tir.Var = None,
):
    """Matrix-matrix multiply intrinsic, inserts the most basic Gemmini instructions

    Args:
        env (Environment): Environment with configurations
        dim_i (int): output first axis dimension
        dim_k (int): reduction axis dimension
        dim_j (int): output second axis dimension
        stride (int, optional): Stride, useful for convolutions. Defaults to 1.
        is_depthwise_conv2d (bool, optional): Flag to explain if this is a GEMM for a depthwise convolution. Defaults to False.
        mode (int, optional): Systolic array mode (WS=1,OS=0). Defaults to 1.
        accum_patch (tvm.tir.Var, optional): Var of the reduction axis loop. Defaults to None.

    Returns:
        TensorIntrin: gemm tensor intrinsic
    """

    # TODO (FP): add assertions here for dim_i, dim_k and dim_j?

    wgt_shape = (dim_k, dim_j)

    inp_shape = (dim_i, dim_k)

    out_shape = (dim_i, dim_j)

    wgt = te.placeholder(wgt_shape, dtype=env.wgt_dtype, name=env.scr_wgt_scope)
    inp = te.placeholder(inp_shape, dtype=env.inp_dtype, name=env.scr_scope)

    bias = te.placeholder(out_shape, dtype=env.inp_dtype, name=env.scr_scope)

    k = te.reduce_axis((0, wgt_shape[0]), name="k")

    out_dtype = env.inp_dtype

    if is_depthwise_conv2d:
        out = te.compute(
            out_shape,
            lambda i, j: te.sum(
                inp[i * stride + k, j].astype(env.inp_dtype) * wgt[0, k].astype(env.inp_dtype)
                + bias[i, j].astype(env.inp_dtype),
                axis=[k],
            ),
            name="out",
        )
    else:
        out = te.compute(
            out_shape,
            lambda i, j: te.sum(
                inp[i * stride, k].astype(env.inp_dtype) * wgt[k, j].astype(env.inp_dtype)
                + bias[i, j].astype(env.inp_dtype),
                axis=[k],
            ),
            name="out",
        )
    wgt_layout = tvm.tir.decl_buffer(
        wgt.shape,
        wgt.dtype,
        "wgt_buff",
        scope=env.scr_wgt_scope,
        strides=[te.var("wgt_k"), te.var("wgt_y")],
        offset_factor=env.DIM,
    )
    inp_layout = tvm.tir.decl_buffer(
        inp.shape,
        inp.dtype,
        "inp_buff",
        scope=env.scr_scope,
        strides=[te.var("inp_x"), te.var("inp_k")],
        offset_factor=env.DIM,
    )
    bias_layout = tvm.tir.decl_buffer(
        bias.shape,
        bias.dtype,
        "bias_buff",
        scope=env.acc_scope,
        strides=[te.var("inp_x"), te.var("inp_k")],
        offset_factor=env.DIM,
    )
    out_layout = tvm.tir.decl_buffer(
        out.shape,
        out_dtype,
        "out_buff",
        scope=env.acc_scope,
        strides=[te.var("out_x"), te.var("out_y")],
        offset_factor=env.DIM,
    )

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt, _ = ins
        dout = outs[0]

        inp_base_address = tvm.runtime.const(env.INP_SCR_BASE_ADDRESS, "uint32")
        wgt_base_address = tvm.runtime.const(env.WGT_SCR_BASE_ADDRESS, "uint32")
        wgt_access_ptr = dwgt.access_ptr("r", "uint32")
        out_base_address = tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
        out_access_ptr = dout.access_ptr("w", "uint32")

        garbage = tvm.runtime.const(0xFFFFFFFF, "uint32")

        def _body():
            """Generate matrix-matrix multiply Gemmini instruction, without accumulate (garbage address in compute_preloaded)"""
            irb = tvm.tir.ir_builder.create()

            inp_access_ptr = dinp.access_ptr("r", "uint32")

            a_access_ptr = inp_base_address + inp_access_ptr
            bd_access_ptr = (
                wgt_base_address + wgt_access_ptr if mode == env.WEIGHT_STATIONARY else garbage
            )
            c_access_ptr = out_base_address + out_access_ptr
            db_access_ptr = (
                garbage if mode == env.WEIGHT_STATIONARY else wgt_base_address + wgt_access_ptr
            )

            a_cols = dinp.shape[1]
            a_rows = dinp.shape[0]
            bd_cols = dwgt.shape[1] if mode == env.WEIGHT_STATIONARY else dout.shape[1]
            bd_rows = dwgt.shape[0] if mode == env.WEIGHT_STATIONARY else dout.shape[0]
            c_cols = dout.shape[1]
            c_rows = dout.shape[0]
            db_cols = c_cols if mode == env.WEIGHT_STATIONARY else dwgt.shape[1]
            db_rows = c_rows if mode == env.WEIGHT_STATIONARY else dwgt.shape[0]

            with irb.if_scope(accum_patch == 0):
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended_preload",
                        bd_access_ptr,
                        c_access_ptr,
                        bd_cols,
                        bd_rows,
                        c_cols,
                        c_rows,
                    )
                )
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended_compute_preloaded",
                        a_access_ptr,
                        db_access_ptr,
                        a_cols,
                        a_rows,
                        db_cols,
                        db_rows,
                    )
                )
            with irb.else_scope():
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended_preload",
                        garbage,
                        c_access_ptr,
                        bd_cols,
                        bd_rows,
                        c_cols,
                        c_rows,
                    )
                )
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended_compute_accumulated",
                        a_access_ptr,
                        db_access_ptr,
                        a_cols,
                        a_rows,
                        db_cols,
                        db_rows,
                    )
                )
            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="GEMM",
        binds={inp: inp_layout, wgt: wgt_layout, bias: bias_layout, out: out_layout},
    )


def gemm_cisc(
    env,
    inp_shape: Tuple[int, ...],
    wgt_shape: Tuple[int, ...],
    bias_shape: Tuple[int, ...],
    scale: float,
    matmul_type: int,
):
    """Matrix-matrix multiply intrinsic, inserts the calls to the function provided by the Gemmini developers to run matrix multiplication using the loop instructions

    Args:
        env (Environment): Environment with configurations
        inp_shape (Tuple[int,...]): Input feature map shape
        wgt_shape (Tuple[int,...]): Weights shape
        bias_shape (Tuple[int,...]): Bias shape
        scale (float): Output scaling factor
        matmul_type (int): Systolic array mode (WS=1,OS=0)

    Returns:
        TensorIntrin: GEMM CISC tensor intrinsic
    """

    # TODO (FP): add assertions here for inp_shape, wgt_shape and bias_shape?

    wgt = te.placeholder(wgt_shape, dtype=env.inp_dtype, name=env.scr_wgt_scope)
    inp = te.placeholder(inp_shape, dtype=env.inp_dtype, name=env.scr_scope)
    bias = te.placeholder(bias_shape, dtype=env.acc_dtype, name=env.scr_scope)

    dim_k = wgt.shape[0]
    dim_j = wgt.shape[1]
    dim_i = inp.shape[0]

    k_reduce = te.reduce_axis((0, dim_k), name="dim_k")

    output_shape = (dim_i, dim_j)

    out = te.compute(
        output_shape,
        lambda x_, y_: te.sum(
            inp[x_, k_reduce].astype(env.inp_dtype) * wgt[k_reduce, y_].astype(env.inp_dtype)
            + bias[y_].astype(env.inp_dtype),
            axis=[k_reduce],
        ),
    )

    wgt_layout = tvm.tir.decl_buffer(
        wgt_shape,
        env.inp_dtype,
        "wgt_buff",
    )
    inp_layout = tvm.tir.decl_buffer(
        inp_shape,
        env.inp_dtype,
        "inp_buff",
        strides=[te.var("inp_x"), te.var("inp_y")],
    )
    bias_layout = tvm.tir.decl_buffer(
        bias_shape,
        env.acc_dtype,
        "bias_buff",
    )
    out_layout = tvm.tir.decl_buffer(
        output_shape,
        env.inp_dtype,
        "out_buff",
    )

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt, dbias = ins
        dout = outs[0]

        def _body():
            irb = tvm.tir.ir_builder.create()
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "tiled_matmul_auto",
                    dinp.shape[0],  # dim_I,
                    dwgt.shape[1],  # dim_J,
                    dinp.shape[1],  # dim_K,
                    dinp.access_ptr("r"),
                    dwgt.access_ptr("r"),
                    dbias.access_ptr("r"),
                    dout.access_ptr("w"),
                    dinp.shape[0],  # stride_A
                    dwgt.shape[1],  # stride_B
                    dwgt.shape[1],  # stride_C
                    dwgt.shape[1],  # stride_D
                    1.0,  # A_scale_factor
                    1.0,  # B_scale_factor
                    1.0,  # D_scale_factor
                    0,  # act
                    scale,
                    0,  # relu6_shift
                    1,  # repeating_bias
                    0,  # transpose_A
                    0,  # transpose_B
                    0,  # full_C
                    0,  # low_D
                    # 0,
                    0,  # weightA
                    matmul_type,
                )
            )
            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="CONV2D_CISC",
        binds={inp: inp_layout, wgt: wgt_layout, bias: bias_layout, out: out_layout},
    )


def conv2d_cisc(
    env,
    inp_shape: Tuple[int, ...],
    wgt_shape: Tuple[int, ...],
    bias_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    strides: int,
    padding: List[int],
    padding_value: int,
    activation: int,
    scale: float,
    pool_size: List[int],
    pool_strides: List[int],
    pool_dilation: List[int],
    pool_padding: List[int],
):
    """2D convolution intrinsic, inserts the calls to the function provided by the Gemmini developers to run a 2D convolution using the loop instructions

    Args:
        env (Environment): Environment with configurations
        inp_shape (Tuple[int,...]): Input feature map shape
        wgt_shape (Tuple[int,...]): Weights shape
        bias_shape (Tuple[int,...]): Bias shape
        out_shape (Tuple[int,...]): Output feature map shape
        strides (int): Convolution stride
        padding (List[int]): Pixels to pad in each direction
        padding_value (int): Value to use for padding
        activation (int): Has activation?
        scale (float): Output scaling factor
        pool_size (List[int]): Size of the output pooling window
        pool_strides (List[int]): Strides for the output pooling window
        pool_dilation (List[int]): Dilation for the output pooling window. Not used for now.
        pool_padding (List[int]): Padding for the output pooling

    Returns:
        TensorIntrin: CONV2D CISC tensor intrinsic
    """

    # TODO (FP): add assertions here for the supported parameters?

    wgt = te.placeholder(wgt_shape, dtype=env.inp_dtype, name=env.scr_wgt_scope)
    inp = te.placeholder(inp_shape, dtype=env.inp_dtype, name=env.scr_scope)
    bias = te.placeholder(bias_shape, dtype=env.acc_dtype, name=env.scr_scope)

    wgt.shape[3]
    k_h = wgt.shape[0]
    k_w = wgt.shape[1]

    inp.shape[0]
    inp.shape[1]
    inp.shape[2]
    i_c = inp.shape[3]

    ric = te.reduce_axis((0, i_c), name="ric")
    rkh = te.reduce_axis((0, k_h), name="rkh")
    rkw = te.reduce_axis((0, k_w), name="rkw")

    hstr = strides[0]
    wstr = strides[1]

    out = te.compute(
        out_shape,
        lambda b_o, i, j, c_o: te.sum(
            inp[b_o, i * hstr + rkh, j * wstr + rkw, ric].astype(env.inp_dtype)
            * wgt[rkh, rkw, ric, c_o].astype(env.inp_dtype)
            + bias[c_o].astype(env.inp_dtype),
            axis=[rkh, rkw, ric],
        ),
    )

    wgt_layout = tvm.tir.decl_buffer(wgt_shape, env.inp_dtype, "wgt_buff")
    inp_layout = tvm.tir.decl_buffer(
        inp_shape,
        env.inp_dtype,
        "inp_buff",
        strides=[te.var("inp_x"), te.var("inp_y"), te.var("inp_b"), te.var("inp_k")],
    )
    bias_layout = tvm.tir.decl_buffer(
        bias_shape,
        env.acc_dtype,
        "bias_buff",
    )
    out_layout = tvm.tir.decl_buffer(
        out_shape,
        env.inp_dtype,
        "out_buff",
    )

    def intrin_func(ins, outs):
        """2D convolution intrinsic function"""
        dinp, dwgt, dbias = ins
        dout = outs[0]

        def _body():
            irb = tvm.tir.ir_builder.create()
            if env.supports_non_zero_padding:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "tiled_conv_auto",
                        dinp.shape[0],  # BATCH_SIZE,
                        dinp.shape[1],  # IN_DIM,
                        dinp.shape[3],  # IN_CHANNELS,
                        dout.shape[3],  # OUT_CHANNELS,
                        dout.shape[1],  # OUT_DIM,
                        strides[0],
                        1,
                        1,
                        padding[2],
                        padding_value,
                        dwgt.shape[0],
                        0,
                        0,
                        0,
                        0,
                        0,
                        dinp.access_ptr("r"),
                        dwgt.access_ptr("r"),
                        dbias.access_ptr("r"),
                        dout.access_ptr("w"),
                        activation,
                        scale,
                        pool_size[0],
                        pool_strides[0],
                        pool_padding[0],
                        1,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "tiled_conv_auto",
                        dinp.shape[0],  # BATCH_SIZE,
                        dinp.shape[1],  # IN_DIM,
                        dinp.shape[3],  # IN_CHANNELS,
                        dout.shape[3],  # OUT_CHANNELS,
                        dout.shape[1],  # OUT_DIM,
                        strides[0],
                        1,
                        1,
                        padding[2],
                        dwgt.shape[0],
                        0,
                        0,
                        0,
                        0,
                        0,
                        dinp.access_ptr("r"),
                        dwgt.access_ptr("r"),
                        dbias.access_ptr("r"),
                        dout.access_ptr("w"),
                        activation,
                        scale,
                        pool_size[0],
                        pool_strides[0],
                        pool_padding[0],
                        1,
                    )
                )
            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="CONV2D_CISC",
        binds={inp: inp_layout, wgt: wgt_layout, bias: bias_layout, out: out_layout},
    )


def dw_conv2d_cisc(
    env,
    inp_shape: Tuple[int, ...],
    wgt_shape: Tuple[int, ...],
    bias_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    strides: int,
    padding: List[int],
    padding_value: int,
    activation: int,
    scale: float,
):
    """2D depthwise convolution intrinsic, inserts the calls to the function provided by the Gemmini developers to run a 2D depthwise convolution using the loop instructions

    Args:
        env (Environment): Environment with configurations
        inp_shape (Tuple[int,...]): Input feature map shape
        wgt_shape (Tuple[int,...]): Weights shape
        bias_shape (Tuple[int,...]): Bias shape
        out_shape (Tuple[int,...]): Output feature map shape
        strides (int): Convolution stride
        padding (List[int]): Pixels to pad in each direction
        padding_value (int): Value to use for padding
        activation (int): Has activation?
        scale (float): Output scaling factor

    Returns:
        TensorIntrin: depthwise convolution 2d tensor intrinsic
    """

    # TODO (FP): add assertions here for the supported parameters?

    wgt = te.placeholder(wgt_shape, dtype=env.inp_dtype, name=env.scr_wgt_scope)
    inp = te.placeholder(inp_shape, dtype=env.inp_dtype, name=env.scr_scope)
    bias = te.placeholder(bias_shape, dtype=env.acc_dtype, name=env.scr_scope)

    wgt.shape[0]
    k_h = wgt.shape[1]
    k_w = wgt.shape[2]

    inp.shape[0]
    inp.shape[1]
    inp.shape[2]
    inp.shape[3]

    rkh = te.reduce_axis((0, k_h), name="rkh")
    rkw = te.reduce_axis((0, k_w), name="rkw")

    hstr = strides[0]
    wstr = strides[1]

    out = te.compute(
        out_shape,
        lambda b_o, i, j, c_o: te.sum(
            inp[b_o, i * hstr + rkh, j * wstr + rkw, c_o].astype(env.inp_dtype)
            * wgt[c_o, rkh, rkw].astype(env.inp_dtype)
            + bias[c_o].astype(env.inp_dtype),
            axis=[rkh, rkw],
        ),
    )

    wgt_layout = tvm.tir.decl_buffer(
        wgt_shape,
        env.inp_dtype,
        "wgt_buff",
        # strides=[te.var("wgt_i"),te.var("wgt_j")]
    )
    inp_layout = tvm.tir.decl_buffer(
        inp_shape,
        env.inp_dtype,
        "inp_buff",
        strides=[te.var("inp_x"), te.var("inp_y"), te.var("inp_b"), te.var("inp_k")],
    )
    bias_layout = tvm.tir.decl_buffer(
        bias_shape,
        env.acc_dtype,
        "bias_buff",
    )
    out_layout = tvm.tir.decl_buffer(
        out_shape,
        env.inp_dtype,
        "out_buff",
    )

    def intrin_func(ins, outs):
        """2D depthwise convolution intrinsic function"""
        dinp, dwgt, dbias = ins
        dout = outs[0]

        def _body():
            irb = tvm.tir.ir_builder.create()
            if env.supports_non_zero_padding:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "tiled_conv_dw_auto",
                        dinp.shape[0],  # BATCH_SIZE,
                        dinp.shape[1],  # IN_DIM,
                        dinp.shape[3],  # IN_CHANNELS,
                        # dout.shape[3],#OUT_CHANNELS,
                        dout.shape[1],  # OUT_DIM,
                        strides[0],
                        # 1, 1,
                        padding[2],
                        padding_value,
                        dwgt.shape[1],
                        # 0, 0, 0, 0, 0,
                        dinp.access_ptr("r"),
                        dwgt.access_ptr("r"),
                        dbias.access_ptr("r"),
                        dout.access_ptr("w"),
                        activation,
                        scale,
                        1,
                        0,
                        0,
                        1,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "tiled_conv_dw_auto",
                        dinp.shape[0],  # BATCH_SIZE,
                        dinp.shape[1],  # IN_DIM,
                        dinp.shape[3],  # IN_CHANNELS,
                        # dout.shape[3],#OUT_CHANNELS,
                        dout.shape[1],  # OUT_DIM,
                        strides[0],
                        # 1, 1,
                        padding[2],
                        dwgt.shape[1],
                        # 0, 0, 0, 0, 0,
                        dinp.access_ptr("r"),
                        dwgt.access_ptr("r"),
                        dbias.access_ptr("r"),
                        dout.access_ptr("w"),
                        activation,
                        scale,
                        1,
                        0,
                        0,
                        1,
                    )
                )

            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="DWCONV2D_CISC",
        binds={inp: inp_layout, wgt: wgt_layout, bias: bias_layout, out: out_layout},
    )


def add_tensorize(env, oshape: Tuple[int, ...]):
    """Add intrinsic, inserts the most basic Gemmini instructions to support the qnn.add operator

    Args:
        env (Environment): Environment with configurations
        oshape (Tuple[int,...]): Output feature map shape

    Returns:
        TensorIntrin: add tensor intrinsic
    """

    # TODO (FP): add assertions here for the supported parameters?

    ifm1 = te.placeholder(oshape, dtype=env.inp_dtype, name=env.acc_scope)
    ifm2 = te.placeholder(oshape, dtype=env.inp_dtype, name=env.acc_scope)

    out = te.compute(
        oshape, lambda i, j: ifm1[i, j].astype(env.inp_dtype) + ifm2[i, j].astype(env.inp_dtype)
    )

    ifm1_dtype = env.inp_dtype

    ifm1_layout = tvm.tir.decl_buffer(
        oshape,
        ifm1_dtype,
        "ifm1_buff",
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )
    ifm2_layout = tvm.tir.decl_buffer(
        oshape,
        env.inp_dtype,
        "ifm2_buff",
        scope=env.acc_scope,
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )
    out_layout = tvm.tir.decl_buffer(
        oshape,
        env.inp_dtype,
        "out_buff",
        scope=env.acc_scope,
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )

    def intrin_func(ins, outs):
        """Add intrinsic function"""
        difm1, difm2 = ins
        outs[0]

        def _body():
            irb = tvm.tir.ir_builder.create()
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin2",
                    difm1.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + difm2.access_ptr("w", "uint32"),
                    difm1.shape[1],
                    difm1.shape[0],
                )
            )

            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="ADD",
        binds={ifm1: ifm1_layout, ifm2: ifm2_layout, out: out_layout},
    )


def add_mvout_tensorize(env, oshape: Tuple[int, ...]):
    """Helper for the add intrinsic

    Args:
        env (Environment): Environment with configurations
        oshape (Tuple[int,...]): Output feature map shape

    Returns:
        TensorIntrin: add mvout tensor intrinsic
    """

    # TODO (FP): add assertions here for the supported parameters?

    ifm1 = te.placeholder(oshape, dtype=env.inp_dtype, name=env.acc_scope)
    ifm2 = te.placeholder(oshape, dtype=env.inp_dtype, name=env.acc_scope)

    out = te.compute(
        oshape, lambda i, j: ifm1[i, j].astype(env.inp_dtype) + ifm2[i, j].astype(env.inp_dtype)
    )

    ifm1_dtype = env.inp_dtype

    ifm1_layout = tvm.tir.decl_buffer(
        oshape,
        ifm1_dtype,
        "ifm1_buff",
        scope=env.acc_scope,
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )
    ifm2_layout = tvm.tir.decl_buffer(
        oshape,
        env.inp_dtype,
        "ifm2_buff",
        scope=env.acc_scope,
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )
    out_layout = tvm.tir.decl_buffer(
        oshape,
        env.inp_dtype,
        "out_buff",
        strides=[te.var("out_b"), te.var("out_x")],
        offset_factor=env.DIM,
    )

    def intrin_func(ins, outs):
        """Add mvout intrinsic function"""
        difm1, difm2 = ins
        dout = outs[0]

        def _body():
            irb = tvm.tir.ir_builder.create()
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvout",
                    dout.access_ptr("w"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + difm2.access_ptr("w", "uint32")
                    - tvm.runtime.const(0x40000000, "uint32"),
                    difm1.shape[1],
                    difm1.shape[0],
                )
            )

            return irb.get()

        def _reduce_reset():
            irb = tvm.tir.ir_builder.create()
            return irb.get()

        def _reduce_update():
            return _body()

        # return a triple of normal-set, reset, update
        return (_body(), _reduce_reset(), _reduce_update())

    return te.decl_tensor_intrin(
        out.op,
        intrin_func,
        name="ADD_MVOUT",
        binds={ifm1: ifm1_layout, ifm2: ifm2_layout, out: out_layout},
    )
