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

"""NNEF frontend converter helper funcs and ops"""
import math

import itertools
from functools import reduce

import numpy as np

import tvm
from tvm import relax
from tvm.relax import expr as tvm_expr
from tvm.relax import op as tvm_op
from tvm import topi


# Base methods


def dimension_picker(prefix, kernel_shape, suffix=""):
    """
    Returns the correct name for nth dimensional operator. Uses the "kernel_shape" attribute.\n
    E.g.call: dimension_picker(op_name)(attr)

    :param prefix: the name of the operator (e.g. conv)
    :param kernel_shape: shape of the tensor to fit the operation
    :param suffix: optional suffix for ops
    :return: "prefix`n`d" where n is the correct dimension for the kernel
    """

    rank = len(kernel_shape[2:])
    if rank == 1:
        return prefix + "1d" + suffix
    if rank == 2:
        return prefix + "2d" + suffix
    if rank == 3:
        return prefix + "3d" + suffix
    op_name = prefix + "1d/2d/3d"
    msg = f"Only 1D, 2D, and 3D kernels are supported for operator {op_name}."
    raise tvm.error.OpAttributeInvalid(msg)


def _size_conv(size, rank):
    # window of size (DH)W is only possible when it is checked outside,
    # which is needed for alternative solution
    if rank == 3:
        if len(size) == 1:
            return size
        if len(size) == 3:
            assert (
                size[0] == 1 and size[1] == 1
            ), "Incorrect window dimensions, first two dimensions must be 1"
            return size[2]
    if rank == 4:
        if len(size) == 2:
            return size
        if len(size) == 4:
            assert (
                size[0] == 1 and size[1] == 1
            ), "Incorrect window dimensions, first two dimensions must be 1"
            return size[2:]
    if rank == 5:
        if len(size) == 3:
            return size
        if len(size) == 5:
            assert (
                size[0] == 1 and size[1] == 1
            ), "Incorrect window dimensions, first two dimensions must be 1"
            return size[2:]

    raise ValueError(f"Unexpected window size, got {len(size)}")


def _stride_conv(stride, rank):
    if rank == 3:
        # {conv style} :: [s] -> [s]
        if len(stride) == 1:
            return stride
        # {pool style} :: [N, C, s] -> asrt N,C == 1; [s]
        if len(stride) == 3:
            assert (
                stride[0] == 1 and stride[1] == 1
            ), "Not supported stride dimensions, first two dimensions must be 1"
            return stride[2:]
    if rank == 4:
        # {conv style} :: [sh, sw] -> [sh, sw]
        if len(stride) == 2:
            return stride
        # {pool style} :: [N, C, sh, sw] -> asrt N,C == 1; [sh, sw]
        if len(stride) == 4:
            assert (
                stride[0] == 1 and stride[1] == 1
            ), "Not supported stride dimensions, first two dimensions must be 1"
            return stride[2:]
    if rank == 5:
        # {conv style} :: [sd, sh, sw] -> [sd, sh, sw]
        if len(stride) == 3:
            return stride
        # {pool style} :: [N, C, sd, sh, sw] -> asrt N,C == 1; [sd, sh, sw]
        if len(stride) == 5:
            assert (
                stride[0] == 1 and stride[1] == 1
            ), "Not supported stride dimensions, first two dimensions must be 1"
            return stride[2:]
    raise ValueError(f"Unexpected stride in {rank - 2}D, got {len(stride)}: {stride}")


def _padding_conv(padding, rank, keepdims=False):
    if isinstance(padding[0], (tuple, list)):
        # 1D
        if rank == 3:
            # {conv style} :: [(l,r)] -> (l,r)
            if len(padding) == 1:
                return padding[0]
            if len(padding) == 3:
                # {pool style} :: [(batch),(channel),(l,r)] -> asrt N,C == 0, (l,r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), (
                        "Incorrect padding. " "Padding on C,I dimensions not supported"
                    )
                    return padding[2]
                # {sliding window style} :: [(batch),(channel),(l,r)] -> [(batch),(channel),(l,r)]
                else:
                    return padding

        # 2D

        if rank == 4:
            # {conv style} :: [(u,d),(l,r)] -> (u, l, d, r)
            if len(padding) == 2:
                # change UDLR to ULDR padding, LC is faster here
                return [x[i] for i in [0, 1] for x in padding]

            if len(padding) == 4:
                # {pool style} :: [(batch size),(channel),(u,d),(l,r)] ->
                #                  -> asrt N,C == 0, (u, l, d, r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), (
                        "Incorrect padding. " "Padding on C,I dimensions not supported"
                    )
                    # itertools is faster than LC (slicing)
                    return list(itertools.chain.from_iterable(zip(padding[2], padding[3])))
                # {sliding window style} :: [(batch),(channel),(u,d),(l,r)] ->
                #                            -> [(batch),(channel),(u,d),(l,r)]
                else:
                    return padding

        # 3D

        if rank == 5:
            # {conv style} :: [(f,b),(u,d),(l,r)] -> (f, u, l, b, d, r)
            if len(padding) == 3:
                # LC is faster
                return [x[i] for i in [0, 1] for x in padding]

            if len(padding) == 5:
                # {pool style} :: [(batch size),(channel),(f,b)(u,p),(l,r)] ->
                #                  -> asrt N,C == 0, (f, u, l, b, d, r)
                if not keepdims:
                    assert padding[0] == (0, 0) and padding[1] == (0, 0), (
                        "Incorrect padding. " "Padding on C,I dimensions not supported"
                    )
                    # itertools faster barely
                    return list(
                        itertools.chain.from_iterable(zip(padding[2], padding[3], padding[4]))
                    )
                # {s-w style} :: [(batch),(channel),(f,b),(u,d),(l,r)] ->
                #                 -> [(batch),(channel),(f,b),(u,d),(l,r)]
                else:
                    return padding

        raise ValueError(
            f"Incorrect padding style for {rank - 2}D operand. Only length of {rank - 2}, {rank} "
            f"supported, got {len(padding)}: {padding}"
        )

    raise ValueError("nnef should not have singular padding")


def _calculate_nnef_padding(active_shape, strides, kernel_shape, dilation):
    """Ordering of nnef autopad and tvm autopad are sometimes different,
    this method calculates nnef like padding from dimensions

    Parameters
    ----------
        active_shape
            the data dimensions
        strides
            the strides over the active dimensions
        kernel_shape
            the shape of the window, must have the same rank as active shape
        dilation
            the dilations over the active dimensions
    """
    output = [(ui + (s - 1)) // s for ui, s in zip(active_shape, strides)]
    dilated = [(f - 1) * d + 1 for f, d in zip(kernel_shape, dilation)]
    total = [
        max(0, (di - 1) * s + df - ui)
        for di, s, df, ui in zip(output, strides, dilated, active_shape)
    ]
    padding = [(pad // 2, (pad + 1) // 2) for pad in total]
    return padding


def _calculate_nnef_padding_deconv(data_sh, strides, kernel_active_sh, dilation, output_shape):
    out_sh = output_shape[2:] if output_shape else [ui * s for ui, s in zip(data_sh, strides)]
    dilated = [(f - 1) * d + 1 for f, d in zip(kernel_active_sh[2:], dilation)]
    total = [
        max(0, (di - 1) * s + df - ui) for di, s, df, ui in zip(data_sh, strides, dilated, out_sh)
    ]
    return total, out_sh


def __unexpected_attrs(op, kwargs):
    raise NotImplementedError(
        f"{op} received unexpected attributes(s), possibly mismatched versions. "
        "Attributes(s) ignored: " + ", ".join(f"{k} := {v}" for k, v in kwargs.items())
    )


# Conversion map, operator functions


def _get_converter_map():
    return {  # Unary
        "copy": copy_converter,  # arithmetic
        "neg": neg_converter,
        "rcp": rcp_converter,
        "exp": exp_converter,
        "log": log_converter,
        "sin": sin_converter,
        "cos": cos_converter,
        "tan": tan_converter,
        "sinh": sinh_converter,
        "cosh": cosh_converter,
        "tanh": tanh_converter,
        "asin": asin_converter,
        "acos": acos_converter,
        "atan": atan_converter,
        "asinh": asinh_converter,
        "acosh": acosh_converter,
        "atanh": atanh_converter,
        "abs": abs_converter,
        "sign": sign_converter,
        "not": not_converter,  # logical
        "floor": floor_converter,  # rounding
        "ceil": ceil_converter,
        "round": round_converter,
        # Binary
        "add": add_converter,  # arithmetic
        "sub": sub_converter,
        "mul": mul_converter,
        "div": div_converter,
        "pow": pow_converter,
        "lt": lt_converter,  # comparison
        "gt": gt_converter,
        "le": le_converter,
        "ge": ge_converter,
        "eq": eq_converter,
        "ne": ne_converter,
        "and": and_converter,  # logical
        "or": or_converter,
        # select
        "select": select_converter,
        # simplifier
        "sqr": sqr_converter,
        "sqrt": sqrt_converter,
        "rsqr": rsqr_converter,
        "rsqrt": rsqrt_converter,
        "log2": log2_converter,
        "min": min_converter,
        "max": max_converter,
        "clamp": clamp_converter,
        # sliding-window
        "conv": conv_converter,
        "deconv": deconv_converter,
        "box": box_converter,
        "debox": debox_converter,
        "argmax_pool": ndop,
        "sample": ndop,
        "desample": ndop,
        "nearest_downsample": nearest_downsample_converter,
        "area_downsample": area_downsample_converter,
        "nearest_upsample": nearest_upsample_converter,
        "multilinear_upsample": multilinear_upsample_converter,
        # reduce
        "sum_reduce": sum_reduce_converter,
        "max_reduce": max_reduce_converter,
        "min_reduce": min_reduce_converter,
        "argmax_reduce": argmax_reduce_converter,
        "argmin_reduce": argmin_reduce_converter,
        "all_reduce": all_reduce_converter,
        "any_reduce": any_reduce_converter,
        "mean_reduce": mean_reduce_converter,
        # tensor shape
        "reshape": reshape_converter,
        "squeeze": squeeze_converter,
        "unsqueeze": unsqueeze_converter,
        "transpose": transpose_converter,
        "split": split_converter,
        "concat": concat_converter,
        "stack": stack_converter,
        "unstack": unstack_converter,
        "slice": slice_converter,
        "pad": pad_converter,
        "tile": tile_converter,
        # region-of-interest - not needed - not supported
        "avg_roi_pool": ndop,
        "max_roi_pool": ndop,
        "roi_resample": ndop,
        "avg_roi_align": ndop,
        "max_roi_align": ndop,
        # matrix multiplication
        "matmul": matmul_converter,
        # variables
        "update": ndop,  # --- not used
        # Compound
        "sigmoid": sigmoid_converter,  # activation
        "relu": relu_converter,
        "prelu": prelu_converter,
        "leaky_relu": leaky_relu_converter,
        "elu": elu_converter,
        "selu": selu_converter,
        "gelu": gelu_converter,
        "silu": silu_converter,
        "softmax": softmax_converter,
        "softplus": softplus_converter,
        "linear": linear_converter,  # linear
        "separable_conv": separable_conv_converter,
        "separable_deconv": separable_deconv_converter,
        "max_pool_with_index": ndop,  # pooling
        "max_pool": max_pool_converter,
        "avg_pool": avg_pool_converter,
        "rms_pool": rms_pool_converter,
        "local_response_normalization": local_response_normalization_converter,  # normalization
        "local_mean_normalization": local_mean_normalization_converter,
        "local_variance_normalization": local_variance_normalization_converter,
        "local_contrast_normalization": local_contrast_normalization_converter,
        "l1_normalization": l1_normalization_converter,
        "l2_normalization": l2_normalization_converter,
        "batch_normalization": batch_normalization_converter,
        "min_max_linear_quantize": ndop,  # quantization
        "zero_point_linear_quantize": ndop,
        "linear_quantize": ndop,
        "logarithmic_quantize": ndop,
        # MISC
        "copy_n": ndop,
        "add_n": ndop,
        "moments": ndop,
    }


# pylint: disable=unused-argument

# not implemented ops
def ndop(*args, **kwargs):
    # print(args, kwargs)
    raise Exception("Not supported operator was called, please check for compatibility")


#   # Unary ops


def copy_converter(bbuilder, data, **kwargs):
    """Copy converter"""
    if kwargs:
        __unexpected_attrs("copy", kwargs)

    return bbuilder.emit_te(topi.identity, data)


def neg_converter(bbuilder, data, **kwargs):
    """Neg converter"""
    if kwargs:
        __unexpected_attrs("neg", kwargs)

    return relax.op.unary.negative(data)


def rcp_converter(bbuilder, data, **kwargs):
    """Rcp converter"""
    if kwargs:
        __unexpected_attrs("rcp", kwargs)

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    return div_converter(bbuilder, tvm_expr.const(1, dtype=d_type), data)


def exp_converter(bbuilder, data, **kwargs):
    """Exp converter"""
    if kwargs:
        __unexpected_attrs("exp", kwargs)

    return relax.op.unary.exp(data)


def log_converter(bbuilder, data, **kwargs):
    """Log converter"""
    if kwargs:
        __unexpected_attrs("log", kwargs)

    return relax.op.unary.log(data)


def sin_converter(bbuilder, data, **kwargs):
    """Sin converter"""
    if kwargs:
        __unexpected_attrs("sin", kwargs)

    return relax.op.unary.sin(data)


def cos_converter(bbuilder, data, **kwargs):
    """Cos converter"""
    if kwargs:
        __unexpected_attrs("cos", kwargs)

    return relax.op.unary.cos(data)


def tan_converter(bbuilder, data, **kwargs):
    """Tan converter"""
    if kwargs:
        __unexpected_attrs("tan", kwargs)

    return relax.op.unary.tan(data)


def sinh_converter(bbuilder, data, **kwargs):
    """Sinh converter"""
    if kwargs:
        __unexpected_attrs("sinh", kwargs)

    return relax.op.unary.sinh(data)


def cosh_converter(bbuilder, data, **kwargs):
    """Cosh converter"""
    if kwargs:
        __unexpected_attrs("cosh", kwargs)

    return relax.op.unary.cosh(data)


def tanh_converter(bbuilder, data, **kwargs):
    """Tanh converter"""
    if kwargs:
        __unexpected_attrs("tanh", kwargs)

    return relax.op.unary.tanh(data)


def asin_converter(bbuilder, data, **kwargs):
    """Asin converter"""
    if kwargs:
        __unexpected_attrs("asin", kwargs)

    return relax.op.unary.asin(data)


def acos_converter(bbuilder, data, **kwargs):
    """Acos converter"""
    if kwargs:
        __unexpected_attrs("acos", kwargs)

    return relax.op.unary.acos(data)


def atan_converter(bbuilder, data, **kwargs):
    """Atan converter"""
    if kwargs:
        __unexpected_attrs("atan", kwargs)

    return relax.op.unary.atan(data)


def asinh_converter(bbuilder, data, **kwargs):
    """Asinh converter"""
    if kwargs:
        __unexpected_attrs("asinh", kwargs)

    return relax.op.unary.asinh(data)


def acosh_converter(bbuilder, data, **kwargs):
    """Acosh converter"""
    if kwargs:
        __unexpected_attrs("acosh", kwargs)

    return relax.op.unary.acosh(data)


def atanh_converter(bbuilder, data, **kwargs):
    """Atanh converter"""
    if kwargs:
        __unexpected_attrs("atanh", kwargs)

    return relax.op.unary.atanh(data)


def abs_converter(bbuilder, data, **kwargs):
    """Abs converter"""
    if kwargs:
        __unexpected_attrs("abs", kwargs)

    return relax.op.unary.abs(data)


def sign_converter(bbuilder, data, **kwargs):
    """Sign converter"""
    if kwargs:
        __unexpected_attrs("sign", kwargs)

    return relax.op.unary.sign(data)


def not_converter(bbuilder, data, **kwargs):
    """Not converter"""
    if kwargs:
        __unexpected_attrs("not", kwargs)

    return relax.op.unary.logical_not(data)


def floor_converter(bbuilder, data, **kwargs):
    """Floor converter"""
    if kwargs:
        __unexpected_attrs("floor", kwargs)

    return relax.op.unary.floor(data)


def ceil_converter(bbuilder, data, **kwargs):
    """Ceil converter"""
    if kwargs:
        __unexpected_attrs("ceil", kwargs)

    return relax.op.unary.ceil(data)


def round_converter(bbuilder, data, **kwargs):
    """Round converter"""
    if kwargs:
        __unexpected_attrs("round", kwargs)

    return relax.op.unary.round(data)


#   # Binary ops


def add_converter(bbuilder, lhs, rhs, **kwargs):
    """Add converter"""
    if kwargs:
        __unexpected_attrs("add", kwargs)

    return relax.op.binary.add(lhs, rhs)


def sub_converter(bbuilder, lhs, rhs, **kwargs):
    """Sub converter"""
    if kwargs:
        __unexpected_attrs("sub", kwargs)

    return relax.op.binary.subtract(lhs, rhs)


def mul_converter(bbuilder, lhs, rhs, **kwargs):
    """Mul converter"""
    if kwargs:
        __unexpected_attrs("mul", kwargs)

    lhs = bbuilder.normalize(lhs)
    rhs = bbuilder.normalize(rhs)

    l_ndim = len(lhs.struct_info.shape)
    r_ndim = len(rhs.struct_info.shape)

    if l_ndim > r_ndim > 0:
        rhs = relax.op.expand_dims(rhs, [d + 2 for d in range(l_ndim - r_ndim)])
    if r_ndim > l_ndim > 0:
        lhs = relax.op.expand_dims(lhs, [d + 2 for d in range(r_ndim - l_ndim)])

    return relax.op.binary.multiply(lhs, rhs)


def div_converter(bbuilder, lhs, rhs, **kwargs):
    """Div converter"""
    if kwargs:
        __unexpected_attrs("div", kwargs)

    return relax.op.binary.divide(lhs, rhs)


def pow_converter(bbuilder, lhs, rhs, **kwargs):
    """Pow converter"""
    if kwargs:
        __unexpected_attrs("pow", kwargs)

    return relax.op.binary.power(lhs, rhs)


def lt_converter(bbuilder, lhs, rhs, **kwargs):
    """Lt converter"""
    if kwargs:
        __unexpected_attrs("lt", kwargs)

    return relax.op.binary.less(lhs, rhs)


def gt_converter(bbuilder, lhs, rhs, **kwargs):
    """Gt converter"""
    if kwargs:
        __unexpected_attrs("gt", kwargs)

    return relax.op.binary.greater(lhs, rhs)


def le_converter(bbuilder, lhs, rhs, **kwargs):
    """Le converter"""
    if kwargs:
        __unexpected_attrs("le", kwargs)

    return relax.op.binary.less_equal(lhs, rhs)


def ge_converter(bbuilder, lhs, rhs, **kwargs):
    """Ge converter"""
    if kwargs:
        __unexpected_attrs("ge", kwargs)

    return relax.op.binary.greater_equal(lhs, rhs)


def eq_converter(bbuilder, lhs, rhs, **kwargs):
    """Eq converter"""
    if kwargs:
        __unexpected_attrs("eq", kwargs)

    return relax.op.binary.equal(lhs, rhs)


def ne_converter(bbuilder, lhs, rhs, **kwargs):
    """Ne converter"""
    if kwargs:
        __unexpected_attrs("ne", kwargs)

    return relax.op.binary.not_equal(lhs, rhs)


def and_converter(bbuilder, lhs, rhs, **kwargs):
    """And converter"""
    if kwargs:
        __unexpected_attrs("and", kwargs)

    return relax.op.binary.logical_and(lhs, rhs)


def or_converter(bbuilder, lhs, rhs, **kwargs):
    """Or converter"""
    if kwargs:
        __unexpected_attrs("or", kwargs)

    return relax.op.binary.logical_or(lhs, rhs)


#   # Select op


def select_converter(bbuilder, condition, t_val, f_val, **kwargs):
    """Select converter"""
    if kwargs:
        __unexpected_attrs("select", kwargs)

    return relax.op.where(condition, t_val, f_val)


#   # Simplifier ops


def sqr_converter(bbuilder, data, **kwargs):
    """sqr converter"""
    if kwargs:
        __unexpected_attrs("sqr", kwargs)

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    return pow_converter(bbuilder, data, tvm_expr.const(2.0, dtype=d_type))


def sqrt_converter(bbuilder, data, **kwargs):
    """sqrt converter"""
    if kwargs:
        __unexpected_attrs("sqrt", kwargs)

    return relax.op.unary.sqrt(data)


def rsqr_converter(bbuilder, data, **kwargs):
    """rsqr converter"""
    if kwargs:
        __unexpected_attrs("rsqr", kwargs)

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    return pow_converter(bbuilder, data, tvm_expr.const(-2.0, dtype=d_type))


def rsqrt_converter(bbuilder, data, **kwargs):
    """rsqrt converter"""
    if kwargs:
        __unexpected_attrs("rsqrt", kwargs)

    return relax.op.unary.rsqrt(data)


def log2_converter(bbuilder, data, **kwargs):
    """log2 converter"""
    if kwargs:
        __unexpected_attrs("log2", kwargs)

    # no equivalent in Relax, using TOpI
    return bbuilder.emit_te(topi.log2, data)


def min_converter(bbuilder, lhs, rhs, **kwargs):
    """Min converter"""
    if kwargs:
        __unexpected_attrs("min", kwargs)

    return relax.op.binary.minimum(lhs, rhs)


def max_converter(bbuilder, lhs, rhs, **kwargs):
    """Max converter"""
    if kwargs:
        __unexpected_attrs("max", kwargs)

    return relax.op.binary.maximum(lhs, rhs)


def clamp_converter(bbuilder, x, a, b, **kwargs):
    """Clamp converter"""
    if kwargs:
        __unexpected_attrs("clamp", kwargs)

    # only works if b and a are Constant floats, not tensors
    if isinstance(a, tvm_expr.Constant) and isinstance(b, tvm_expr.Constant):
        return relax.op.clip(
            x, tvm_expr.PrimValue(a.data.numpy().item()), tvm_expr.PrimValue(b.data.numpy().item())
        )

    return max_converter(bbuilder, min_converter(bbuilder, x, b), a)


#   # Sliding-window ops


def conv_converter(
    bbuilder, data, kernel, bias, border, stride, padding, dilation, groups, **kwargs
):
    """Convolution converter,
    skips bias if it's 0.0 (no bias)"""
    if kwargs:
        __unexpected_attrs("conv", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    kernel_shape = [v.value for v in kernel.struct_info.shape.values]
    dshape = [v.value for v in data.struct_info.shape.values]

    if hasattr(data.struct_info, "ndim"):
        ndim = data.struct_info.ndim
    else:
        ndim = len(data.struct_info.shape)

    strides = _stride_conv(stride, ndim) if stride else (1,) * (ndim - 2)

    dilation = _stride_conv(dilation, ndim) if dilation else (1,) * (ndim - 2)

    if not padding:
        padding = _calculate_nnef_padding(dshape[2:], strides, kernel_shape[2:], dilation)

    pad = _padding_conv(padding, ndim)

    channels = kernel_shape[0]

    if groups == 0:
        groups = channels

    if ndim == 3:
        op = relax.op.nn.conv1d
    elif ndim == 4:
        op = relax.op.nn.conv2d
    elif ndim == 5:
        op = relax.op.nn.conv3d
    else:
        raise NotImplementedError("Ndim > 5 not supported for convolution.")

    conv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
    )

    res = None
    if isinstance(bias, tvm_expr.Constant):
        # nnef has bias of 0 if it is not needed
        if (bias.data.numpy() == 0).all():
            res = conv_out

    if not res:
        bias = relax.op.reshape(
            bias,
            [1, -1]
            + [
                1,
            ]
            * (ndim - 2),
        )
        res = relax.op.add(conv_out, bias)

    return res


def deconv_converter(
    bbuilder, data, kernel, bias, border, stride, padding, dilation, output_shape, groups, **kwargs
):
    """Deconvolution converter, using convxd_transpose
    skips bias if it's 0.0 (no bias)"""
    if kwargs:
        __unexpected_attrs("deconv", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    kernel_shape = [v.value for v in kernel.struct_info.shape.values]

    rank = len(kernel_shape)

    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = _stride_conv(dilation, rank) if dilation else (1,) * (rank - 2)

    total, out_sh = _calculate_nnef_padding_deconv(
        [v.value for v in data.struct_info.shape.values],
        strides,
        kernel_shape,
        dilation,
        output_shape,
    )

    if padding:
        pad = _padding_conv(padding, rank)
    else:
        pad = _padding_conv([(pad // 2, (pad + 1) // 2) for pad in total], rank)

    if groups == 0:
        groups = kernel_shape[0]

    # limit output padding to modulo stride because of tvm checks
    out_pad = (
        [(x - (y - t)) % s for x, y, t, s in zip(output_shape[2:], out_sh, total, stride)]
        if output_shape
        else (0, 0)
    )

    if rank == 3:
        op = relax.op.nn.conv1d_transpose
    elif rank == 4:
        op = relax.op.nn.conv2d_transpose
    else:
        raise NotImplementedError("Ndim > 4 not supported for deconvolution. 3D WIP.")

    deconv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
        output_padding=out_pad,
    )

    res = None
    if isinstance(bias, tvm_expr.Constant):
        if (bias.data.numpy() == 0).all():
            res = deconv_out

    if not res:
        bias = relax.op.reshape(
            bias,
            [1, -1]
            + [
                1,
            ]
            * (rank - 2),
        )
        res = relax.op.add(deconv_out, bias)

    return res


def box_converter(bbuilder, data, size, border, padding, stride, dilation, normalize, **kwargs):
    """Box operator converter,
    summation over sliding window, equal to conv with constant filter"""
    if kwargs:
        __unexpected_attrs("box", kwargs)

    dshape = [v.value for v in data.struct_info.shape.values]

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    if size[:2] == [1, 1]:
        size[0] = dshape[1]
        if normalize:
            kernel = relax.op.full(size, relax.const(1 / math.prod(size[2:]), d_type), d_type)
        else:
            kernel = relax.op.ones(size, d_type)

        kernel = bbuilder.normalize(kernel)

        out = conv_converter(
            bbuilder,
            data,
            kernel,
            tvm_expr.const(0, dtype=d_type),
            border,
            stride,
            padding,
            dilation,
            dshape[1],
        )
    else:
        # if boxing on channel or batch dims avg pool can solve with permute
        # we need permute indexes with inactive shape + active shape format, so active at the back

        def _apply_permutation(items, perm):
            return [items[ind] for ind in perm]

        inactive = [i for i, s in enumerate(size) if s == 1]
        active = [i for i, s in enumerate(size) if s != 1]
        permuted_ins = inactive + active
        inverse = [0] * len(permuted_ins)
        for i, p in enumerate(permuted_ins):
            inverse[p] = i

        data = relax.op.permute_dims(data, permuted_ins)
        size = _apply_permutation(size, permuted_ins)

        data = bbuilder.normalize(data)

        out = avg_pool_converter(
            bbuilder, data, size[2:], border, padding, stride[2:], dilation[2:]
        )

        out = relax.op.permute_dims(out, inverse)

        if not normalize:
            out = bbuilder.normalize(out)
            out = mul_converter(
                bbuilder, out, tvm_expr.const(math.prod(size), dtype=out.struct_info.dtype)
            )

    return out


def debox_converter(
    bbuilder, data, size, border, padding, stride, dilation, normalize, output_shape, **kwargs
):
    """Debox operator converter,
    inverse of box, equal to deconv with constant filter"""
    if kwargs:
        __unexpected_attrs("debox", kwargs)

    dshape = [v.value for v in data.struct_info.shape.values]

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    size[0] = dshape[1]
    if normalize:
        kernel = relax.op.full(relax.const(1 / math.prod(size[2:]), d_type), size, d_type)
    else:
        kernel = relax.op.ones(size, d_type)

    kernel = bbuilder.normalize(kernel)

    out = deconv_converter(
        bbuilder,
        data,
        kernel,
        tvm_expr.const(0, dtype=d_type),
        border,
        stride,
        padding,
        dilation,
        output_shape,
        groups=dshape[1],
    )
    return out


def nearest_downsample_converter(bbuilder, data, factor, **kwargs):
    """Nearest neighbour downsample converter"""
    if kwargs:
        __unexpected_attrs("nearest_downsample", kwargs)

    dims = 2 + len(factor)

    return box_converter(
        bbuilder,
        data,
        size=[1] * dims,
        border="constant",
        padding=[(0, 0)] * dims,
        stride=[1, 1] + factor,
        dilation=(1,) * (dims - 2),
        normalize=False,
    )


def area_downsample_converter(bbuilder, data, factor, **kwargs):
    """Area downsample converter"""
    if kwargs:
        __unexpected_attrs("area_downsample", kwargs)

    dims = 2 + len(factor)

    return box_converter(
        bbuilder,
        data,
        size=[1, 1] + factor,
        border="constant",
        padding=[(0, 0)] * dims,
        stride=[1, 1] + factor,
        dilation=(1,) * (dims - 2),
        normalize=True,
    )


def nearest_upsample_converter(bbuilder, data, factor, **kwargs):
    """Nearest neighbour upsample converter"""
    if kwargs:
        __unexpected_attrs("nearest_upsample", kwargs)

    dshape = [v.value for v in data.struct_info.shape.values]
    new_size = [d * f for d, f in zip(dshape[2:], factor)]

    ndims = len(dshape)

    if ndims == 3:
        op = topi.image.resize1d
    if ndims == 4:
        op = topi.image.resize2d
    if ndims == 5:
        op = topi.image.resize3d

    return bbuilder.emit_te(
        op,
        data,
        [
            0,
        ]
        * ndims,  # dummy value so typecheck goes through, roi is not used
        new_size,
        method="nearest_neighbor",
        rounding_method="round",
    )


def multilinear_upsample_converter(bbuilder, data, factor, method, border, **kwargs):
    """Multilinear upsampling converter"""
    if kwargs:
        __unexpected_attrs("linear_upsample", kwargs)

    # for aligned and symmetric replicate resize can be used
    dshape = [v.value for v in data.struct_info.shape.values]
    ndims = len(dshape)

    if ndims == 3:
        op = topi.image.resize1d
    if ndims == 4:
        op = topi.image.resize2d
    if ndims == 5:
        op = topi.image.resize3d

    new_size = [d * f for d, f in zip(dshape[2:], factor)]
    if method == "aligned":
        # conversion from nn.upsampling to image.resizexd, re: discuss:11650
        return bbuilder.emit_te(
            op,
            data,
            [
                0,
            ]
            * ndims,  # dummy value so typecheck goes through, roi is not used
            new_size,
            method="linear",
            coordinate_transformation_mode="align_corners",
        )
    if method == "symmetric" and border == "replicate":
        return bbuilder.emit_te(
            op,
            data,
            [
                0,
            ]
            * ndims,  # dummy value so typecheck goes through, roi is not used
            new_size,
            method="linear",
            coordinate_transformation_mode="half_pixel",
        )

    # other combinations need to be calculated with convolution
    def _upsample_weights_1d(fact, symm):
        if symm:
            _weights = [1 - (i + 0.5) / fact for i in range(fact)]
            _weights = list(reversed(_weights)) + _weights
        else:
            _weights = [1 - abs(i) / float(fact) for i in range(-fact + 1, fact)]
        return np.array(_weights)

    def _upsample_weights_nd(fact, symm):
        _weights = [_upsample_weights_1d(f, symm) for f in fact]
        return reduce(np.multiply, np.ix_(*_weights))

    n, c = dshape[:2]

    symmetric = method == "symmetric"
    weights = _upsample_weights_nd(factor, symmetric)
    weights = np.reshape(weights, newshape=(1, 1) + weights.shape)
    kernel = tile_converter(bbuilder, tvm_expr.const(weights), (c, 1) + (1,) * len(factor))
    kernel = bbuilder.normalize(kernel)

    output_shape = [n, c] + [f * s for f, s in zip(factor, dshape[2:])]

    if symmetric:
        return deconv_converter(
            bbuilder,
            data,
            kernel,
            tvm_expr.const(0.0),
            border="constant",
            stride=factor,
            padding=[(f - 1, f - 1) for f in factor],
            dilation=[],
            groups=c,
            output_shape=output_shape,
        )
    else:
        replicate = border == "replicate"
        if replicate:
            data = pad_converter(
                bbuilder,
                data,
                [(0, 0), (0, 0)] + [(1, 0)] * len(factor),
                border,
                tvm_expr.const(0.0),
            )
            data = bbuilder.normalize(data)
            padding = factor
        else:
            padding = [f // 2 for f in factor]

        return deconv_converter(
            bbuilder,
            data,
            kernel,
            tvm_expr.const(0.0),
            border="constant",
            stride=factor,
            padding=[(p, p - 1) for p in padding],
            dilation=[],
            groups=c,
            output_shape=output_shape,
        )


#   # Reduce ops


def sum_reduce_converter(bbuilder, data, axes, normalize, keepdims=True, **kwargs):
    """Sum reduce converter"""

    if kwargs:
        __unexpected_attrs("sum_reduce", kwargs)

    out = relax.op.sum(data, axes, keepdims=keepdims)
    if normalize:
        return l2_normalization_converter(bbuilder, out, 0, [x - 2 for x in axes], 0.0)
    return out


def max_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Max reduce converter"""
    if kwargs:
        __unexpected_attrs("max_reduce", kwargs)

    return relax.op.max(data, axes, keepdims=keepdims)


def min_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Min reduce converter"""
    if kwargs:
        __unexpected_attrs("min_reduce", kwargs)

    return relax.op.min(data, axes, keepdims=keepdims)


def argmax_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Argmax reduce converter"""
    if kwargs:
        __unexpected_attrs("argmax_reduce", kwargs)

    # relax.op.argmax only supports singular axis, using TOpI
    return bbuilder.emit_te(topi.argmax, data, axes, keepdims=keepdims)


def argmin_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Argmin reduce converter"""
    if kwargs:
        __unexpected_attrs("argmin_reduce", kwargs)

    # relax.op.argmin only supports singular axis, using TOpI
    return bbuilder.emit_te(topi.argmin, data, axes, keepdims=keepdims)


def all_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """All reduce converter"""
    if kwargs:
        __unexpected_attrs("all_reduce", kwargs)

    # no equivalent in Relax, using TOpI
    return bbuilder.emit_te(topi.all, data, axes, keepdims)


def any_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Any reduce converter"""
    if kwargs:
        __unexpected_attrs("any_reduce", kwargs)

    # no equivalent in Relax, using TOpI
    return bbuilder.emit_te(topi.any, data, axes, keepdims)


def mean_reduce_converter(bbuilder, data, axes, keepdims=True, **kwargs):
    """Mean reduce converter"""
    if kwargs:
        __unexpected_attrs("mean_reduce", kwargs)

    return relax.op.mean(data, axes, keepdims=keepdims)


#   # Tensor shape ops


def reshape_converter(bbuilder, data, shape, axis_start, axis_count, **kwargs):
    """Reshape converter"""
    if kwargs:
        __unexpected_attrs("reshape", kwargs)

    dshape = [v.value for v in data.struct_info.shape.values]
    if axis_count == -1:
        newshape = dshape[:axis_start] + shape
    else:
        newshape = dshape
        newshape[axis_start : axis_start + axis_count] = shape

    return relax.op.reshape(data, newshape)


def squeeze_converter(bbuilder, data, axes, **kwargs):
    """Squeeze converter"""
    if kwargs:
        __unexpected_attrs("squeeze", kwargs)

    return relax.op.squeeze(data, axes)


def unsqueeze_converter(bbuilder, data, axes, **kwargs):
    """Unsqueeze converter"""
    if kwargs:
        __unexpected_attrs("unsqueeze", kwargs)

    axes = sorted(axes)
    for axis in axes:
        if axis < 0 and isinstance(data, tvm_expr.Var):
            axis = len(data.type_annotation.concrete_shape) + len(axes) + axis

        data = tvm_op.expand_dims(data, axis=axis)
    return data


def transpose_converter(bbuilder, data, axes, **kwargs):
    """Transpose converter"""
    if kwargs:
        __unexpected_attrs("transpose", kwargs)

    return relax.op.permute_dims(data, axes)


def split_converter(bbuilder, data, axis, ratios, **kwargs):
    """Split converter"""
    if kwargs:
        __unexpected_attrs("split", kwargs)

    axis_len = [v.value for v in data.struct_info.shape.values][axis]
    rat_mul = axis_len / sum(ratios)
    ratio_list = [(r * rat_mul) for r in ratios]

    s = 0
    indices = []
    for rat in ratio_list[:-1]:
        s += rat
        # Strictly needs int
        indices.append(int(s))

    return relax.op.split(data, indices, axis)


def concat_converter(bbuilder, *data, axis, **kwargs):
    """Concat converter"""
    if kwargs:
        __unexpected_attrs("concat", kwargs)

    return relax.op.concat(data, axis)


def stack_converter(bbuilder, *data, axis, **kwargs):
    """Stack converter"""
    if kwargs:
        __unexpected_attrs("stack", kwargs)

    data = [relax.op.expand_dims(d, axis) for d in data]

    return relax.op.concat(data, axis)


def unstack_converter(bbuilder, data, axis, **kwargs):
    """Unstack converter"""
    if kwargs:
        __unexpected_attrs("unstack", kwargs)

    split = split_converter(
        bbuilder, data, axis, [1] * [v.value for v in data.struct_info.shape.values][axis]
    )
    split = bbuilder.normalize(split)
    res = []

    for i in range(len(split.struct_info.fields)):
        res.append(squeeze_converter(bbuilder, split[i], axis))
    return tvm.relax.Tuple(relax.Tuple(res))


def slice_converter(bbuilder, data, axes, begin, end, stride, **kwargs):
    """Slice converter"""
    if kwargs:
        __unexpected_attrs("slice", kwargs)

    if not stride:
        stride = [1] * len(axes)

    return relax.op.strided_slice(data, begin=begin, end=end, strides=stride, axes=axes)


def pad_converter(bbuilder, data, padding, border, value, **kwargs):
    """Pad converter"""
    if kwargs:
        __unexpected_attrs("pad", kwargs)

    if border not in ["constant", "replicate", "reflect"]:
        print(f"{border} border type is not supported in padding. Assumed constant")
        border = "constant"
    if border == "replicate":
        border = "edge"

    # padding can only be tuple<int> even though docs say tuple<tuple<int>>
    pad = sum(padding, ())
    pad_before, pad_after = zip(*padding)

    # reflect can only work with TOPI mirror_pad
    if border == "reflect":
        return bbuilder.emit_te(tvm.topi.nn.mirror_pad, data, pad_before, pad_after, "REFLECT")
    if border == "edge":
        raise tvm.error.OpNotImplemented(
            "Replicate - Edge mode is currently not supported in TVM relax"
        )

    # constant works with normal relax.nn.pad
    return relax.op.nn.pad(data, pad, value, border)


def tile_converter(bbuilder, data, repeats, **kwargs):
    """Tile converter"""
    if kwargs:
        __unexpected_attrs("tile", kwargs)

    return relax.op.tile(data, repeats)


#   # Region-of-interest ops


#   # Matrix multiplication
def matmul_converter(bbuilder, a, b, **kwargs):
    """Matmul converter
    real signature: matmul_converter(a, b, transposeA, transposeB)"""

    transpose_a = kwargs.pop("transposeA")
    transpose_b = kwargs.pop("transposeB")
    if kwargs:
        __unexpected_attrs("matmul", kwargs)

    if transpose_a:
        ndim = len(a.struct_info.shape.values)
        axes = list(range(ndim - 2))
        axes.append(ndim - 1)
        axes.append(ndim - 2)
        a = relax.op.permute_dims(a, axes)

    if transpose_b:
        ndim = len(a.struct_info.shape.values)
        axes = list(range(ndim - 2))
        axes.append(ndim - 1)
        axes.append(ndim - 2)
        b = relax.op.permute_dims(b, axes)

    a = bbuilder.normalize(a)
    b = bbuilder.normalize(b)

    return relax.op.matmul(a, b)


#   # Variable updates
#   # Compound ops


def sigmoid_converter(bbuilder, data, **kwargs):
    """Sigmoid converter"""
    if kwargs:
        __unexpected_attrs("sigmoid", kwargs)

    return relax.op.unary.sigmoid(data)


def relu_converter(bbuilder, data, **kwargs):
    """RELU converter"""
    if kwargs:
        __unexpected_attrs("relu", kwargs)

    return relax.op.nn.relu(data)


def prelu_converter(bbuilder, data, alpha, **kwargs):
    """PRELU converter"""
    if kwargs:
        __unexpected_attrs("prelu", kwargs)

    # prelu can't handle float vals but NNEF supports direct parameter, this is just in case
    if isinstance(alpha, tvm_expr.Constant):
        if alpha.data.numpy().size == 1:
            return relax.op.nn.leakyrelu(data, alpha.data.numpy().item())

    # alpha needs to be a tensor whose rank is the same as of data,
    # and the only non 1 dim is the channel dims
    axes = [
        0,
    ] + [a + 2 for a in range(data.struct_info.ndim - 2)]
    alpha = relax.op.expand_dims(alpha, axes)

    # using select for prelu
    return select_converter(
        bbuilder, data < tvm_expr.const(0.0), mul_converter(bbuilder, alpha, data), data
    )


def leaky_relu_converter(bbuilder, data, alpha, **kwargs):
    """Leaky RELU converter"""
    if kwargs:
        __unexpected_attrs("leaky_relu", kwargs)

    return relax.op.nn.leakyrelu(data, alpha)


def elu_converter(bbuilder, data, alpha, **kwargs):
    """ELU converter"""
    if kwargs:
        __unexpected_attrs("elu", kwargs)

    return select_converter(
        bbuilder,
        lt_converter(bbuilder, data, tvm_expr.const(0.0)),
        mul_converter(
            bbuilder,
            tvm_expr.const(alpha),
            sub_converter(bbuilder, exp_converter(bbuilder, data), tvm_expr.const(1.0)),
        ),
        data,
    )


def selu_converter(bbuilder, data, alpha, **kwargs):
    """SELU converter
    True signature is selu_converter(data, alpha, lambda)"""
    lambda_var = kwargs.pop("lambda")

    if kwargs:
        __unexpected_attrs("selu", kwargs)

    return mul_converter(
        bbuilder,
        tvm_expr.const(lambda_var),
        select_converter(
            bbuilder,
            data < tvm_expr.const(0.0),
            mul_converter(
                bbuilder,
                tvm_expr.const(alpha),
                sub_converter(bbuilder, exp_converter(bbuilder, data), tvm_expr.const(1.0)),
            ),
            data,
        ),
    )


def gelu_converter(bbuilder, data, **kwargs):
    """GELU converter
    NNEF definition for GELU:
    the exact definition of GELU is x * Phi(x) where Phi(x) is the
    CDF of the standard normal distribution, which can be approximated
    for example by sigmoid(1.702 * x)

    `mul_converter(data, sigmoid_converter(mul_converter(tvm_expr.const(1.702), data)))`

    But in this case we will use the erf to calculate normcdf (same as to pytorch GELU impl)
    """
    if kwargs:
        __unexpected_attrs("gelu", kwargs)

    return relax.op.nn.gelu(data)


def silu_converter(bbuilder, data, **kwargs):
    """SiLU converter"""
    if kwargs:
        __unexpected_attrs("silu", kwargs)

    return mul_converter(bbuilder, data, sigmoid_converter(bbuilder, data))


def softmax_converter(bbuilder, data, axes, **kwargs):
    """Softmax converter"""
    if kwargs:
        __unexpected_attrs("softmax", kwargs)

    if len(axes) > 1:
        print("Multiple axes not supported, operation has been done along the first axis in axes.")
    axis = axes[0]

    return relax.op.nn.softmax(data, axis)


def softplus_converter(bbuilder, data, **kwargs):
    """Softplus converter"""
    if kwargs:
        __unexpected_attrs("softplus", kwargs)

    return log_converter(
        bbuilder, add_converter(bbuilder, exp_converter(bbuilder, data), tvm_expr.const(1.0))
    )


#   # linear ops


def linear_converter(bbuilder, data, _filter, bias, **kwargs):
    """Linear converter"""
    if kwargs:
        __unexpected_attrs("linear", kwargs)

    out = matmul_converter(bbuilder, data, _filter, transposeA=False, transposeB=True)
    out = bbuilder.normalize(out)
    res = None

    if isinstance(bias, tvm_expr.Constant):
        if (bias.data.numpy() == 0).all():
            res = out

    if hasattr(data.struct_info, "ndim"):
        ndim = data.struct_info.ndim
    else:
        ndim = len(data.struct_info.shape)

    if not res:
        bias = relax.op.reshape(
            bias,
            [1, -1]
            + [
                1,
            ]
            * (ndim - 2),
        )
        res = relax.op.add(out, bias)

    return res


def separable_conv_converter(
    bbuilder,
    data,
    plane_filter,
    point_filter,
    bias,
    border,
    padding,
    stride,
    dilation,
    groups,
    **kwargs,
):
    """Separable convolution converter"""
    if kwargs:
        __unexpected_attrs("separable_conv", kwargs)

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    filtered = conv_converter(
        bbuilder,
        data,
        plane_filter,
        tvm_expr.const(0, dtype=d_type),
        border,
        stride,
        padding,
        dilation,
        0,
    )

    filtered = bbuilder.normalize(filtered)

    return conv_converter(bbuilder, filtered, point_filter, bias, "constant", [], [], [], groups)


def separable_deconv_converter(
    bbuilder,
    data,
    plane_filter,
    point_filter,
    bias,
    border,
    padding,
    stride,
    dilation,
    output_shape,
    groups,
    **kwargs,
):
    """Separable deconvolution converter"""
    if kwargs:
        __unexpected_attrs("separable_deconv", kwargs)

    if isinstance(data, relax.Call):
        d_type = data.checked_type.dtype
    else:
        d_type = data.struct_info.dtype

    filtered = deconv_converter(
        bbuilder,
        data,
        point_filter,
        tvm_expr.const(0, dtype=d_type),
        "constant",
        [],
        [],
        [],
        [],
        groups,
    )

    filtered = bbuilder.normalize(filtered)

    return deconv_converter(
        bbuilder, filtered, plane_filter, bias, border, stride, padding, dilation, output_shape, 0
    )


def max_pool_converter(bbuilder, data, size, border, padding, stride, dilation, **kwargs):
    """Max pool converter"""
    if kwargs:
        __unexpected_attrs("max_pool", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    dshape = [v.value for v in data.struct_info.shape.values]
    rank = len(dshape)

    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = _stride_conv(dilation, rank) if dilation else (1,) * (rank - 2)

    if not padding:
        # padding is truncated to `conv style` (only active layers are present)
        padding = _calculate_nnef_padding(dshape[2:], strides, pool_size, dilation)

    pad = _padding_conv(padding, rank)

    if border == "constant":
        padding = [(0, 0), (0, 0)] + padding
        data = pad_converter(bbuilder, data, padding, border, tvm_expr.const(0.0))
        data = bbuilder.normalize(data)
        pad = (0, 0)

    if rank == 3:
        op = relax.op.nn.max_pool1d
    elif rank == 4:
        op = relax.op.nn.max_pool2d
    elif rank == 5:
        op = relax.op.nn.max_pool3d
    else:
        raise NotImplementedError("Ndim > 5 not supported for max pool.")

    return op(
        data,
        pool_size=pool_size,
        strides=strides,
        dilation=dilation,
        padding=pad,
    )


def avg_pool_converter(bbuilder, data, size, border, padding, stride, dilation, **kwargs):
    """Avg pool converter"""
    if kwargs:
        __unexpected_attrs("avg_pool", kwargs)

    if border not in ["constant", "ignore"]:
        print(f"Currently {border} border is not supported, used `constant` border")

    dshape = [v.value for v in data.struct_info.shape.values]
    rank = len(dshape)
    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = _stride_conv(dilation, rank) if dilation else (1,) * (rank - 2)

    # padding is truncated to `conv style` (only active layers are present)
    active_shape = dshape[2:]
    if not padding:
        padding = _calculate_nnef_padding(active_shape, strides, pool_size, dilation)

    pad = _padding_conv(padding, rank)

    if rank == 3:
        op = relax.op.nn.avg_pool1d
    elif rank == 4:
        op = relax.op.nn.avg_pool2d
    elif rank == 5:
        op = relax.op.nn.avg_pool3d
    else:
        raise NotImplementedError("Ndim > 5 not supported for avg pool.")

    return op(
        data,
        pool_size=pool_size,
        strides=strides,
        dilation=dilation,
        padding=pad,
        count_include_pad=border != "ignore",
    )


def rms_pool_converter(bbuilder, data, size, border, padding, stride, dilation, **kwargs):
    """Rms pool converter"""
    if kwargs:
        __unexpected_attrs("rms_pool", kwargs)

    return sqrt_converter(
        bbuilder,
        avg_pool_converter(
            bbuilder,
            bbuilder.normalize(sqr_converter(bbuilder, data)),
            size=size,
            border=border,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ),
    )


#   # Normalization


def local_response_normalization_converter(bbuilder, data, size, alpha, beta, bias, **kwargs):
    """LRN converter"""
    if kwargs:
        __unexpected_attrs("local_response_normalization", kwargs)

    axis = [i for i in range(len(size)) if size[i] > 1]
    if len(axis) == 1:
        axis = axis[0]
    else:
        print("Multi axis LRN is not implemented properly, using first axis where size != 1")
        axis = axis[0]
    size = size[axis]

    return bbuilder.emit_te(topi.nn.lrn, data, size, axis, alpha, beta, bias)


def local_mean_normalization_converter(bbuilder, data, size, **kwargs):
    """LMN converter"""
    if kwargs:
        __unexpected_attrs("local_mean_normalization", kwargs)

    mean = box_converter(bbuilder, data, size, "constant", [], [], [], normalize=True)
    mean = bbuilder.normalize(mean)

    return sub_converter(bbuilder, data, mean)


def local_variance_normalization_converter(bbuilder, data, size, bias, epsilon, **kwargs):
    """LVN converter"""
    if kwargs:
        __unexpected_attrs("local_variance_normalization", kwargs)

    sigma = box_converter(
        bbuilder,
        bbuilder.normalize(sqr_converter(bbuilder, data)),
        size,
        "constant",
        [],
        [],
        [],
        normalize=True,
    )
    sigma = bbuilder.normalize(sigma)

    return div_converter(
        bbuilder,
        data,
        max_converter(
            bbuilder,
            add_converter(bbuilder, sqrt_converter(bbuilder, sigma), tvm_expr.const(bias)),
            tvm_expr.const(epsilon),
        ),
    )


def local_contrast_normalization_converter(bbuilder, data, size, bias, epsilon, **kwargs):
    """LCN converter"""
    if kwargs:
        __unexpected_attrs("local_contrast_normalization", kwargs)

    centered = local_mean_normalization_converter(bbuilder, data, size)
    centered = bbuilder.normalize(centered)
    return local_variance_normalization_converter(bbuilder, centered, size, bias, epsilon)


def l1_normalization_converter(bbuilder, data, axes, bias, epsilon, **kwargs):
    """L1 norm converter"""
    if kwargs:
        __unexpected_attrs("l1_normalization", kwargs)

    sigma = sum_reduce_converter(bbuilder, abs_converter(bbuilder, data), axes, False)
    return div_converter(
        bbuilder,
        data,
        max_converter(
            bbuilder, add_converter(bbuilder, sigma, tvm_expr.const(bias)), tvm_expr.const(epsilon)
        ),
    )


def l2_normalization_converter(bbuilder, data, axes, bias, epsilon, **kwargs):
    """L2 norm converter"""
    if kwargs:
        __unexpected_attrs("l2_normalization", kwargs)

    # relay style l2_norm not supported, used equation from NNEF

    sigma = sum_reduce_converter(
        bbuilder, sqr_converter(bbuilder, data), axes=axes, normalize=False
    )

    res = div_converter(
        bbuilder,
        data,
        max_converter(
            bbuilder,
            add_converter(bbuilder, sqrt_converter(bbuilder, sigma), tvm_expr.const(bias)),
            tvm_expr.const(epsilon),
        ),
    )
    return res


def batch_normalization_converter(bbuilder, data, mean, variance, offset, scale, epsilon, **kwargs):
    """Batch norm converter"""
    if kwargs:
        __unexpected_attrs("batch_normalization", kwargs)

    mean = squeeze_converter(bbuilder, mean, 0)
    variance = squeeze_converter(bbuilder, variance, 0)
    offset = squeeze_converter(bbuilder, offset, 0)
    scale = squeeze_converter(bbuilder, scale, 0)

    mean = bbuilder.normalize(mean)
    variance = bbuilder.normalize(variance)
    offset = bbuilder.normalize(offset)
    scale = bbuilder.normalize(scale)

    res = bbuilder.emit_te(topi.nn.batch_norm, data, scale, offset, mean, variance, 1, epsilon)
    return res[0]


#   # Misc ops
