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

"""NNEF frontend converter helper functions"""
import math

import itertools
from functools import reduce

import numpy as np

import tvm
from tvm import relay
from tvm.relay import expr as tvm_expr
from tvm.relay import op as tvm_op
from tvm.relay.frontend.common import get_relay_op, infer_shape, infer_type


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


def ndop(*args, **kwargs):
    # not implemented ops
    raise Exception("Not supported operator was called, please check for compatibility")


def copy_converter(data, **kwargs):
    """Copy converter"""
    if kwargs:
        __unexpected_attrs("copy", kwargs)

    return get_relay_op("copy")(data)


def neg_converter(data, **kwargs):
    """Neg converter"""
    if kwargs:
        __unexpected_attrs("neg", kwargs)

    return get_relay_op("negative")(data)


def rcp_converter(data, **kwargs):
    """Rcp converter"""
    if kwargs:
        __unexpected_attrs("rcp", kwargs)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    return div_converter(tvm_expr.const(1, dtype=d_type), data)


def exp_converter(data, **kwargs):
    """Exp converter"""
    if kwargs:
        __unexpected_attrs("exp", kwargs)

    return get_relay_op("exp")(data)


def log_converter(data, **kwargs):
    """Log converter"""
    if kwargs:
        __unexpected_attrs("log", kwargs)

    return get_relay_op("log")(data)


def sin_converter(data, **kwargs):
    """Sin converter"""
    if kwargs:
        __unexpected_attrs("sin", kwargs)

    return get_relay_op("sin")(data)


def cos_converter(data, **kwargs):
    """Cos converter"""
    if kwargs:
        __unexpected_attrs("cos", kwargs)

    return get_relay_op("cos")(data)


def tan_converter(data, **kwargs):
    """Tan converter"""
    if kwargs:
        __unexpected_attrs("tan", kwargs)

    return get_relay_op("tan")(data)


def sinh_converter(data, **kwargs):
    """Sinh converter"""
    if kwargs:
        __unexpected_attrs("sinh", kwargs)

    return get_relay_op("sinh")(data)


def cosh_converter(data, **kwargs):
    """Cosh converter"""
    if kwargs:
        __unexpected_attrs("cosh", kwargs)

    return get_relay_op("cosh")(data)


def tanh_converter(data, **kwargs):
    """Tanh converter"""
    if kwargs:
        __unexpected_attrs("tanh", kwargs)

    return get_relay_op("tanh")(data)


def asin_converter(data, **kwargs):
    """Asin converter"""
    if kwargs:
        __unexpected_attrs("asin", kwargs)

    return get_relay_op("asin")(data)


def acos_converter(data, **kwargs):
    """Acos converter"""
    if kwargs:
        __unexpected_attrs("acos", kwargs)

    return get_relay_op("acos")(data)


def atan_converter(data, **kwargs):
    """Atan converter"""
    if kwargs:
        __unexpected_attrs("atan", kwargs)

    return get_relay_op("atan")(data)


def asinh_converter(data, **kwargs):
    """Asinh converter"""
    if kwargs:
        __unexpected_attrs("asinh", kwargs)

    return get_relay_op("asinh")(data)


def acosh_converter(data, **kwargs):
    """Acosh converter"""
    if kwargs:
        __unexpected_attrs("acosh", kwargs)

    return get_relay_op("acosh")(data)


def atanh_converter(data, **kwargs):
    """Atanh converter"""
    if kwargs:
        __unexpected_attrs("atanh", kwargs)

    return get_relay_op("atanh")(data)


def abs_converter(data, **kwargs):
    """Abs converter"""
    if kwargs:
        __unexpected_attrs("abs", kwargs)

    return get_relay_op("abs")(data)


def sign_converter(data, **kwargs):
    """Sign converter"""
    if kwargs:
        __unexpected_attrs("sign", kwargs)

    return get_relay_op("sign")(data)


def not_converter(data, **kwargs):
    """Not converter"""
    if kwargs:
        __unexpected_attrs("not", kwargs)

    return get_relay_op("logical_not")(data)


def floor_converter(data, **kwargs):
    """Floor converter"""
    if kwargs:
        __unexpected_attrs("floor", kwargs)

    return get_relay_op("floor")(data)


def ceil_converter(data, **kwargs):
    """Ceil converter"""
    if kwargs:
        __unexpected_attrs("ceil", kwargs)

    return get_relay_op("ceil")(data)


def round_converter(data, **kwargs):
    """Round converter"""
    if kwargs:
        __unexpected_attrs("round", kwargs)

    return get_relay_op("round")(data)


def add_converter(lhs, rhs, **kwargs):
    """Add converter"""
    if kwargs:
        __unexpected_attrs("add", kwargs)

    return get_relay_op("add")(lhs, rhs)


def sub_converter(lhs, rhs, **kwargs):
    """Sub converter"""
    if kwargs:
        __unexpected_attrs("sub", kwargs)

    return get_relay_op("subtract")(lhs, rhs)


def mul_converter(lhs, rhs, **kwargs):
    """Mul converter"""
    if kwargs:
        __unexpected_attrs("mul", kwargs)

    return get_relay_op("multiply")(lhs, rhs)


def div_converter(lhs, rhs, **kwargs):
    """Div converter"""
    if kwargs:
        __unexpected_attrs("div", kwargs)

    return get_relay_op("divide")(lhs, rhs)


def pow_converter(lhs, rhs, **kwargs):
    """Pow converter"""
    if kwargs:
        __unexpected_attrs("pow", kwargs)

    return get_relay_op("power")(lhs, rhs)


def lt_converter(lhs, rhs, **kwargs):
    """Lt converter"""
    if kwargs:
        __unexpected_attrs("lt", kwargs)

    return get_relay_op("less")(lhs, rhs)


def gt_converter(lhs, rhs, **kwargs):
    """Gt converter"""
    if kwargs:
        __unexpected_attrs("gt", kwargs)

    return get_relay_op("greater")(lhs, rhs)


def le_converter(lhs, rhs, **kwargs):
    """Le converter"""
    if kwargs:
        __unexpected_attrs("le", kwargs)

    return get_relay_op("less_equal")(lhs, rhs)


def ge_converter(lhs, rhs, **kwargs):
    """Ge converter"""
    if kwargs:
        __unexpected_attrs("ge", kwargs)

    return get_relay_op("greater_equal")(lhs, rhs)


def eq_converter(lhs, rhs, **kwargs):
    """Eq converter"""
    if kwargs:
        __unexpected_attrs("eq", kwargs)

    return get_relay_op("equal")(lhs, rhs)


def ne_converter(lhs, rhs, **kwargs):
    """Ne converter"""
    if kwargs:
        __unexpected_attrs("ne", kwargs)

    return get_relay_op("not_equal")(lhs, rhs)


def and_converter(lhs, rhs, **kwargs):
    """And converter"""
    if kwargs:
        __unexpected_attrs("and", kwargs)

    return get_relay_op("logical_and")(lhs, rhs)


def or_converter(lhs, rhs, **kwargs):
    """Or converter"""
    if kwargs:
        __unexpected_attrs("or", kwargs)

    return get_relay_op("logical_or")(lhs, rhs)


def select_converter(condition, t_val, f_val, **kwargs):
    """Select converter"""
    if kwargs:
        __unexpected_attrs("select", kwargs)

    return get_relay_op("where")(condition, t_val, f_val)


def sqr_converter(data, **kwargs):
    """sqr converter"""
    if kwargs:
        __unexpected_attrs("sqr", kwargs)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    return get_relay_op("power")(data, tvm_expr.const(2.0, dtype=d_type))


def sqrt_converter(data, **kwargs):
    """sqrt converter"""
    if kwargs:
        __unexpected_attrs("sqrt", kwargs)

    return get_relay_op("sqrt")(data)


def rsqr_converter(data, **kwargs):
    """rsqr converter"""
    if kwargs:
        __unexpected_attrs("rsqr", kwargs)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    return get_relay_op("power")(data, tvm_expr.const(-2.0, dtype=d_type))


def rsqrt_converter(data, **kwargs):
    """rsqrt converter"""
    if kwargs:
        __unexpected_attrs("rsqrt", kwargs)

    return get_relay_op("rsqrt")(data)


def log2_converter(data, **kwargs):
    """log2 converter"""
    if kwargs:
        __unexpected_attrs("log2", kwargs)

    return get_relay_op("log2")(data)


def min_converter(lhs, rhs, **kwargs):
    """Min converter"""
    if kwargs:
        __unexpected_attrs("min", kwargs)

    return get_relay_op("minimum")(lhs, rhs)


def max_converter(lhs, rhs, **kwargs):
    """Max converter"""
    if kwargs:
        __unexpected_attrs("max", kwargs)

    return get_relay_op("maximum")(lhs, rhs)


def clamp_converter(x, a, b, **kwargs):
    """Clamp converter"""
    if kwargs:
        __unexpected_attrs("clamp", kwargs)

    # only works if b and a are Constant floats, not tensors
    if isinstance(a, tvm_expr.Constant) and isinstance(b, tvm_expr.Constant):
        return get_relay_op("clip")(x, float(a.data.numpy()), float(b.data.numpy()))

    return max_converter(min_converter(x, b), a)


def conv_converter(data, kernel, bias, border, stride, padding, dilation, groups, **kwargs):
    """Convolution converter,
    skips bias if it's 0.0 (no bias)"""
    if kwargs:
        __unexpected_attrs("conv", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    kernel_shape = infer_shape(kernel)
    dshape = infer_shape(data)

    strides = _stride_conv(stride, len(kernel_shape)) if stride else (1,) * (len(kernel_shape) - 2)

    dilation = dilation if dilation else ((1,) * (len(kernel_shape) - 2))

    if not padding:
        padding = _calculate_nnef_padding(dshape[2:], strides, kernel_shape[2:], dilation)

    pad = _padding_conv(padding, len(kernel_shape))

    channels = kernel_shape[0]

    if groups == 0:
        groups = channels

    op = get_relay_op(dimension_picker("conv", kernel_shape))
    conv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_shape[2:],
    )

    res = None
    if isinstance(bias, tvm_expr.Constant):
        # nnef has bias of 0 if it is not needed
        if (bias.data.numpy() == 0).all():
            res = conv_out

    if not res:
        # squeeze needed as nnef has bias of shape [1, channel]
        res = tvm_op.nn.bias_add(conv_out, relay.squeeze(bias, axis=0))

    return res


def deconv_converter(
    data, kernel, bias, border, stride, padding, dilation, output_shape, groups, **kwargs
):
    """Deconvolution converter, using convxd_transpose
    skips bias if it's 0.0 (no bias)"""
    if kwargs:
        __unexpected_attrs("deconv", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    kernel_shape = infer_shape(kernel)

    rank = len(kernel_shape)

    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = dilation if dilation else ((1,) * (rank - 2))

    total, out_sh = _calculate_nnef_padding_deconv(
        infer_shape(data), strides, kernel_shape, dilation, output_shape
    )

    if padding:
        pad = _padding_conv(padding, rank)
    else:
        pad = _padding_conv([(pad // 2, (pad + 1) // 2) for pad in total], rank)

    if groups == 0:
        groups = kernel_shape[0]
    channels = kernel_shape[1] * groups

    # limit output padding to modulo stride because of tvm checks
    out_pad = (
        [(x - (y - t)) % s for x, y, t, s in zip(output_shape[2:], out_sh, total, stride)]
        if output_shape
        else (0, 0)
    )

    op = get_relay_op(dimension_picker("conv", kernel_shape, suffix="_transpose"))
    deconv_out = op(
        data=data,
        weight=kernel,
        strides=strides,
        padding=pad,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_shape[2:],
        output_padding=out_pad,
    )

    res = None
    if isinstance(bias, tvm_expr.Constant):
        if bias.data.numpy() == np.array([0.0]):
            res = deconv_out

    if not res:
        # squeeze needed bc nnef has bias of shape [1, channel]
        res = tvm_op.nn.bias_add(deconv_out, relay.squeeze(bias, axis=0))

    return res


def box_converter(data, size, border, padding, stride, dilation, normalize, **kwargs):
    """Box operator converter,
    summation over sliding window, equal to conv with constant filter"""
    if kwargs:
        __unexpected_attrs("box", kwargs)

    dshape = infer_shape(data)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    size[0] = dshape[1]
    if normalize:
        kernel = relay.full(tvm_op.const(1 / math.prod(size[2:]), d_type), size, d_type)
    else:
        kernel = relay.ones(size, d_type)

    out = conv_converter(
        data, kernel, tvm_expr.const(0, dtype=d_type), border, stride, padding, dilation, dshape[1]
    )
    return out


def debox_converter(
    data, size, border, padding, stride, dilation, normalize, output_shape, **kwargs
):
    """Debox operator converter,
    inverse of box, equal to deconv with constant filter"""
    if kwargs:
        __unexpected_attrs("debox", kwargs)

    dshape = infer_shape(data)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    size[0] = dshape[1]
    if normalize:
        kernel = relay.full(tvm_op.const(1 / math.prod(size[2:]), d_type), size, d_type)
    else:
        kernel = relay.ones(size, d_type)
    out = deconv_converter(
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


def nearest_downsample_converter(data, factor, **kwargs):
    """Nearest neighbour downsample converter"""
    if kwargs:
        __unexpected_attrs("nearest_downsample", kwargs)

    dims = 2 + len(factor)

    return box_converter(
        data,
        size=[1] * dims,
        border="constant",
        padding=[(0, 0)] * dims,
        stride=[1, 1] + factor,
        dilation=(1,) * (dims - 2),
        normalize=False,
    )


def area_downsample_converter(data, factor, **kwargs):
    """Area downsample converter"""
    if kwargs:
        __unexpected_attrs("area_downsample", kwargs)

    dims = 2 + len(factor)

    return box_converter(
        data,
        size=[1, 1] + factor,
        border="constant",
        padding=[(0, 0)] * dims,
        stride=[1, 1] + factor,
        dilation=(1,) * (dims - 2),
        normalize=True,
    )


def nearest_upsample_converter(data, factor, **kwargs):
    """Nearest neighbour upsample converter"""
    if kwargs:
        __unexpected_attrs("nearest_upsample", kwargs)

    # conversion from nn.upsampling to image.resizexd, re: discuss:11650
    #
    dshape = infer_shape(data)
    new_size = [d * f for d, f in zip(dshape[2:], factor)]
    return get_relay_op(dimension_picker("resize", dshape))(
        data,
        new_size,
        method="nearest_neighbor",
        # coordinate_transformation_mode="asymmetric",
        rounding_method="round",
    )


def multilinear_upsample_converter(data, factor, method, border, **kwargs):
    """Multilinear upsampling converter"""
    if kwargs:
        __unexpected_attrs("linear_upsample", kwargs)

    # for aligned and symmetric replicate resize can be used
    dshape = infer_shape(data)
    new_size = [d * f for d, f in zip(dshape[2:], factor)]
    if method == "aligned":
        # conversion from nn.upsampling to image.resizexd, re: discuss:11650
        return get_relay_op(dimension_picker("resize", dshape))(
            data,
            new_size,
            method="linear",
            coordinate_transformation_mode="align_corners",
        )
    if method == "symmetric" and border == "replicate":
        return get_relay_op(dimension_picker("resize", dshape))(
            data,
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
    kernel = tile_converter(tvm_expr.const(weights), (c, 1) + (1,) * len(factor))

    output_shape = [n, c] + [f * s for f, s in zip(factor, dshape[2:])]

    if symmetric:
        return deconv_converter(
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
                data, [(0, 0), (0, 0)] + [(1, 0)] * len(factor), border, tvm_expr.const(0.0)
            )
            padding = factor
        else:
            padding = [f // 2 for f in factor]

        return deconv_converter(
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


def sum_reduce_converter(data, axes, normalize, keepdims=True, **kwargs):
    """Sum reduce converter"""

    if kwargs:
        __unexpected_attrs("sum_reduce", kwargs)

    out = get_relay_op("sum")(data, axes, keepdims=keepdims)
    if normalize:
        return l2_normalization_converter(out, 0, [x - 2 for x in axes], 0.0)
    return out


def max_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Max reduce converter"""
    if kwargs:
        __unexpected_attrs("max_reduce", kwargs)

    return get_relay_op("max")(data, axes, keepdims=keepdims)


def min_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Min reduce converter"""
    if kwargs:
        __unexpected_attrs("min_reduce", kwargs)

    return get_relay_op("min")(data, axes, keepdims=keepdims)


def argmax_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Argmax reduce converter"""
    if kwargs:
        __unexpected_attrs("argmax_reduce", kwargs)

    return get_relay_op("argmax")(data, axes, keepdims=keepdims)


def argmin_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Argmin reduce converter"""
    if kwargs:
        __unexpected_attrs("argmin_reduce", kwargs)

    return get_relay_op("argmin")(data, axes, keepdims=keepdims)


def all_reduce_converter(data, axes, keepdims=True, **kwargs):
    """All reduce converter"""
    if kwargs:
        __unexpected_attrs("all_reduce", kwargs)

    return get_relay_op("all")(data, axes, keepdims=keepdims)


def any_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Any reduce converter"""
    if kwargs:
        __unexpected_attrs("any_reduce", kwargs)

    return get_relay_op("any")(data, axes, keepdims=keepdims)


def mean_reduce_converter(data, axes, keepdims=True, **kwargs):
    """Mean reduce converter"""
    if kwargs:
        __unexpected_attrs("mean_reduce", kwargs)

    return get_relay_op("mean")(data, axes, keepdims=keepdims)


def reshape_converter(data, shape, axis_start, axis_count, **kwargs):
    """Reshape converter"""
    if kwargs:
        __unexpected_attrs("reshape", kwargs)

    dshape = list(infer_shape(data))
    if axis_count == -1:
        newshape = dshape[:axis_start] + shape
    else:
        newshape = dshape
        newshape[axis_start : axis_start + axis_count] = shape

    return get_relay_op("reshape")(data, newshape)


def squeeze_converter(data, axes, **kwargs):
    """Squeeze converter"""
    if kwargs:
        __unexpected_attrs("squeeze", kwargs)
    return relay.squeeze(data, axes)


def unsqueeze_converter(data, axes, **kwargs):
    """Unsqueeze converter"""
    if kwargs:
        __unexpected_attrs("unsqueeze", kwargs)

    axes = sorted(axes)
    for axis in axes:
        if axis < 0 and isinstance(data, tvm_expr.Var):
            axis = len(data.type_annotation.concrete_shape) + len(axes) + axis

        data = tvm_op.expand_dims(data, axis=axis, num_newaxis=1)
    return data


def transpose_converter(data, axes, **kwargs):
    """Transpose converter"""
    if kwargs:
        __unexpected_attrs("transpose", kwargs)

    return get_relay_op("transpose")(data, axes)


def split_converter(data, axis, ratios, **kwargs):
    """Split converter"""
    if kwargs:
        __unexpected_attrs("split", kwargs)

    axis_len = infer_shape(data)[axis]
    rat_mul = axis_len / sum(ratios)
    ratio_list = [(r * rat_mul) for r in ratios]

    s = 0
    indices = []
    for rat in ratio_list[:-1]:
        s += rat
        # Strictly needs int
        indices.append(int(s))

    return get_relay_op("split")(data, indices, axis)


def concat_converter(*data, axis, **kwargs):
    """Concat converter"""
    if kwargs:
        __unexpected_attrs("concat", kwargs)

    return get_relay_op("concatenate")(data, axis)


def stack_converter(*data, axis, **kwargs):
    """Stack converter"""
    if kwargs:
        __unexpected_attrs("stack", kwargs)

    return get_relay_op("stack")(data, axis)


def unstack_converter(data, axis, **kwargs):
    """Unstack converter"""
    if kwargs:
        __unexpected_attrs("unstack", kwargs)

    split = split_converter(data, axis, [1] * infer_shape(data)[axis])
    res = []
    for i in range(len(split)):
        res.append(squeeze_converter(split[i], axis))
    return tvm_expr.TupleWrapper(relay.Tuple(res), len(res))


def slice_converter(data, axes, begin, end, stride, **kwargs):
    """Slice converter"""
    if kwargs:
        __unexpected_attrs("slice", kwargs)

    if not stride:
        stride = [1] * len(axes)

    return get_relay_op("strided_slice")(data, begin, end, strides=stride, axes=axes)


def pad_converter(data, padding, border, value, **kwargs):
    """Pad converter"""
    if kwargs:
        __unexpected_attrs("pad", kwargs)

    if border not in ["constant", "replicate", "reflect"]:
        print(f"{border} border type is not supported in padding. Assumed constant")
        border = "constant"
    if border == "replicate":
        border = "edge"

    return get_relay_op("pad")(data, padding, value, border)


def tile_converter(data, repeats, **kwargs):
    """Tile converter"""
    if kwargs:
        __unexpected_attrs("tile", kwargs)

    return get_relay_op("tile")(data, repeats)


def matmul_converter(a, b, **kwargs):
    """Matmul converter
    real signature: matmul_converter(a, b, transposeA, transposeB)"""

    transpose_a = kwargs.pop("transposeA")
    transpose_b = kwargs.pop("transposeB")
    if kwargs:
        __unexpected_attrs("matmul", kwargs)

    a_shape = infer_shape(a)
    b_shape = infer_shape(b)
    a_rank = len(a_shape)
    b_rank = len(b_shape)

    if a_rank == 2 and b_rank == 2:
        out = get_relay_op("matmul")(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    else:
        batch_shape = [1] * (max(a_rank, b_rank) - 2)

        for i, j in enumerate(reversed(a_shape[:-2])):
            batch_shape[i] = j

        for i, j in enumerate(reversed(b_shape[:-2])):
            # Need to check if axis can be broadcasted
            if batch_shape[i] == 1 or j == 1 or batch_shape[i] == j:
                batch_shape[i] = max(batch_shape[i], j)
            else:
                msg = "Batch dimensions are not broadcastable."
                raise AssertionError(msg)

        batch_shape = batch_shape[::-1]

        a = tvm_op.broadcast_to(a, batch_shape + list(a_shape[-2:]))
        b = tvm_op.broadcast_to(b, batch_shape + list(b_shape[-2:]))

        out = get_relay_op("batch_matmul")(
            tvm_op.reshape(a, [-1, *a_shape[-2:]]),
            tvm_op.reshape(b, [-1, *b_shape[-2:]]),
            transpose_b=transpose_b,
            transpose_a=transpose_a,
        )

        out_shape = batch_shape + [a_shape[-2]] + [b_shape[-1]]
        out = tvm_op.reshape(out, out_shape)

    return out


def sigmoid_converter(data, **kwargs):
    """Sigmoid converter"""
    if kwargs:
        __unexpected_attrs("sigmoid", kwargs)

    return get_relay_op("sigmoid")(data)


def relu_converter(data, **kwargs):
    """RELU converter"""
    if kwargs:
        __unexpected_attrs("relu", kwargs)

    return get_relay_op("relu")(data)


def prelu_converter(data, alpha, **kwargs):
    """PRELU converter"""
    if kwargs:
        __unexpected_attrs("prelu", kwargs)

    # prelu can"t handle float vals but NNEF supports direct parameter, this is just in case
    if isinstance(alpha, tvm_expr.Constant):
        if alpha.data.numpy().size == 1:
            return get_relay_op("leaky_relu")(data, alpha.data.numpy().item())

    return get_relay_op("prelu")(data, alpha)


def leaky_relu_converter(data, alpha, **kwargs):
    """Leaky RELU converter"""
    if kwargs:
        __unexpected_attrs("leaky_relu", kwargs)

    return get_relay_op("leaky_relu")(data, alpha)


def elu_converter(data, alpha, **kwargs):
    """ELU converter"""
    if kwargs:
        __unexpected_attrs("elu", kwargs)

    return select_converter(
        lt_converter(data, tvm_expr.const(0.0)),
        mul_converter(
            tvm_expr.const(alpha), sub_converter(exp_converter(data), tvm_expr.const(1.0))
        ),
        data,
    )


def selu_converter(data, alpha, **kwargs):
    """SELU converter
    True signature is selu_converter(data, alpha, lambda)"""
    lambda_var = kwargs.pop("lambda")

    if kwargs:
        __unexpected_attrs("selu", kwargs)

    return mul_converter(
        tvm_expr.const(lambda_var),
        select_converter(
            data < tvm_expr.const(0.0),
            mul_converter(
                tvm_expr.const(alpha), sub_converter(exp_converter(data), tvm_expr.const(1.0))
            ),
            data,
        ),
    )


def gelu_converter(data, **kwargs):
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

    return data * (
        tvm_expr.const(0.5) + tvm_op.erf(data * tvm_expr.const(0.5**0.5)) * tvm_expr.const(0.5)
    )


def silu_converter(data, **kwargs):
    """SiLU converter"""
    if kwargs:
        __unexpected_attrs("silu", kwargs)

    return mul_converter(data, sigmoid_converter(data))


def softmax_converter(data, axes, **kwargs):
    """Softmax converter"""
    if kwargs:
        __unexpected_attrs("softmax", kwargs)

    if len(axes) > 1:
        print("Multiple axes not supported, operation has been done along the first axis in axes.")
    axis = axes[0]

    return get_relay_op("softmax")(data, axis)


def softplus_converter(data, **kwargs):
    """Softplus converter"""
    if kwargs:
        __unexpected_attrs("softplus", kwargs)

    return log_converter(add_converter(exp_converter(data), tvm_expr.const(1.0)))


def linear_converter(data, _filter, bias, **kwargs):
    """Linear converter"""
    if kwargs:
        __unexpected_attrs("linear", kwargs)

    out = get_relay_op("matmul")(data, _filter, transpose_b=True)
    res = None

    if isinstance(bias, tvm_expr.Constant):
        if (bias.data.numpy() == 0).all():
            res = out

    if not res:
        # squeeze needed because nnef has bias of shape [1, channel]
        res = tvm_op.nn.bias_add(out, relay.squeeze(bias, axis=0))

    return res


def separable_conv_converter(
    data, plane_filter, point_filter, bias, border, padding, stride, dilation, groups, **kwargs
):
    """Separable convolution converter"""
    if kwargs:
        __unexpected_attrs("separable_conv", kwargs)

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    filtered = conv_converter(
        data, plane_filter, tvm_expr.const(0, dtype=d_type), border, stride, padding, dilation, 0
    )

    return conv_converter(filtered, point_filter, bias, "constant", [], [], [], groups)


def separable_deconv_converter(
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

    if isinstance(data, relay.Call):
        d_type = infer_type(data).checked_type.dtype
    else:
        d_type = data.type_annotation.dtype

    filtered = deconv_converter(
        data, point_filter, tvm_expr.const(0, dtype=d_type), "constant", [], [], [], [], groups
    )

    return deconv_converter(
        filtered, plane_filter, bias, border, stride, padding, dilation, output_shape, 0
    )


def max_pool_converter(data, size, border, padding, stride, dilation, **kwargs):
    """Max pool converter"""
    if kwargs:
        __unexpected_attrs("max_pool", kwargs)

    if border != "constant":
        print(f"Currently {border} border is not supported, used `constant` border")

    dshape = infer_shape(data)
    rank = len(dshape)

    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = dilation if dilation else ((1,) * (rank - 2))

    if not padding:
        # padding is truncated to `conv style` (only active layers are present)
        padding = _calculate_nnef_padding(dshape[2:], strides, pool_size, dilation)

    pad = _padding_conv(padding, rank)

    if border == "constant":
        padding = [(0, 0), (0, 0)] + padding
        data = pad_converter(data, padding, border, tvm_expr.const(0.0))
        pad = (0, 0)

    op = get_relay_op(dimension_picker("max_pool", dshape))
    return op(
        data,
        pool_size=pool_size,
        strides=strides,
        dilation=dilation,
        padding=pad,
    )


def avg_pool_converter(data, size, border, padding, stride, dilation, **kwargs):
    """Avg pool converter"""
    if kwargs:
        __unexpected_attrs("avg_pool", kwargs)

    if border not in ["constant", "ignore"]:
        print(f"Currently {border} border is not supported, used `constant` border")

    dshape = infer_shape(data)
    rank = len(dshape)
    pool_size = _size_conv(size, rank)
    strides = _stride_conv(stride, rank) if stride else (1,) * (rank - 2)

    dilation = dilation if dilation else ((1,) * (rank - 2))

    # padding is truncated to `conv style` (only active layers are present)
    active_shape = dshape[2:]
    if not padding:
        padding = _calculate_nnef_padding(active_shape, strides, pool_size, dilation)

    pad = _padding_conv(padding, rank)

    op = get_relay_op(dimension_picker("avg_pool", dshape))
    return op(
        data,
        pool_size=pool_size,
        strides=strides,
        dilation=dilation,
        padding=pad,
        count_include_pad=border != "ignore",
    )


def rms_pool_converter(data, size, border, padding, stride, dilation, **kwargs):
    """Rms pool converter"""
    if kwargs:
        __unexpected_attrs("rms_pool", kwargs)

    return sqrt_converter(
        avg_pool_converter(
            sqr_converter(data),
            size=size,
            border=border,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
    )


def local_response_normalization_converter(data, size, alpha, beta, bias):
    """LRN converter"""
    axis = [i for i in range(len(size)) if size[i] > 1]
    if len(axis) == 1:
        axis = axis[0]
    else:
        print("Multi axis LRN is not implemented properly, using first axis where size != 1")
        axis = axis[0]
    size = size[axis]
    return get_relay_op("lrn")(data, size, axis, bias, alpha, beta)


def local_mean_normalization_converter(data, size, **kwargs):
    """LMN converter"""
    if kwargs:
        __unexpected_attrs("local_mean_normalization", kwargs)

    mean = box_converter(data, size, "constant", [], [], [], normalize=True)
    return sub_converter(data, mean)


def local_variance_normalization_converter(data, size, bias, epsilon, **kwargs):
    """LVN converter"""
    if kwargs:
        __unexpected_attrs("local_variance_normalization", kwargs)

    sigma = box_converter(sqr_converter(data), size, "constant", [], [], [], normalize=True)
    return div_converter(
        data,
        max_converter(
            add_converter(sqrt_converter(sigma), tvm_expr.const(bias)), tvm_expr.const(epsilon)
        ),
    )


def local_contrast_normalization_converter(data, size, bias, epsilon, **kwargs):
    """LCN converter"""
    if kwargs:
        __unexpected_attrs("local_contrast_normalization", kwargs)

    centered = local_mean_normalization_converter(data, size)
    return local_variance_normalization_converter(centered, size, bias, epsilon)


def l1_normalization_converter(data, axes, bias, epsilon, **kwargs):
    """L1 norm converter"""
    if kwargs:
        __unexpected_attrs("l1_normalization", kwargs)

    sigma = sum_reduce_converter(abs_converter(data), axes, False)
    return div_converter(
        data, max_converter(add_converter(sigma, tvm_expr.const(bias)), tvm_expr.const(epsilon))
    )


def l2_normalization_converter(data, axes, bias, epsilon, **kwargs):
    """L2 norm converter"""
    if kwargs:
        __unexpected_attrs("l2_normalization", kwargs)

    epsilon = epsilon**2
    if bias != 0.0:
        print("Bias is not supported, assumed 0.0.")
    #     data = add_converter(data, tvm_expr.const(bias))

    return get_relay_op("l2_normalize")(data, epsilon, axes)


def batch_normalization_converter(data, mean, variance, offset, scale, epsilon, **kwargs):
    """Batch norm converter"""
    if kwargs:
        __unexpected_attrs("batch_normalization", kwargs)

    mean = squeeze_converter(mean, 0)
    variance = squeeze_converter(variance, 0)
    offset = squeeze_converter(offset, 0)
    scale = squeeze_converter(scale, 0)

    return get_relay_op("batch_norm")(data, scale, offset, mean, variance, epsilon=epsilon)[0]
