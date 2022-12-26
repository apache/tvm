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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Tensorcore alter op and legalize functions for cuda backend"""

import logging
import math
from tvm import relay, tir

from .. import nn

logger = logging.getLogger("topi")


@nn.batch_matmul_legalize.register("cuda")
def _batch_matmul_legalize(attrs, inputs, arg_types):
    """Legalizes batch_matmul op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    arg_types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # Collect the input tensors.
    x_tensor, y_tensor = arg_types[0], arg_types[1]
    dtype = x_tensor.dtype

    if attrs.transpose_a:
        B, K, M = x_tensor.shape
    else:
        B, M, K = x_tensor.shape

    if attrs.transpose_b:
        B, N, K = y_tensor.shape
    else:
        B, K, N = y_tensor.shape

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    if (
        isinstance(B, tir.expr.Any)
        or isinstance(M, tir.expr.Any)
        or isinstance(K, tir.expr.Any)
        or isinstance(N, tir.expr.Any)
    ):
        # Dynamic shape do not support alter op layout now
        return None

    M = M.value
    K = K.value
    N = N.value

    # Pad input and output channels to use tensorcore schedule.
    if dtype in ["float16", "int8", "uint8"]:
        # The shape of (M, K, N) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32)
        if (
            (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
            or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
            or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
        ):
            # no need to pad
            return None
        candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
    elif dtype in ["int4", "uint4"]:
        if M % 8 == 0 and K % 32 == 0 and N % 8 == 0:
            # no need to pad
            return None

        candidates = [(8, 32, 8)]
    else:
        return None

    (dm, dk, dn), extra_flops = pad_to_tensorcore(M, K, N, candidates)

    if extra_flops > 2:
        logger.info("batch_matmul pad_to_tensorcore skipped, extra_flops %s", extra_flops)
        return None

    logger.info("batch_matmul pad_to_tensorcore, extra_flops %s", extra_flops)

    if attrs.transpose_a:
        pad_width = ((0, 0), (0, dk), (0, dm))
    else:
        pad_width = ((0, 0), (0, dm), (0, dk))

    x_ = relay.nn.pad(x, pad_width=pad_width) if dm or dk else x

    if attrs.transpose_b:
        pad_width = ((0, 0), (0, dn), (0, dk))
    else:
        pad_width = ((0, 0), (0, dk), (0, dn))

    y_ = relay.nn.pad(y, pad_width=pad_width) if dn or dk else y

    out_ = relay.nn.batch_matmul(x_, y_, **attrs)

    out = (
        relay.strided_slice(out_, begin=[0, 0, 0], end=[x.value for x in output_tensor.shape])
        if dm or dn
        else out_
    )
    return out


@nn.dense_legalize.register("cuda")
def _dense_legalize(attrs, inputs, arg_types):
    """Legalizes dense op.

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
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    # Collect the input tensors.
    x_tensor, y_tensor = arg_types[0], arg_types[1]
    dtype = x_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    M, K = x_tensor.shape
    N, K = y_tensor.shape
    try:
        M = M.value
        K = K.value
        N = N.value
    except AttributeError:
        # todo: deal with unfixed shape when compiling wdl model
        return None

    # Pad input and output channels to use tensorcore schedule.
    if dtype in ["float16", "int8", "uint8"]:
        # The shape of (M, K, N) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32)
        if (
            (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
            or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
            or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
        ):
            # no need to pad
            return None

        candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
    elif dtype in ["int4", "uint4"]:
        if M % 8 == 0 and K % 32 == 0 and N % 8 == 0:
            # no need to pad
            return None
        candidates = [(8, 32, 8)]
    else:
        return None

    (dm, dk, dn), extra_flops_ratio = pad_to_tensorcore(M, K, N, candidates)
    skip_pad = extra_flops_ratio > 2

    if skip_pad and dtype in ["int8", "uint8"]:
        skip_pad = False
        # If tensorcore schedule padding fails, pad to nearest upward 4x4x4 as long as
        # the additional flops ratio isn't double or more.
        # Note that 4x4x4 is invalid for tensorcore scheduling, but padding upwards to 4x4x4
        # doesn't hurt if tensorcore padding has already failed.
        if M % 4 == 0 and K % 4 == 0 and N % 4 == 0:
            # No need to pad
            return None
        (dm, dk, dn) = _pad_to(M, K, N, (4, 4, 4))
        extra_flops_ratio = _extra_flops(M, K, N, dm, dk, dn) / (M * K * N)
        skip_pad = extra_flops_ratio > 2

    if skip_pad:
        logger.info("dense pad_to_tensorcore skipped, extra_flops_ratio %s", extra_flops_ratio)
        return None

    logger.info("dense pad_to_tensorcore, extra_flops_ratio %s", extra_flops_ratio)

    x_ = relay.nn.pad(x, pad_width=((0, dm), (0, dk))) if dm or dk else x
    y_ = relay.nn.pad(y, pad_width=((0, dn), (0, dk))) if dn or dk else y

    # If units is explicitly specified, it is used to compute the output shape.
    # We need to update units after padding to prevent a type error.
    if attrs["units"] is not None:
        new_attrs["units"] = N + dn

    out_ = relay.nn.dense(x_, y_, **new_attrs)
    out = (
        relay.strided_slice(out_, begin=[0, 0], end=[x.value for x in output_tensor.shape])
        if dm or dn
        else out_
    )
    return out


def pad_to_tensorcore(M, K, N, candidates):
    """pad shape to enable tensorcore"""
    flops = M * K * N
    extra_flops = math.inf
    best_pad = (0, 0, 0)
    for padding in candidates:
        dm, dk, dn = _pad_to(M, K, N, padding)
        e = _extra_flops(M, K, N, dm, dk, dn)
        # print(dm, dk, dn, e, flops)
        if e < extra_flops:
            extra_flops = e
            best_pad = (dm, dk, dn)
    return best_pad, extra_flops / flops


def _extra_flops(M, K, N, dm, dk, dn):
    return (M + dm) * (N + dn) * (K + dk) - M * N * K


def _pad_to(M, K, N, PADDING):
    dm, dk, dn = 0, 0, 0

    if M % PADDING[0] != 0:
        M_ = ((M + PADDING[0]) // PADDING[0]) * PADDING[0]
        dm = M_ - M
    if K % PADDING[1] != 0:
        K_ = ((K + PADDING[1]) // PADDING[1]) * PADDING[1]
        dk = K_ - K
    if N % PADDING[2] != 0:
        N_ = ((N + PADDING[2]) // PADDING[2]) * PADDING[2]
        dn = N_ - N

    return dm, dk, dn
