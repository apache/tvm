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
from tvm import relay

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

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    # Pad input and output channels to use tensorcore schedule.
    if dtype in ["float16"]:  # todo: support int8/int4
        B, M, K = x_tensor.shape
        B, N, K = y_tensor.shape
        M = M.value
        K = K.value
        N = N.value

        # The shape of (M, K, N) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32)
        if (
            (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
            or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
            or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
        ):
            # no need to pad
            return None

        (dm, dk, dn), extra_flops = pad_to_tensorcore(M, K, N)

        if extra_flops > 2:
            logger.info("batch_matmul pad_to_tensorcore skipped, extra_flops %s", extra_flops)
            return None

        logger.info("batch_matmul pad_to_tensorcore, extra_flops %s", extra_flops)
        if dm or dk:
            x_ = relay.nn.pad(x, pad_width=((0, 0), (0, dm), (0, dk)))
        else:
            x_ = x
        if dn or dk:
            y_ = relay.nn.pad(y, pad_width=((0, 0), (0, dn), (0, dk)))
        else:
            y_ = y
        out_ = relay.nn.batch_matmul(x_, y_)
        if dm or dn:
            original_out_shape = [x.value for x in output_tensor.shape]
            out = relay.strided_slice(out_, begin=[0, 0, 0], end=original_out_shape)
        else:
            out = out_
        return out
    return None


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
    # Collect the input tensors.
    x_tensor, y_tensor = arg_types[0], arg_types[1]
    dtype = x_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    # Pad input and output channels to use tensorcore schedule.
    if dtype in ["float16"]:  # todo: support int8/int4
        M, K = x_tensor.shape
        N, K = y_tensor.shape
        try:
            M = M.value
            K = K.value
            N = N.value
        except AttributeError:
            # todo: deal with unfixed shape when compiling wdl model
            return None

        # The shape of (M, K, N) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32)
        if (
            (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
            or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
            or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
        ):
            # no need to pad
            return None

        (dm, dk, dn), extra_flops_ratio = pad_to_tensorcore(M, K, N)

        if extra_flops_ratio > 2:
            logger.info("dense pad_to_tensorcore skipped, extra_flops_ratio %s", extra_flops_ratio)
            return None

        logger.info("dense pad_to_tensorcore, extra_flops_ratio %s", extra_flops_ratio)

        if dm or dk:
            x_ = relay.nn.pad(x, pad_width=((0, dm), (0, dk)))
        else:
            x_ = x
        if dn or dk:
            y_ = relay.nn.pad(y, pad_width=((0, dn), (0, dk)))
        else:
            y_ = y
        out_ = relay.nn.dense(x_, y_)
        if dm or dn:
            original_out_shape = [x.value for x in output_tensor.shape]
            out = relay.strided_slice(out_, begin=[0, 0], end=original_out_shape)
        else:
            out = out_
        return out
    return None


def pad_to_tensorcore(M, K, N):
    """pad shape to enable tensorcore"""
    candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]

    flops = M * K * N
    extra_flops = math.inf
    best_pad = (0, 0, 0)
    for padding in candidates:
        dm, dk, dn = _pad_to(M, K, N, padding)
        e = (M + dm) * (N + dn) * (K + dk) - M * N * K
        # print(dm, dk, dn, e, flops)
        if e < extra_flops:
            extra_flops = e
            best_pad = (dm, dk, dn)
    return best_pad, extra_flops / flops


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
