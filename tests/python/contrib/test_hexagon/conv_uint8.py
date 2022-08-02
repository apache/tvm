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

import numpy as np
import tvm

from numbers import Integral
from tvm import te

def get_const_int(expr):
            """Verifies expr is integer and get the constant value.

            Parameters
            ----------
            expr : tvm.Expr or int
                The input expression.

            Returns
            -------
            out_value : int
                The output.
            """
            if isinstance(expr, Integral):
                return expr
            if not isinstance(expr, tvm.tir.IntImm):
                ana = tvm.arith.Analyzer()
                expr = ana.simplify(expr)
            if not isinstance(expr, tvm.tir.IntImm):
                raise ValueError("Expect value to be constant int")
            return int(expr.value)

def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)

def Pad(Input, padding):
    batch, in_height, in_width, in_channel = Input.shape
    return te.compute(
        (batch, in_height + 2 * padding, in_width + 2 * padding, in_channel),
        lambda nn, yy, xx, cc: tvm.tir.if_then_else(
            te.all(
                yy >= padding,
                yy - padding < in_height,
                xx >= padding,
                xx - padding < in_width,
            ),
            Input[nn, yy - padding, xx - padding, cc],
            tvm.tir.const(0, Input.dtype),
        ),
        name="Apad",
    )

def schedule_qconv2d_nhwc(outs, target, device):
    s = te.create_schedule([x.op for x in outs])
    x = outs[0]
    nn, yy, xx, cc = s[x].op.axis
    px1, px2 = s[x].split(nn, nparts=1)
    return s

def qconv2d_nhwc(Input, in_offset, Filter, filt_offset, stride, padding, out_dtype=None):
    if out_dtype is None:
        out_dtype = Input.dtype

    batch, in_height, in_width, in_channel = Input.shape
    filt_height, filt_width, _, num_filter = Filter.shape
    # Input is already padded. No need to add padding while computing
    # out_height and out_width.
    out_height = (in_height - filt_height) // stride + 1
    out_width = (in_width - filt_width) // stride + 1
    out_channel = num_filter

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, filt_height), name="ry")
    rx = te.reduce_axis((0, filt_width), name="rx")

    return te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            (Input[nn, yy * stride + ry, xx * stride + rx, rc] - in_offset)
            * (Filter[ry, rx, rc, ff] - filt_offset).astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="qconv2d_nhwc",
    )


def run_conv_te(hexagon_session, a, w, a_offset, w_offset, padding):

    # Input tensor size
    A = te.placeholder(a.shape, name="A", dtype="uint8")
    W = te.placeholder(w.shape, name="W", dtype="uint8")

    # Pad input and create computation for quantized conv2d
    Apad = Pad(A, padding)
    B = qconv2d_nhwc(Apad, a_offset, W, w_offset, 1, padding)
    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    device = hexagon_session.device
    s = schedule_qconv2d_nhwc([B], target_hexagon, device)
    nn, yy, xx, cc = s[B].op.axis
    yo, yi = s[B].split(yy, nparts=1)
    s[Apad].compute_at(s[B], yi)
    s[B].vectorize(cc)

    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), device=device)
    func_te = tvm.build(s, [A, W, B], target=tvm.target.Target(target_hexagon, host=target_hexagon), name="quant_conv2d")

    module_te = hexagon_session.load_module(func_te)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
    w_hexagon = tvm.runtime.ndarray.array(w, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device)

    module_te(a_hexagon, w_hexagon, b_hexagon)
    evaluator = module_te.time_evaluator(module_te.entry_name, hexagon_session.device, number=1, repeat=1)
    mean_ms = evaluator(a_hexagon, w_hexagon, b_hexagon).mean * 1000

    out = b_hexagon.numpy()

    return out, mean_ms
    