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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments, condition-evals-to-constant
"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm.topi.utils import get_const_int, get_const_tuple
from .. import tag
from ..nn.bitserial_util import bitpack, binary_op_multiplier


@autotvm.register_topi_compute("bitserial_dense.x86")
def bitserial_dense(
    cfg, data, weight, data_bits, weight_bits, pack_dtype="uint32", out_dtype="int16", unipolar=True
):
    """Bitserial dense implementation. TODO: Why are these separate

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim] or
        3-D with shape [out_dim, weight_bits, in_dim]
    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    if len(weight.shape) == 2:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else:
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)
    ######## Search space
    x, y = cfg.axis(X), cfg.axis(Y)
    db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(K)
    ko, ki = cfg.define_split("tile_k", k, num_outputs=2)
    yo, yi = cfg.define_split("tile_y", y, num_outputs=2)
    xo, xi = cfg.define_split("tile_x", x, num_outputs=2)

    cfg.define_reorder(
        "reorder_0",
        [yo, xo, ko, yi, wb, db, ki, xi],
        policy="candidate",
        candidate=[[yo, xo, ko, yi, wb, db, ki, xi], [yo, xo, yi, ko, wb, db, ki, xi]],
    )

    cfg.define_annotate("ann_reduce", [db, wb], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [yi, xi], policy="try_unroll_vec")

    ###### Compute rule
    VX = cfg["tile_x"].size[-1]

    wvshape = (X // VX, WB, VX, K)
    oshape = (Y, X)

    k = te.reduce_axis((0, K), name="k")
    db = te.reduce_axis((0, DB), name="db")
    wb = te.reduce_axis((0, WB), name="wb")

    # Tile data and weights
    weight_vec = te.compute(
        wvshape, lambda xo, wb, vx, k: weight_packed[xo * VX + vx][wb][k], name="weight_vec"
    )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    matmul_unipolar = te.compute(
        oshape,
        lambda i, j: te.sum(
            (
                tvm.tir.popcount(
                    weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k]
                )
                - tvm.tir.popcount(
                    ~weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k]
                )
            ).astype(out_dtype)
            << (db + wb).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense_unipolar",
    )

    matmul = te.compute(
        oshape,
        lambda i, j: te.sum(
            tvm.tir.popcount(
                weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k]
            ).astype(out_dtype)
            << (db + wb).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense",
    )

    # binary ops
    cfg.add_flop(2 * Y * X * K * binary_op_multiplier(pack_dtype))

    if unipolar:
        return matmul_unipolar
    return matmul


@autotvm.register_topi_schedule("bitserial_dense.x86")
def schedule_bitserial_dense(cfg, outs):
    """Schedule for bitserial_dense.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of bitserial dense operator.
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for bitserial_dense.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, data_vec, weight_vec, output):
        s[data_vec].parallel(s[data_vec].op.axis[0])
        s[weight_vec].parallel(s[weight_vec].op.axis[0])

        y, x = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)

        cfg["reorder_0"].apply(s, output, [yo, xo, ko, yi, wb, db, ki, xi])
        cfg["ann_reduce"].apply(
            s,
            output,
            [db, wb],
            axis_lens=[get_const_int(db.dom.extent), get_const_int(wb.dom.extent)],
            max_unroll=8,
            cfg=cfg,
        )
        cfg["ann_spatial"].apply(
            s,
            output,
            [yi, xi],
            axis_lens=[cfg["tile_y"].size[-1], cfg["tile_x"].size[-1]],
            max_unroll=8,
            cfg=cfg,
        )
        s[output].vectorize(xi)
        s[output].parallel(yo)
        return s

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or "elemwise" in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)

        elif op.tag == "bitserial_dense" or "bitserial_dense_unipolar":
            output = op.output(0)
            weight_vec = op.input_tensors[0]

            data_vec = op.input_tensors[1]
            data = data_vec.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                data = data.op.input_tensors[0]
            _schedule(cfg, s, data_vec, weight_vec, output)
        else:
            raise RuntimeError(f"Unsupported operator: {op.tag}")

    traverse(outs[0].op)
    return s
