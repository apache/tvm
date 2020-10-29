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
# pylint: disable=invalid-name, invalid-name, too-many-locals, too-many-arguments
"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm.topi.utils import get_const_tuple
from .. import tag
from .bitserial_conv2d import _intrin_popcount
from ..nn.pad import pad
from ..nn.bitserial_util import bitpack, binary_op_multiplier


@autotvm.register_topi_compute("bitserial_dense.arm_cpu")
def bitserial_dense(cfg, data, weight, data_bits, weight_bits, pack_dtype, out_dtype, unipolar):
    """The default implementation of bitserial dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

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

    batch, DB, in_dim = get_const_tuple(data_packed.shape)
    out_dim, WB, in_dim = get_const_tuple(weight_packed.shape)

    # Pad Inputs so that microkernel can be used
    # out_dim and in_dim need to be multiples of 8
    if out_dim % 8 != 0:
        out_dim_pad = out_dim % 8
        data_packed = pad(data_packed, [0, 0, 0], [out_dim_pad, 0, 0], name="PaddedInput")
        out_dim += out_dim_pad

    ######## Search space

    x, y = cfg.axis(batch), cfg.axis(out_dim)
    db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(in_dim)

    ko, ki = cfg.define_split(
        "tile_k", k, num_outputs=2, filter=lambda xx: xx.size[-1] == 8 or xx.size[-1] == 16
    )
    xo, xi = cfg.define_split("tile_x", x, num_outputs=2)
    yo, yi = cfg.define_split("tile_y", y, num_outputs=2, filter=lambda xx: xx.size[-1] == 8)

    cfg.define_reorder(
        "reorder_0",
        [yo, xo, ko, xi, wb, db, yi, ki],
        policy="candidate",
        candidate=[
            [yo, xo, ko, xi, wb, db, yi, ki],
            [yo, xo, xi, ko, wb, db, yi, ki],
            [yo, xo, ko, xi, wb, db, yi, ki],
        ],
    )

    ###### Compute rule
    VY = cfg["tile_y"].size[-1]
    VK = cfg["tile_k"].size[-1]

    wvshape = (out_dim // VY, in_dim // VK, WB, VY, VK)
    oshape = (batch, out_dim)

    k = te.reduce_axis((0, in_dim), name="k")
    db = te.reduce_axis((0, DB), name="db")
    wb = te.reduce_axis((0, WB), name="wb")

    # Tile data and weights
    weight_vec = te.compute(
        wvshape,
        lambda yo, ko, wb, vy, vk: weight_packed[yo * VY + vy][wb][ko * VK + vk],
        name="weight_vec",
    )
    matmul_unipolar = te.compute(
        oshape,
        lambda x, y: te.sum(
            (
                tvm.tir.popcount(
                    weight_vec[y // VY, k // VK, wb, y % VY, k % VK].astype(out_dtype)
                    & data_packed[x, db, k].astype(out_dtype)
                )
                - tvm.tir.popcount(
                    ~weight_vec[y // VY, k // VK, wb, y % VY, k % VK].astype(out_dtype)
                    & data_packed[x, db, k].astype(out_dtype)
                )
            )
            << (wb + db).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense_unipolar",
    )

    matmul = te.compute(
        oshape,
        lambda x, y: te.sum(
            tvm.tir.popcount(
                weight_vec[y // VY, k // VK, wb, y % VY, k % VK].astype(out_dtype)
                & data_packed[x, db, k].astype(out_dtype)
            )
            << (wb + db).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense",
    )

    cfg.add_flop(batch * out_dim * in_dim * binary_op_multiplier(pack_dtype))

    if unipolar:
        return matmul_unipolar
    return matmul


@autotvm.register_topi_schedule("bitserial_dense.arm_cpu")
def schedule_bitserial_dense(cfg, outs):
    """Schedule for binary_dense.

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

    def _schedule(cfg, s, data_vec, weight_vec, output, unipolar):

        z, k, _, y, x = s[weight_vec].op.axis
        s[weight_vec].parallel(z)
        s[weight_vec].vectorize(x)

        x, y = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis
        _, DB, _ = get_const_tuple(data_vec.shape)
        _, _, WB, _, _ = get_const_tuple(weight_vec.shape)

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)

        cfg["reorder_0"].apply(s, output, [yo, xo, ko, xi, wb, db, yi, ki])

        fused = s[output].fuse(xo, yo)
        s[output].parallel(fused)

        nfactor = cfg["tile_y"].size[-1]
        kfactor = cfg["tile_k"].size[-1]
        if nfactor % 8 == 0:
            pc = _intrin_popcount(nfactor, kfactor, WB, DB, unipolar)
            s[output].tensorize(wb, pc)

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
            unipolar = output.op.tag == "bitserial_dense_unipolar"
            _schedule(cfg, s, data_vec, weight_vec, output, unipolar)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

    traverse(outs[0].op)
    return s
