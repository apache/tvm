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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Bitserial Dense operator."""
from __future__ import absolute_import
import tvm
from tvm import autotvm
from topi.util import get_const_tuple
from .bitserial_util import bitpack, binary_op_multiplier

@tvm.target.generic_func
def bitserial_dense(data, weight, data_bits, weight_bits, pack_dtype='uint32',
                    out_dtype='int16', unipolar=True):
    """The default implementation of bitserial dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim] or
        3-D with shape [out_dim, weight_bits, in_dim]
    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    if len(weight.shape) == 2:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else:
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)

    oshape = (Y, X)
    k = tvm.reduce_axis((0, K), name='k')
    db = tvm.reduce_axis((0, DB), name='db')
    wb = tvm.reduce_axis((0, WB), name='wb')

    matmul_unipolar = tvm.compute(oshape, lambda i, j: tvm.sum(
        (tvm.popcount(weight_packed[j, wb, k] & data_packed[i, db, k]) -
         tvm.popcount(~weight_packed[j, wb, k] & data_packed[i, db, k])).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]),
                                  tag='bitserial_dense_unipolar')

    matmul = tvm.compute(oshape, lambda i, j: tvm.sum(
        tvm.popcount(weight_packed[j, wb, k] & data_packed[i, db, k]).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]), tag='bitserial_dense')


    if unipolar:
        return matmul_unipolar
    return matmul


@autotvm.register_topi_compute(bitserial_dense, ['cpu'], 'direct')
def bitserial_dense_default(cfg, data, weight, data_bits, weight_bits, pack_dtype='uint32',
                            out_dtype='int16', unipolar=True):
    """Bitserial dense implementation. TODO: Why are these separate

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim] or
        3-D with shape [out_dim, weight_bits, in_dim]
    Returns
    -------
    output : tvm.Tensor
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
    ko, ki = cfg.define_split('tile_k', k, policy='all', num_outputs=2)
    yo, yi = cfg.define_split('tile_y', y, policy='all', num_outputs=2)
    xo, xi = cfg.define_split('tile_x', x, policy='all', num_outputs=2)

    cfg.define_reorder('reorder_0', [yo, xo, ko, yi, wb, db, ki, xi],
                       policy='candidate', candidate=[
                           [yo, xo, ko, yi, wb, db, ki, xi],
                           [yo, xo, yi, ko, wb, db, ki, xi]])

    cfg.define_annotate('ann_reduce', [db, wb], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [yi, xi], policy='try_unroll_vec')

    ###### Compute rule
    VX = cfg['tile_x'].size[-1]

    wvshape = (X//VX, WB, VX, K)
    oshape = (Y, X)

    k = tvm.reduce_axis((0, K), name='k')
    db = tvm.reduce_axis((0, DB), name='db')
    wb = tvm.reduce_axis((0, WB), name='wb')

    # Tile data and weights
    weight_vec = tvm.compute(wvshape, lambda xo, wb, vx, k:
                             weight_packed[xo*VX+vx][wb][k], name='weight_vec')

    matmul_unipolar = tvm.compute(oshape, lambda i, j: tvm.sum(
        (tvm.popcount(weight_vec[j//VX, wb, j%VX, k] & data_packed[i, db, k]) -
         tvm.popcount(~weight_vec[j//VX, wb, j%VX, k] & data_packed[i, db, k])).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]), tag='bitserial_dense_unipolar')

    matmul = tvm.compute(oshape, lambda i, j: tvm.sum(
        tvm.popcount(weight_vec[j//VX, wb, j%VX, k] & data_packed[i, db, k]).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]), tag='bitserial_dense')

    # binary ops
    cfg.add_flop(2 * Y * X * K * binary_op_multiplier(pack_dtype))

    if unipolar:
        return matmul_unipolar
    return matmul
