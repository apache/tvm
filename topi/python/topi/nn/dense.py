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
"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag

def dense_default(data, weight, bias=None, out_dtype=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i, k].astype(out_dtype) * \
                                              weight[j, k].astype(out_dtype), axis=k), \
                         name='T_dense', tag='dense')
    if bias is not None:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j].astype(out_dtype), \
                             tag=tag.BROADCAST)
    return matmul


@tvm.target.override_native_generic_func("dense")
def dense(data, weight, bias=None, out_dtype=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    return dense_default(data, weight, bias, out_dtype)
