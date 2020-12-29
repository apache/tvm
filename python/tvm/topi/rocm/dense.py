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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for dense operator"""
from tvm import te
from tvm import autotvm
from tvm.contrib import rocblas
from .. import generic, nn
from .. import tag
from ..utils import traverse_inline


@autotvm.register_topi_compute("dense.rocm")
def dense(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for rocm backend.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.te.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense.rocm")
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            Dense = op.output(0)
            num_thread = 64
            k = Dense.op.reduce_axis[0]
            ko, kf = s[Dense].split(k, factor=num_thread)
            DenseF = s.rfactor(Dense, kf)

            if Dense.op in s.outputs:
                Out = Dense
            else:
                Out = outs[0].op.output(0)
                s[Dense].compute_at(s[Out], s[Out].op.axis[1])
            s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
            s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

            tx = s[Dense].op.reduce_axis[0]
            thread_x = te.thread_axis("threadIdx.x")
            s[Dense].bind(tx, thread_x)
            s[DenseF].compute_at(s[Dense], tx)
            s[Dense].set_store_predicate(thread_x.var.equal(0))
            s[Out].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("dense_rocblas.rocm")
def dense_rocblas(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for rocm backend with cblas.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.te.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    if out_dtype is None:
        out_dtype = data.dtype
    assert out_dtype == data.dtype, "Mixed precision not supported."
    matmul = rocblas.matmul(data, weight, False, True)
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    cfg.add_flop(batch * in_dim * out_dim * 2)
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim), lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_schedule("dense_rocblas.rocm")
def schedule_dense_rocblas(_, outs):
    """Schedule for dense operator with rocm cblas"""
    return generic.schedule_extern(outs)
