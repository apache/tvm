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
"""Schedule for batch_matmul operator"""
from tvm import autotvm
from tvm.contrib import rocblas
from .. import generic
from ..utils import get_const_tuple


@autotvm.register_topi_compute("batch_matmul_rocblas.rocm")
def batch_matmul_rocblas(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Computes matrix multiplication of `x` and `y` via rocblas when
    `x` and `y` are batched matrices.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]
    y : tvm.te.Tensor
        3-D with shape [batch, N, K]
    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    del out_dtype
    batch, M, K = get_const_tuple(x.shape)
    _, N, _ = get_const_tuple(y.shape)
    if out_shape is not None:
        assert out_shape[0] == batch, "Input and output batch sizes must match"
        assert out_shape[1] == M and out_shape[2] == N, "Invalid output shape"
    result = rocblas.batch_matmul(x, y, transpose_a, transpose_b)
    cfg.add_flop(batch * M * N * K * 2)
    return result


@autotvm.register_topi_schedule("batch_matmul_rocblas.rocm")
def schedule_batch_matmul_rocblas(_, outs):
    """Schedule for batch_matmul operator with rocm cblas"""
    return generic.schedule_extern(outs)
