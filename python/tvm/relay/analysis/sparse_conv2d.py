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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains helper functions for convert dense model
to block sparse model
"""
from collections import namedtuple

import numpy as np

import tvm

from . import _ffi_api

SparseAnalysisResult = namedtuple(
    "SparseAnalysisResult",
    [
        "weight_name",
        "weight_shape",
    ],
)


def _search_conv2d_op_weight(expr):
    """Search name of weight in all ```nn.conv2d``` operator
       This is a helpful function to determine which param need
       to be converted to sparse

    Parameters
    ----------
    expr : relay.Expr
        Expr will be searched

    Returns
    -------
    ret : Array[String]
        name of weight in all ``nn.conv2d``` operator
    """
    return _ffi_api.search_conv2d_op_weight(expr)


def process_params(
    expr, params, block_size, sparsity_threshold, layout, kernel_size, reg_task_input=True
):
    """Process parameters of conv2d from dense to sparse.

    Parameters
    ----------
    expr : Relay.Expr
        Expr of the network
    params : Dict[String, tvm.nd.array]
        parameters of the network
    block_size : Tuple(int, int)
        Blocksize in BSR matrix
    sparsity_threshold : float
        Minimal sparsity requirement for converting to sparse operation
    layout : str
        layout of network

    Returns
    -------
    ret : Namedtuple[weight_name: Array[String], weight_shape: Array[Array[IntImm]]]
        return names of qualified conv2d weight and the shape in BSR format
    """

    # pylint: disable=import-outside-toplevel
    import scipy.sparse as sp

    from tvm.auto_scheduler.search_task import (  # lazily import to avoid recursive dependency
        register_task_input_buffer,
    )

    memo = SparseAnalysisResult(weight_name=[], weight_shape=[])
    weight_names = _search_conv2d_op_weight(expr)
    for name in weight_names:
        name = str(name)
        w_np = params[name].numpy()

        if layout == "NHWC":  # HWIO
            weight_kernel = (w_np.shape[0], w_np.shape[1])
        elif layout == "NCHW":  # OIHW
            weight_kernel = (w_np.shape[2], w_np.shape[3])
        if weight_kernel[0] != weight_kernel[1]:
            continue

        if weight_kernel[0] == kernel_size == 1:
            sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
            if sparsity < sparsity_threshold:
                continue
            if layout == "NHWC":
                w_np = w_np.squeeze().T
            elif layout == "NCHW":
                w_np = w_np.squeeze()

            sparse_weight = sp.bsr_matrix(w_np, blocksize=block_size)

            # when bs_c=1, remove this dim
            if block_size[1] == 1:
                sparse_weight_data = sparse_weight.data.reshape(
                    sparse_weight.data.shape[0], block_size[0]
                )
            else:
                sparse_weight_data = sparse_weight.data
        elif weight_kernel[0] == kernel_size == 3:
            if layout == "NHWC":  # HWIO
                w_np = w_np.reshape((-1, w_np.shape[-1])).T
            elif layout == "NCHW":  # OIHW
                w_np = w_np.reshape((w_np.shape[0], -1))
            sparse_weight = sp.bsr_matrix(w_np, blocksize=block_size)
            if 1 - (sparse_weight.nnz / w_np.size) < sparsity_threshold:
                continue
            sparse_weight_data = sparse_weight.data
        else:
            continue

        # remove dense weight
        del params[name]
        memo.weight_name.append(name)
        memo.weight_shape.append(
            list(sparse_weight_data.shape)
            + list(sparse_weight.indices.shape)
            + list(sparse_weight.indptr.shape)
        )
        params[name + ".data"] = tvm.nd.array(sparse_weight_data)
        params[name + ".indices"] = tvm.nd.array(sparse_weight.indices)
        params[name + ".indptr"] = tvm.nd.array(sparse_weight.indptr)

        if reg_task_input:
            prefix = "sparse_conv2d_bsr_%d_%d_%d_%d_%d_%d_" % (
                w_np.shape[0],
                w_np.shape[1],
                block_size[0],
                block_size[1],
                sparse_weight.indices.shape[0],
                sparse_weight.indptr.shape[0],
            )
            register_task_input_buffer(
                "default",
                prefix + "W_data",
                tvm.runtime.ndarray.array(sparse_weight_data),
                overwrite=True,
            )
            register_task_input_buffer(
                "default",
                prefix + "W_indices",
                tvm.runtime.ndarray.array(sparse_weight.indices),
                overwrite=True,
            )
            register_task_input_buffer(
                "default",
                prefix + "W_indptr",
                tvm.runtime.ndarray.array(sparse_weight.indptr),
                overwrite=True,
            )
    ret = SparseAnalysisResult(
        weight_name=tvm.runtime.convert(memo.weight_name),
        weight_shape=tvm.runtime.convert(memo.weight_shape),
    )
    return ret
