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
import scipy.sparse as sp
import tvm
from . import _ffi_api


SparseAnalysisResult = namedtuple(
    "SparseAnalysisResult",
    [
        "weight_name",
        "weight_shape",
    ],
)


def _search_dense_op_weight(expr):
    """Search name of weight in all ```nn.dense``` operator
       This is a helpful function to determine which param need
       to be converted to sparse

    Parameters
    ----------
    expr : relay.Expr
        Expr will be searched

    Returns
    -------
    ret : Array[String]
        name of weight in all ``nn.dense``` operator
    """
    return _ffi_api.search_dense_op_weight(expr)


def process_params(expr, params, block_size, sparsity_threshold):
    """[summary]

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

    Returns
    -------
    ret : Namedtuple[weight_name: Array[String], weight_shape: Array[Array[IntImm]]]
        return names of qualified dense weight and the shape in BSR format
    """

    # pylint: disable=import-outside-toplevel
    from tvm.auto_scheduler.search_task import (
        register_task_input_buffer,
    )  # lazily import to avoid recursive dependency

    memo = SparseAnalysisResult(weight_name=[], weight_shape=[])
    weight_names = _search_dense_op_weight(expr)
    for name in weight_names:
        name = str(name)
        w_np = params[name].numpy()
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
        if sparsity >= sparsity_threshold:
            sparse_weight = sp.bsr_matrix(w_np, blocksize=block_size)
            # remove dense weight
            del params[name]
            memo.weight_name.append(name)
            memo.weight_shape.append(
                list(sparse_weight.data.shape)
                + list(sparse_weight.indices.shape)
                + list(sparse_weight.indptr.shape)
            )
            params[name + ".data"] = tvm.nd.array(sparse_weight.data)
            params[name + ".indices"] = tvm.nd.array(sparse_weight.indices)
            params[name + ".indptr"] = tvm.nd.array(sparse_weight.indptr)

            prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%d_" % (
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
                tvm.runtime.ndarray.array(sparse_weight.data),
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
