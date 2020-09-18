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
# pylint: disable=unused-argument, not-context-manager
"""Automatic convert model from dense to block sparse"""

from tvm import relay
from tvm.relay.analysis.sparse_dense import process_params

from .utils import _run_opt_pass


def convert(func, params, blocksize, sparsity_threshold):
    """Convert a dense func and according parameters to block sparse

    Parameters
    ----------
    func : relay.Expr
        Expr will be optimized to sparse operation
    params : Dict[Srting, tvm.nd.array]
        Parameters of the Expr
    blocksize : Tuple(int, int)
        Blocksize for BSR matrix
    sparsity_threshold : float
        Minimal sparsity requirement for converting.
        If weight sparsity is lower than this threshold,
        the dense operation will be kept.

    Returns
    -------
    new_func: relay.Expr
        Mutated Expr with sparse operations

    params: Dict[Srting, tvm.nd.array]
        New params with BSR matrix for mutated Expr
    """
    weight_info = process_params(func, params, blocksize, sparsity_threshold)
    new_func = _run_opt_pass(
        func, relay.transform.DenseToSparse(weight_info.weight_name, weight_info.weight_shape)
    )
    return new_func, params
