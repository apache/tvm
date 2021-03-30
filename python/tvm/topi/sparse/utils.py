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
"""Some utils for Sparse operation."""
import tvm
from tvm import relay


def random_bsr_matrix(m, n, bs_r, bs_c, density, dtype):
    """Generate a random sparse matrix in bsr format.

    Returns
    -------
    scipy.sparse.bsr_matrix
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import itertools
    import scipy.sparse as sp

    y = np.zeros((m, n), dtype=dtype)
    assert m % bs_r == 0
    assert n % bs_c == 0
    nnz = int(density * m * n)
    num_blocks = int(nnz / (bs_r * bs_c)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, m, bs_r), range(0, n, bs_c))))
    assert candidate_blocks.shape[0] == m // bs_r * n // bs_c
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    # pylint: disable=invalid-name
    for (r, c) in chosen_blocks:
        y[r : r + bs_r, c : c + bs_c] = np.random.randn(bs_r, bs_c)
    s = sp.bsr_matrix(y, blocksize=(bs_r, bs_c))
    assert s.data.shape == (num_blocks, bs_r, bs_c)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (m // bs_r + 1,)
    return s


def random_sparse_dense_params(func, params, bs_r, bs_c, density):
    """Replace the dense parameters with random sparse parameters. Mainly used for testing.

    Parameters
    ----------
    func : tvm.relay.Expr
        Expr will be optimized to sparse operation.
    params : Dict[Srting, tvm.nd.array]
        Parameters of the Expr.
    bs_r : int
        The row of BSR matrix block.
    bs_c : int
        The column of BSR matrix block.
    density : float
        The density of the random sparse parameters.

    Returns
    -------
    Dict[Srting, tvm.nd.array]
        The generated random parameters.
    """

    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.asnumpy())
        return ret

    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if shape[0] % bs_r == 0 and shape[1] % bs_c == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], bs_r, bs_c, density, "float32").todense()
            new_params[name] = tvm.nd.array(new_w)
    return new_params


def convert_model_dense_to_sparse(mod, params, random_params=False, bs_r=1, bs_c=1, sparsity=0.85):
    """Convert a dense model to sparse model.

    Parameters
    ----------
    mod : tvm.Module
        The dense model.
    params : Dict[Srting, tvm.nd.array]
        Parameters of the dense model.
    random_params : Bool = False
        True to replace the parameters of the dense model with some random sparse tensors.
        This is mainly used for testing.
    bs_r : int
        The row of BSR matrix block.
    bs_c : int
        The column of BSR matrix block.
    sparsity : float
        The sparsity of the random sparse parameters.

    Returns
    -------
    tvm.Module
        The updated sparse model.
    Dict[Srting, tvm.nd.array]
        The updated parameters.
    """
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    if random_params:
        # Manually replace the parameters of dense model to sparse tensors
        params = random_sparse_dense_params(mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity)
    # Currently we only support to conver dense matmul to sparse dense matmul
    mod, params = ddo.bsr_dense.convert(mod, params, (bs_r, bs_c), sparsity_threshold=0.8)

    return tvm.IRModule.from_expr(mod), params
