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
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
from tvm.auto_scheduler import _ffi_api


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
            ret[k] = tvm.nd.array(v.numpy())
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


def random_sparse_conv2d_params(func, params, bs_r, bs_c, density, layout):
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
    layout : str
        layout of network

    Returns
    -------
    Dict[Srting, tvm.nd.array]
        The generated random parameters.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.numpy())
        return ret

    new_params = deepcopy(params)
    conv2d_weight_names = relay.analysis.sparse_conv2d._search_conv2d_op_weight(func)
    for item in conv2d_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if not ((shape[0] == 1 and shape[1] == 1) or (shape[2] == 1 and shape[3] == 1)):
            continue
        if layout == "NCHW" and shape[0] % bs_r == 0 and shape[1] % bs_c == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], bs_r, bs_c, density, "float32").todense()
            new_params[name] = tvm.nd.array(np.array(new_w).reshape(shape))
        elif layout == "NHWC" and shape[3] % bs_r == 0 and shape[2] % bs_c == 0:
            new_w = random_bsr_matrix(shape[3], shape[2], bs_r, bs_c, density, "float32").todense()
            new_params[name] = tvm.nd.array(np.array(new_w).reshape(shape))
    return new_params


def convert_model_dense_to_sparse(
    mod, params, random_params=False, bs_r=1, bs_c=1, sparsity=0.85, layout="NHWC"
):
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
    layout : str
        layout of network

    Returns
    -------
    tvm.Module
        The updated sparse model.
    Dict[Srting, tvm.nd.array]
        The updated parameters.
    """

    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    if random_params:
        # Manually replace the parameters of dense to sparse tensors
        params = random_sparse_dense_params(mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity)
        # Manually replace the parameters of conv2d to sparse tensors
        params = random_sparse_conv2d_params(
            mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity, layout=layout
        )
    # convert dense matmul to sparse matmul
    mod, params = ddo.bsr_dense.convert(mod, params, (bs_r, bs_c), sparsity_threshold=0.8)
    # convert dense conv2d to sparse conv2d
    mod, params = ddo.bsr_conv2d.convert(
        mod, params, (bs_r, bs_c), sparsity_threshold=0.8, layout=layout
    )

    return tvm.IRModule.from_expr(mod), params


def sparse_sketch_rules():
    """Return the sketch rules for sparse op"""
    sparse_sketch_rule_list = [
        auto_scheduler.PreloadCustomSketchRule(
            sparse_conv2d_meet_condition_func, sparse_conv2d_apply_func, "SparseConv2D"
        ),
        auto_scheduler.PreloadCustomSketchRule(
            sparse_dense_meet_condition_func, sparse_dense_apply_func, "SparseDense"
        ),
        # Add more sketch rules for sparse
    ]
    return sparse_sketch_rule_list


def sparse_conv2d_meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_conv2d_sp_bsrmm",
        "sparse_conv2d_sp_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    return auto_scheduler.PreloadCustomSketchRule.PASS


def sparse_conv2d_apply_func(search_policy, state, stage_id):
    """Describe how to generate the initial sketch for sparse conv2d"""
    ret = []
    s_0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s_0.stages[stage_id].op.tag == "sparse_conv2d_sp_bsrmm_block":
        return [s_0.state_object, stage_id - 1]

    sparse_conv2d = s_0.stages[stage_id].op
    sparse_conv2d_block = s_0.stages[stage_id - 1].op
    assert sparse_conv2d.tag == "sparse_conv2d_sp_bsrmm"
    assert sparse_conv2d_block.tag == "sparse_conv2d_sp_bsrmm_block"
    layout = sparse_conv2d.attrs["layout"]

    # Set the default consumer of compute block
    consumer = sparse_conv2d

    # If sparse conv2d has a single elementwise consumer
    # We can compute inline the sparse_conv2d output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s_0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s_0.state_object, stage_id, consumer_id
        ):
            consumer = s_0.stages[consumer_id].op
            s_0.compute_inline(sparse_conv2d)

    c = None
    if layout == "NHWC":
        if len(s_0[sparse_conv2d_block].iters) == 6:
            # bs_c = 1
            i, h, w, nb_j, j, row_offset = s_0[  # pylint: disable=invalid-name
                sparse_conv2d_block
            ].iters
        else:
            i, h, w, nb_j, j, row_offset, c = s_0[  # pylint: disable=invalid-name
                sparse_conv2d_block
            ].iters
        m, x, y, n = s_0[consumer].iters
    elif layout == "NCHW":
        if len(s_0[sparse_conv2d_block].iters) == 6:
            # bs_c = 1
            i, nb_j, j, h, w, row_offset = s_0[  # pylint: disable=invalid-name
                sparse_conv2d_block
            ].iters
        else:
            i, nb_j, j, h, w, row_offset, c = s_0[  # pylint: disable=invalid-name
                sparse_conv2d_block
            ].iters
        m, n, x, y = s_0[consumer].iters

    i_0, i_1, i_2 = s_0.split(sparse_conv2d_block, i, [None, None])
    m_0, m_1 = s_0.follow_split(consumer, m, len(s_0.transform_steps) - 1, 1)
    h_0, h_1, h_2 = s_0.split(sparse_conv2d_block, h, [None, None])
    x_0, x_1 = s_0.follow_split(consumer, x, len(s_0.transform_steps) - 1, 1)
    w_0, w_1, w_2 = s_0.split(sparse_conv2d_block, w, [None, None])  # pylint: disable=invalid-name
    y_0, y_1 = s_0.follow_split(consumer, y, len(s_0.transform_steps) - 1, 1)
    j_0, j_1 = s_0.split(sparse_conv2d_block, nb_j, [None])
    n_0, n_1 = s_0.follow_split(consumer, n, len(s_0.transform_steps) - 1, 1)
    if layout == "NHWC":
        if c is None:
            s_0.reorder(
                sparse_conv2d_block,
                [i_0, h_0, w_0, j_0, i_1, h_1, w_1, j_1, row_offset, i_2, h_2, w_2, j],
            )
        else:
            s_0.reorder(
                sparse_conv2d_block,
                [i_0, h_0, w_0, j_0, i_1, h_1, w_1, j_1, row_offset, i_2, h_2, w_2, j, c],
            )
        s_0.reorder(consumer, [m_0, x_0, y_0, n_0, m_1, x_1, y_1, n_1])
    elif layout == "NCHW":
        if c is None:
            s_0.reorder(
                sparse_conv2d_block,
                [i_0, j_0, h_0, w_0, i_1, j_1, h_1, w_1, row_offset, i_2, j, h_2, w_2],
            )
        else:
            s_0.reorder(
                sparse_conv2d_block,
                [i_0, j_0, h_0, w_0, i_1, j_1, h_1, w_1, row_offset, i_2, j, c, h_2, w_2],
            )
        s_0.reorder(consumer, [m_0, n_0, x_0, y_0, m_1, n_1, x_1, y_1])
    s_0.compute_at(sparse_conv2d_block, consumer, n_0)

    ret.append([s_0.state_object, stage_id - 2])

    return ret


def sparse_dense_meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm",
        "sparse_dense_sp_rhs_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    return auto_scheduler.PreloadCustomSketchRule.PASS


def sparse_dense_apply_func(search_policy, state, stage_id):
    """Describe how to generate the initial sketch for sparse dense"""
    ret = []
    s_0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s_0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s_0.state_object, stage_id - 1]

    sparse_dense = s_0.stages[stage_id].op
    sparse_dense_block = s_0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    # Set the default consumer of compute block
    consumer = sparse_dense

    # If sparse dense has a single elementwise consumer
    # We can compute inline the sparse_dense output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s_0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s_0.state_object, stage_id, consumer_id
        ):
            consumer = s_0.stages[consumer_id].op
            s_0.compute_inline(sparse_dense)

    i, nb_j, j, row_offset, c = s_0[sparse_dense_block].iters
    m, n = s_0[consumer].iters
    i_0, i_1, i_2 = s_0.split(sparse_dense_block, i, [None, None])
    m_0, m_1 = s_0.follow_split(consumer, m, len(s_0.transform_steps) - 1, 1)
    j_0, j_1 = s_0.split(sparse_dense_block, nb_j, [None])
    n_0, n_1 = s_0.follow_split(consumer, n, len(s_0.transform_steps) - 1, 1)
    s_0.reorder(sparse_dense_block, [i_0, j_0, i_1, j_1, row_offset, i_2, j, c])
    s_0.reorder(consumer, [m_0, n_0, m_1, n_1])
    s_0.compute_at(sparse_dense_block, consumer, n_0)

    ret.append([s_0.state_object, stage_id - 2])

    return ret
