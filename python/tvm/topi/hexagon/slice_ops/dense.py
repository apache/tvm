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

"""Schedule for dense operator"""

import tvm
from tvm import topi, te, tir
from ..utils import get_layout_transform_fn

def dense_compute(tensor_a, tensor_b, bias=None, out_dtype=None):
    """The default implementation of dense in topi.
    This is an alias of matmul_nt operator for data tensor in non-transposed format and weight
    tensor in transposed format.

    Parameters
    ----------
    tensor_a : tvm.te.Tensor
        data 2-D with shape [batch, in_dim]

    tensor_b : tvm.te.Tensor
        weight 2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]

    """
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = tensor_a.dtype
    batch, _, _, in_dim = tensor_a.shape
    red_dim, out_dim = tensor_b.shape

    # cmp should be done by values
    assert int(in_dim) == int(red_dim)

    k = te.reduce_axis((0, in_dim), name="k")
    compute_lambda = lambda i, h, w, j: te.sum(
        tensor_a[i, h, w, k].astype(out_dtype) * tensor_b[k, j].astype(out_dtype), axis=k
    )
    compute_name = "T_matmul_sliced"
    compute_tag = "matmul"

    mat = te.compute(
        (batch, 1, 1, out_dim),
        compute_lambda,
        name=compute_name,
        tag=compute_tag,
        attrs={"layout_free_placeholders": [tensor_b]},
    )

    if bias is not None:
        mat = te.compute(
            (batch, 1, 1, out_dim),
            lambda i, j: mat[i, h, w, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )

    return mat

def dense_schedule(outs, ins, output_layout: str, input_layout: str):
    """Schedule for dense op.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    if not isinstance(ins, list):
        ins = [ins]
    if not isinstance(outs, list):
        outs = [outs]

    func = te.create_prim_func([*ins, *outs])
    s = tir.Schedule(func)

    matmul = s.get_block("T_matmul_sliced")

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(matmul, ("read", 0), input_transform_fn)
    s.transform_layout(matmul, ("write", 0), output_transform_fn)

    bn, bh, bw, bc, rc = s.get_loops(matmul)
    bco, bci = s.split(bc, [None, 1024])
    s.vectorize(bci)

    return s
