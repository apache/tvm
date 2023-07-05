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

from tvm import te, tir
from tvm.topi import tag
from ..utils import get_layout_transform_fn


def dense_compute(tensor_a, tensor_b, bias=None, out_dtype=None):
    """Hexagon's implementation of a sliced dense operator in Topi.
    Uses matmul.

    Parameters
    ----------
    tensor_a : tvm.te.Tensor
        data 2-D with shape [batch, in_dim]

    tensor_b : tvm.te.Tensor
        weight 2-D with shape [in_dim, out_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]

    """
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = tensor_a.dtype

    batch, in_dim = tensor_a.shape
    out_dim, red_dim = tensor_b.shape

    # cmp should be done by values
    assert int(in_dim) == int(red_dim)

    k = te.reduce_axis((0, in_dim), name="k")
    compute_lambda = lambda n, m: te.sum(
        tensor_a[n, k].astype(out_dtype) * tensor_b[k, m].astype(out_dtype), axis=k
    )
    compute_name = "matmul_sliced"
    compute_tag = "matmul"

    mat = te.compute(
        (batch, out_dim),
        compute_lambda,
        name=compute_name,
        tag=compute_tag,
        attrs={"layout_free_placeholders": [tensor_b]},
    )

    if bias is not None:
        mat = te.compute(
            (batch, out_dim),
            lambda i, j: mat[i, j] + bias[j],
            tag=tag.BROADCAST,
            name="bias",
        )

    return mat


def dense_schedule(outs, ins, output_layout: str, input_layout: str):
    """Schedule for dense op.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense in the format
        of an array of tensors.

    ins: Array of Tensor
        Input tensors into graph.

    output_layout: str
        Descriptor string for physical layout

    input_layout: str
        Descriptor string for physical layout

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

    matmul = s.get_block("matmul_sliced")
    try:
        bias = s.get_block("bias")
    except tir.schedule.schedule.ScheduleError:
        bias = None

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)

    # No bias
    if bias is None:
        s.transform_layout(matmul, ("read", 0), input_transform_fn)
        # s.transform_layout(matmul, ("read", 1), input_transform_fn)
        s.transform_layout(matmul, ("write", 0), output_transform_fn)
    else:
        s.transform_layout(matmul, ("read", 0), input_transform_fn)
        s.transform_layout(bias, ("write", 0), output_transform_fn)

    _, matmul_c, _ = s.get_loops(matmul)
    _, matmul_c_inner = s.split(matmul_c, [None, 64])
    s.vectorize(matmul_c_inner)

    if bias is not None:
        _, bias_c = s.get_loops(bias)
        _, bias_c_inner = s.split(bias_c, [None, 64])
        s.vectorize(bias_c_inner)

    return s
