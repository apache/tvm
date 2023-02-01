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


def qdense_compute(
    tensor_a,
    tensor_b,
    zero_a,
    scale_a,
    zero_b,
    scale_b,
    zero_out=None,
    scale_out=None,
    bias=None,
    q_dtype=None,
):
    """Hexagon's implementation of a sliced dense operator in Topi.
    Uses matmul.

    Parameters
    ----------
    tensor_a : tvm.te.Tensor
        data 2-D with shape [batch, in_dim]

    tensor_b : tvm.te.Tensor
        weight 2-D with shape [in_dim, out_dim]

    zero_a : integer
        quantization zero point for tensor a.

    scale_a : float
        quantization scale for tensor a.

    zero_b : integer
        quantization zero point for tensor b.

    scale_b : float
        quantization scale for tensor b.

    zero_out : Optional[integer]
        quantization zero point for output.

    scale_out : Optional[float]
        quantization scale for output.

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    q_dtype : Optional[str]
        The output type.

    Returns
    -------
    mat : tvm.te.Tensor
        2-D with shape [batch, out_dim]

    """
    if bias is not None:
        assert len(bias.shape) == 1
    if q_dtype is None:
        q_dtype = tensor_a.dtype

    batch, in_dim = tensor_a.shape
    out_dim, red_dim = tensor_b.shape

    # cmp should be done by values
    assert int(in_dim) == int(red_dim)

    k = te.reduce_axis((0, in_dim), name="k")
    compute_lambda = lambda n, m: te.sum(
        scale_a
        * (tensor_a[n, k].astype("float32") - zero_a)
        * scale_b
        * (tensor_b[k, m].astype("float32") - zero_b),
        axis=k,
    )
    compute_name = "qmatmul_sliced"

    out = te.compute(
        (batch, out_dim),
        compute_lambda,
        name=compute_name,
        attrs={"layout_free_placeholders": [tensor_b]},
    )

    if bias is not None:
        out = te.compute(
            (batch, out_dim),
            lambda i, j: out[i, j] + bias[j],
            tag=tag.BROADCAST,
            name="bias",
        )

    # Requantization of dense
    if scale_out is not None:
        out = te.compute(
            (batch, out_dim),
            lambda *i: (out[i] / scale_out + zero_out).astype(q_dtype),
            name="requantize",
        )

    return out


def qdense_schedule(outs, ins, output_layout: str, input_layout: str):
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

    matmul = s.get_block("qmatmul_sliced")
    try:
        requantize = s.get_block("requantize")
    except tir.schedule.schedule.ScheduleError:
        requantize = None
    try:
        bias = s.get_block("bias")
    except tir.schedule.schedule.ScheduleError:
        bias = None

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)

    # Transform input and output buffer
    s.transform_layout(matmul, ("read", 0), input_transform_fn)
    if requantize is not None:
        s.transform_layout(requantize, ("write", 0), output_transform_fn)
    elif bias is not None:
        s.transform_layout(bias, ("write", 0), output_transform_fn)
    else:
        s.transform_layout(matmul, ("write", 0), output_transform_fn)

    # Vectorize
    _, matmul_c, _ = s.get_loops(matmul)
    _, matmul_c_inner = s.split(matmul_c, [None, 128])
    s.vectorize(matmul_c_inner)

    # Compute everything inline
    if bias is not None and requantize is not None:
        _, bias_c = s.get_loops(bias)
        s.compute_at(matmul, bias_c)
        _, out_c = s.get_loops(requantize)
        s.compute_at(bias, out_c)
    elif bias is not None and requantize is None:
        _, out_c = s.get_loops(bias)
        s.compute_at(matmul, out_c)

    return s
