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
# pylint: disable=invalid-name,unused-argument
"""TVM operator fully connected compute."""
import tvm
from tvm import te, auto_scheduler
from .. import tag


def matmul(
    data,
    weight,
    bias=None,
    out_dtype=None,
    data_transposed=False,
    weight_transposed=False,
    auto_scheduler_rewritten_layout="",
):
    """The default implementation of matmul in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    data_transposed : Optional[bool]
        Whether the data tensor is in transposed format.

    weight_transposed : Optional[bool]
        Whether the weight tensor is in transposed format.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    if data_transposed:
        in_dim, batch = data.shape
    else:
        batch, in_dim = data.shape

    if auto_scheduler_rewritten_layout:
        # Infer shape for the rewritten layout
        out_dim, red_dim = auto_scheduler.get_shape_from_rewritten_layout(
            auto_scheduler_rewritten_layout, ["j", "k"] if weight_transposed else ["k", "j"]
        )
        auto_scheduler.remove_index_check(weight)
    elif weight_transposed:
        out_dim, red_dim = weight.shape
    else:
        red_dim, out_dim = weight.shape
    assert in_dim == red_dim

    k = te.reduce_axis((0, in_dim), name="k")
    if data_transposed:
        if weight_transposed:
            compute_lambda = lambda i, j: te.sum(
                data[k, i].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k
            )
            compute_name = "T_matmul_TT"
        else:
            compute_lambda = lambda i, j: te.sum(
                data[k, i].astype(out_dtype) * weight[k, j].astype(out_dtype), axis=k
            )
            compute_name = "T_matmul_TN"
        compute_tag = "matmul"
    else:
        if weight_transposed:
            compute_lambda = lambda i, j: te.sum(
                data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k
            )
            compute_name = "T_dense"
            compute_tag = "dense"
        else:
            compute_lambda = lambda i, j: te.sum(
                data[i, k].astype(out_dtype) * weight[k, j].astype(out_dtype), axis=k
            )
            compute_name = "T_matmul"
            compute_tag = "matmul"

    mat = te.compute(
        (batch, out_dim),
        compute_lambda,
        name=compute_name,
        tag=compute_tag,
        attrs={"layout_free_placeholders": [weight]},
    )

    if bias is not None:
        mat = te.compute(
            (batch, out_dim),
            lambda i, j: mat[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )

    if auto_scheduler_rewritten_layout:
        mat = auto_scheduler.rewrite_compute_body(mat, auto_scheduler_rewritten_layout)

    return mat


@tvm.target.generic_func
def matmul_legalize(attrs, inputs, types):
    """Legalizes matmul op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current dense
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # not to change by default
    # pylint: disable=unused-argument
    return None


def dense(data, weight, bias=None, out_dtype=None, auto_scheduler_rewritten_layout=""):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

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
    return matmul(data, weight, bias, out_dtype, False, True, auto_scheduler_rewritten_layout)


@tvm.target.generic_func
def dense_legalize(attrs, inputs, types):
    """Legalizes dense op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current dense
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # not to change by default
    # pylint: disable=unused-argument
    return None


def dense_pack(data, weight, bias=None, out_dtype=None):
    """The default implementation of dense_pack in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
    N = N * packw_bn

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * weight[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        name="T_dense_pack",
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@tvm.target.generic_func
def dense_alter_layout(attrs, inputs, tinfos, out_type):
    """Change dense layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None
