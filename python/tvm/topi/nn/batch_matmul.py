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
"""Batch matrix multiplication"""
# pylint: disable=invalid-name
import tvm
from tvm import te, auto_scheduler
from ..utils import get_const_tuple


def batch_matmul(x, y, oshape=None, auto_scheduler_rewritten_layout="", out_dtype=None):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch. Supports broadcasting for batch dimension.

    Parameters
    ----------
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]

    y : tvm.te.Tensor
        3-D with shape [batch, N, K]

    oshape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    x_shape = get_const_tuple(x.shape)
    if auto_scheduler_rewritten_layout:
        # Infer shape for the rewritten layout
        y_shape = auto_scheduler.get_shape_from_rewritten_layout(
            auto_scheduler_rewritten_layout, ["b", "j", "k"]
        )
        auto_scheduler.remove_index_check(y)
    else:
        y_shape = get_const_tuple(y.shape)
    assert len(x_shape) == 3 and len(y_shape) == 3, "only support 3-dim batch_matmul"

    XB = x_shape[0]
    YB = y_shape[0]
    _, M, K = x.shape
    k = te.reduce_axis((0, K), name="k")
    if oshape is None:
        assert XB == YB or XB == 1 or YB == 1, "batch dimension doesn't match"
        assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistent"
        batch = te.max(XB, YB)
        N = y.shape[1]
        oshape = (batch, M, N)

    if out_dtype is None or out_dtype == x.dtype:
        output = te.compute(
            oshape,
            lambda b, i, j: te.sum(
                x[b if XB != 1 else 0, i, k] * y[b if YB != 1 else 0, j, k], axis=k
            ),
            tag="batch_matmul",
            attrs={"layout_free_placeholders": [y]},
        )
    else:
        output = te.compute(
            oshape,
            lambda b, i, j: te.sum(
                x[b if XB != 1 else 0, i, k].astype(out_dtype)
                * y[b if YB != 1 else 0, j, k].astype(out_dtype),
                axis=k,
            ),
            tag="batch_matmul",
            attrs={"layout_free_placeholders": [y]},
        )

    if auto_scheduler_rewritten_layout:
        output = auto_scheduler.rewrite_compute_body(output, auto_scheduler_rewritten_layout)

    return output


@tvm.target.generic_func
def batch_matmul_legalize(attrs, inputs, types):
    """Legalizes batch_matmul op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current batch_matmul
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
