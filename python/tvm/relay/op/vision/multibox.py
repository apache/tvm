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
"""Multibox operations."""
from tvm.relay import expr
from . import _make


def multibox_prior(
    data, sizes=(1.0,), ratios=(1.0,), steps=(-1.0, -1.0), offsets=(0.5, 0.5), clip=False
):
    """Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    sizes : tuple of float, optional
        Tuple of sizes for anchor boxes.

    ratios : tuple of float, optional
        Tuple of ratios for anchor boxes.

    steps : Tuple of float, optional
        Priorbox step across y and x, -1 for auto calculation.

    offsets : tuple of int, optional
        Priorbox center offsets, y and x respectively.

    clip : boolean, optional
        Whether to clip out-of-boundary boxes.

    Returns
    -------
    out : relay.Expr
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    return _make.multibox_prior(data, sizes, ratios, steps, offsets, clip)


def multibox_transform_loc(
    cls_prob, loc_pred, anchor, clip=True, threshold=0.01, variances=(0.1, 0.1, 0.2, 0.2)
):
    """Location transformation for multibox detection

    Parameters
    ----------
    cls_prob : tvm.relay.Expr
        Class probabilities.

    loc_pred : tvm.relay.Expr
        Location regression predictions.

    anchor : tvm.relay.Expr
        Prior anchor boxes.

    clip : boolean, optional
        Whether to clip out-of-boundary boxes.

    threshold : double, optional
        Threshold to be a positive prediction.

    variances : Tuple of float, optional
        variances to be decoded from box regression output.

    Returns
    -------
    ret : tuple of tvm.relay.Expr
    """
    return expr.TupleWrapper(
        _make.multibox_transform_loc(cls_prob, loc_pred, anchor, clip, threshold, variances), 2
    )
