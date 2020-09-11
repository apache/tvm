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
"""Yolo operations."""
from . import _make


def yolo_reorg(data, stride):
    """Yolo reorg operation used in darknet models.
    This layer shuffles the input tensor values based on the stride value.
    Along with the shuffling, it does the shape transform.
    If '(n, c, h, w)' is the data shape and 's' is stride, output shape is '(n, c*s*s, h/s, w/s)'.

    Example:

    .. code-block:: python

        data(1, 4, 2, 2) = [[[[ 0  1] [ 2  3]]
                            [[ 4  5] [ 6  7]]
                            [[ 8  9] [10 11]]
                            [[12 13] [14 15]]]]
        stride = 2
        ret(1, 16, 1, 1) = [[[[ 0]]  [[ 2]]  [[ 8]]  [[10]]
                            [[ 1]]  [[ 3]]  [[ 9]]  [[11]]
                            [[ 4]]  [[ 6]]  [[12]]  [[14]]
                            [[ 5]]  [[ 7]]  [[13]]  [[15]]]]

    .. note::

        stride=1 has no significance for reorg operation.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    stride : int
        The stride value for reorganisation.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.yolo_reorg(data, stride)
