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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Scatter operator"""
from ..tir import decl_buffer, ir_builder, Cast, AssertStmt, StringImm, Evaluate
from ..te import extern, hybrid


@hybrid.script
def _sparse_fill_empty_rows(sparse_indices, sparse_values, default_value, dense_shape):
    out = output_tensor((dense_shape[0],), "int64")
    for i in range(dense_shape[0]):
        out[i] = int64(1)
    return out

def sparse_fill_empty_rows(sparse_indices, sparse_values, default_value, dense_shape):
    """Update data at positions defined by indices with values in updates

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    axis : int
        The axis to scatter on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _sparse_fill_empty_rows(sparse_indices, sparse_values, default_value, dense_shape)
    # if axis < 0:
    #     axis += len(data.shape)
    # assert axis >= 0
    # assert axis < len(data.shape)

    # if len(data.shape) == 1:
    #     return _scatter_1d(data, indices, updates)
    # if len(data.shape) == 2:
    #     return _scatter_2d(data, indices, updates, axis)
    # if len(data.shape) == 3:
    #     return _scatter_3d(data, indices, updates, axis)
    # if len(data.shape) == 4:
    #     return _scatter_4d(data, indices, updates, axis)
    # raise ValueError("scatter only support for 1-4 dimensions")