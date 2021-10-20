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
"""Scatter Add operator"""
from tvm.te import hybrid


@hybrid.script
def _scatter_add_1d(data, indices, updates):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        out[i] = data[i]
    for i in range(indices.shape[0]):
        out[indices[i] if indices[i] >= 0 else indices[i] + data.shape[0]] += updates[i]
    return out


@hybrid.script
def _scatter_add_2d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = data[i, j]
    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                out[
                    indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis], j
                ] += updates[i, j]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                out[
                    i, indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis]
                ] += updates[i, j]

    return out


@hybrid.script
def _scatter_add_3d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                out[i, j, k] = data[i, j, k]
    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                        j,
                        k,
                    ] += updates[i, j, k]
    elif axis == 1:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        i,
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                        k,
                    ] += updates[i, j, k]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        i,
                        j,
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                    ] += updates[i, j, k]

    return out


@hybrid.script
def _scatter_add_4d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                for l in range(data.shape[3]):
                    out[i, j, k, l] = data[i, j, k, l]

    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            j,
                            k,
                            l,
                        ] += updates[i, j, k, l]
    elif axis == 1:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            k,
                            l,
                        ] += updates[i, j, k, l]
    elif axis == 2:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            j,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            l,
                        ] += updates[i, j, k, l]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            j,
                            k,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                        ] += updates[i, j, k, l]

    return out


def scatter_add(data, indices, updates, axis=0):
    """Update data by adding values in updates at positions defined by indices

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    axis : int
        The axis to scatter_add on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    if axis < 0:
        axis += len(data.shape)
    assert axis >= 0
    assert axis < len(data.shape)

    if len(data.shape) == 1:
        return _scatter_add_1d(data, indices, updates)
    if len(data.shape) == 2:
        return _scatter_add_2d(data, indices, updates, axis)
    if len(data.shape) == 3:
        return _scatter_add_3d(data, indices, updates, axis)
    if len(data.shape) == 4:
        return _scatter_add_4d(data, indices, updates, axis)
    raise ValueError("scatter_add only support for 1-4 dimensions")
