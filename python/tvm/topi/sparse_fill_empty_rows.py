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
# "AS IS" BASIS, WITHnew_sparse_indices WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=no-else-return, too-many-locals, too-many-arguments, too-many-branches
# pylint: disable=undefined-variable, invalid-name
"""SparseFillEmptyRows operator"""
from ..te import hybrid


@hybrid.script
def _sparse_fill_empty_rows(
    sparse_indices,
    sparse_values,
    dense_shape,
    default_value,
    new_sparse_indices_shape,
    new_sparse_values_shape,
    empty_row_indicator_shape,
):
    default_value_ = int64(default_value[0])
    new_sparse_indices = output_tensor(new_sparse_indices_shape, sparse_indices.dtype)
    new_sparse_values = output_tensor(new_sparse_values_shape, sparse_values.dtype)
    empty_row_indicator = output_tensor(empty_row_indicator_shape, "int64")
    idx = 0

    if int64(sparse_indices.shape[0]) == int64(0):
        for i in range(0, int64(new_sparse_indices_shape[0])):
            new_sparse_indices[i, 0] = int64(i)
            new_sparse_values[i] = default_value_
            empty_row_indicator[i] = int64(1)
            for k in range(1, int64(new_sparse_indices_shape[1])):
                new_sparse_indices[i, k] = int64(0)

        return (new_sparse_indices, new_sparse_values, empty_row_indicator)

    else:
        for i in range(0, int64(sparse_indices[0, 0])):
            new_sparse_indices[idx, 0] = int64(i)
            for k in range(1, int64(new_sparse_indices_shape[1])):
                new_sparse_indices[idx, k] = int64(0)

            new_sparse_values[idx] = default_value_
            empty_row_indicator[i] = int64(1)
            idx += 1

        for i in range(0, int64(sparse_indices.shape[0])):
            index = int64(sparse_indices[i, 0])
            if i == 0:
                new_sparse_indices[idx, 0] = index
                for k in range(1, int64(new_sparse_indices_shape[1])):
                    new_sparse_indices[idx, k] = int64(sparse_indices[i, k])
                new_sparse_values[idx] = int64(sparse_values[i])
                empty_row_indicator[index] = int64(0)
                idx += 1
            else:
                prev_index = int64(sparse_indices[i - 1, 0] + 1)
                for j in range(prev_index, index):
                    new_sparse_indices[idx, 0] = int64(j)
                    for k in range(1, int64(new_sparse_indices_shape[1])):
                        new_sparse_indices[idx, k] = int64(0)
                    empty_row_indicator[prev_index] = int64(1)
                    new_sparse_values[idx] = default_value_
                    idx += 1

                new_sparse_indices[idx, 0] = index
                for k in range(1, int64(new_sparse_indices_shape[1])):
                    new_sparse_indices[idx, k] = int64(sparse_indices[i, k])
                new_sparse_values[idx] = int64(sparse_values[i])
                empty_row_indicator[index] = int64(0)
                idx += 1

        for i in range(
            int64(sparse_indices[sparse_indices.shape[0] - 1, 0] + 1), int64(dense_shape[0])
        ):

            new_sparse_indices[idx, 0] = int64(i)
            for k in range(1, int64(new_sparse_indices_shape[1])):
                new_sparse_indices[idx, k] = int64(0)
            empty_row_indicator[i] = int64(1)
            new_sparse_values[idx] = default_value_
            idx += 1

        return (new_sparse_indices, new_sparse_values, empty_row_indicator)


def sparse_fill_empty_rows(
    sparse_indices,
    sparse_values,
    dense_shape,
    default_value,
    new_sparse_indices_shape,
    new_sparse_values_shape,
    empty_row_indicator_shape,
):
    return _sparse_fill_empty_rows(
        sparse_indices,
        sparse_values,
        dense_shape,
        default_value,
        new_sparse_indices_shape,
        new_sparse_values_shape,
        empty_row_indicator_shape,
    )
