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
# pylint: disable=invalid-name
"""Unique operator"""
from ..te import hybrid
from .cumsum import cumsum
from .sort import sort, argsort


@hybrid.script
def _calc_adjacent_diff(data):
    output = output_tensor(data.shape, "int32")
    output[0] = int32(0)
    for i in parallel(1, data.shape[0]):
        output[i] = int32(1) if data[i] != data[i - 1] else int32(0)
    return output


@hybrid.script
def _calc_num_unique(data):
    output = output_tensor((1,), "int32")
    output[0] = data[data.shape[0] - 1] + int32(1)
    return output


@hybrid.script
def _calc_unique_sorted(data, argsorted_indices, inc_scan):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    for i in parallel(data.shape[0]):
        indices[argsorted_indices[i]] = inc_scan[i]
        unique_elements[inc_scan[i]] = data[argsorted_indices[i]]
    return unique_elements, indices


@hybrid.script
def _calc_unique_sorted_with_counts(data, argsorted_indices, inc_scan):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    counts = output_tensor(data.shape, "int32")
    for i in parallel(data.shape[0]):
        counts[i] = int32(0)
    for i in parallel(data.shape[0]):
        indices[argsorted_indices[i]] = inc_scan[i]
        unique_elements[inc_scan[i]] = data[argsorted_indices[i]]
    for i in range(data.shape[0]):
        counts[inc_scan[i]] += int32(1)
    return unique_elements, indices, counts


@hybrid.script
def _calc_first_occurence(argsorted_indices, inc_scan):
    first_occurence = output_tensor(argsorted_indices.shape, "int32")
    for i in parallel(argsorted_indices.shape[0]):
        first_occurence[i] = argsorted_indices.shape[0]
    for i in parallel(argsorted_indices.shape[0]):
        if i == 0 or inc_scan[i] != inc_scan[i - 1]:
            first_occurence[inc_scan[i]] = argsorted_indices[i]
    return first_occurence


@hybrid.script
def _calc_unique_unsorted(data, argsorted_indices, inc_scan, index_converter):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    for i in parallel(data.shape[0]):
        new_unique_idx = index_converter[inc_scan[i]]
        new_data_idx = argsorted_indices[i]
        unique_elements[new_unique_idx] = data[new_data_idx]
        indices[new_data_idx] = new_unique_idx
    return unique_elements, indices


@hybrid.script
def _calc_unique_unsorted_with_counts(data, argsorted_indices, inc_scan, index_converter):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    counts = output_tensor(data.shape, "int32")
    for i in parallel(data.shape[0]):
        counts[i] = int32(0)
    for i in parallel(data.shape[0]):
        new_unique_idx = index_converter[inc_scan[i]]
        new_data_idx = argsorted_indices[i]
        unique_elements[new_unique_idx] = data[new_data_idx]
        indices[new_data_idx] = new_unique_idx
    for i in range(data.shape[0]):
        idx = index_converter[inc_scan[i]]
        counts[idx] += int32(1)
    return unique_elements, indices, counts


def unique(data, is_sorted=True, return_counts=False):
    """
    Find the unique elements of a tensor
    Parameters
    ----------
    data : relay.Expr
        A 1-D tensor of integers
    sorted : bool
        Whether to sort the unique elements in ascending order before returning as output
    return_counts : bool
        Whether to return the array with count of each unique element
    Returns
    -------
    output : relay.Expr
        A 1-D tensor containing the unique elements of the input data tensor
    indices : relay.Expr
        A 1-D tensor containing the index of each data element in the output tensor
    num_unique : relay.Expr
        A 0-D tensor containing the number of unique elements in the input data tensor
    counts (optional) : relay.Expr
        A 1-D tensor containing the count of each unique element in the output
    Examples
    --------
    .. code-block:: python
        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], sorted=False, return_counts=False)
        output         =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]

        [output, indices, num_unique, counts] = unique([4, 5, 1, 2, 3, 3, 4, 5], sorted=False, return_counts=True)
        output         =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]
        counts         =  [2, 2, 1, 1, 2, ?, ?, ?]

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], sorted=True)
        output         =  [1, 2, 3, 4, 5, ?, ?, ?]
        indices        =  [3, 4, 0, 1, 2, 2, 3, 4]
        num_unique     =  [5]
    """

    sorted_data = sort(data)
    argsorted_indices = argsort(data, dtype="int32")
    adjacent_diff = _calc_adjacent_diff(sorted_data)
    inc_scan = cumsum(adjacent_diff, dtype="int32", exclusive=0)
    num_unique_elements = _calc_num_unique(inc_scan)
    if is_sorted:
        if return_counts:
            unique_elements, inverse_indices, counts = _calc_unique_sorted_with_counts(
                data, argsorted_indices, inc_scan
            )
            return [unique_elements, inverse_indices, num_unique_elements, counts]
        else:
            unique_elements, inverse_indices = _calc_unique_sorted(
                data, argsorted_indices, inc_scan
            )
            return [unique_elements, inverse_indices, num_unique_elements]
    else:
        first_occurence = _calc_first_occurence(argsorted_indices, inc_scan)
        argsorted_first_occurence = argsort(first_occurence, dtype="int32")
        index_converter = argsort(argsorted_first_occurence, dtype="int32")
        if return_counts:
            unique_elements, inverse_indices, counts = _calc_unique_unsorted_with_counts(
                data, argsorted_indices, inc_scan, index_converter
            )
            return [unique_elements, inverse_indices, num_unique_elements, counts]
        else:
            unique_elements, inverse_indices = _calc_unique_unsorted(
                data, argsorted_indices, inc_scan, index_converter
            )
            return [unique_elements, inverse_indices, num_unique_elements]
