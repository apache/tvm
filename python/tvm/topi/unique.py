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
from tvm import te, tir
from ..te import hybrid
from .scan import cumsum
from .sort import sort, argsort


def _calc_adjacent_diff_ir(data, output, binop=tir.Sub):
    """Low level IR to calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : Buffer
        Input 1-D Buffer.

    output: Buffer
        A buffer to store adjacent difference, of the same shape as data. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).

    binop: function, optional
        A binary associative op to use for calculating adjacent difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.
    """
    ib = tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    output_ptr = ib.buffer_ptr(output)
    with ib.for_range(0, data.shape[0], kind="parallel") as i:
        with ib.if_scope(i == 0):
            output_ptr[0] = 0
        with ib.else_scope():
            output_ptr[i] = tir.Cast(output.dtype, binop(data_ptr[i], data_ptr[i - 1]))
    return ib.get()


def _calc_adjacent_diff(data, out_dtype="int32", binop=tir.Sub):
    """Function calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input 1-D tensor.

    output_dtype : str
        The output tensor data type.

    binop: function, optional
        A binary associative op to use for calculating difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.

    Returns
    -------
    output : tvm.te.Tensor
        1-D tensor storing the adjacent difference of the input tensor. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).
    """
    return te.extern(
        [data.shape],
        [data],
        lambda ins, outs: _calc_adjacent_diff_ir(ins[0], outs[0], binop=binop),
        dtype=[out_dtype],
        name="_calc_adjacent_diff",
        tag="_calc_adjacent_diff_cpu",
    )


@hybrid.script
def _calc_num_unique(inc_scan):
    """Helper function to get the number of unique elements fron inc_scan tensor"""
    output = output_tensor((1,), "int32")
    output[0] = inc_scan[inc_scan.shape[0] - 1] + int32(1)
    return output


def _calc_unique_ir(
    data, argsorted_indices, inc_scan, index_converter, unique_elements, inverse_indices, counts
):
    """Low level IR to calculate unique elements, inverse indices, and counts (optional) of
    unique elements of 1-D array.

    Parameters
    ----------
    data : Buffer
        Input 1-D Buffer.

    argsorted_indices : Buffer
        A buffer that stores the argsorted indices of the input data.

    inc_scan : Buffer
        A buffer that stores the inclusive scan of the binary tir.NE adjacent difference
        of the sorted data.

    index_converter (optional) : Buffer
        An optional index converter that transforms the unique element index
        such that new_idx = index_converter[old_idx].

    unique_elements : Buffer
        A buffer that stores the unique elements.

    inverse_indices : Buffer
        A buffer that stores the index of each input data element in the unique element array.

    counts (optional) : Buffer
        A buffer that stores the count of each unique element.
    """
    ib = tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    unique_elements_ptr = ib.buffer_ptr(unique_elements)
    inverse_indices_ptr = ib.buffer_ptr(inverse_indices)

    index_converter_ptr = None
    if isinstance(index_converter, tir.Buffer):
        index_converter_ptr = ib.buffer_ptr(index_converter)

    if isinstance(counts, tir.Buffer):
        counts_ptr = ib.buffer_ptr(counts)
        # use indices_ptr as a tmp buffer to store tids with inc_scan[tid] != inc_scan[tid-1]
        unique_seq_indices_ptr = ib.buffer_ptr(inverse_indices)

    data_length = data.shape[0]

    # if need to return counts
    if isinstance(counts, tir.Buffer):
        num_unique = inc_scan_ptr[inc_scan.shape[0] - 1] + 1
        num_elements = data.shape[0]
        unique_seq_indices_ptr[num_unique - 1] = num_elements
        with ib.new_scope():
            with ib.for_range(0, data_length, kind="parallel") as i:
                with ib.if_scope(i > 0):
                    with ib.if_scope(inc_scan_ptr[i] != inc_scan_ptr[i - 1]):
                        unique_seq_indices_ptr[inc_scan_ptr[i] - 1] = i
        with ib.new_scope():
            with ib.for_range(0, num_unique, kind="parallel") as i:
                unique_idx = i if not index_converter_ptr else index_converter_ptr[i]
                with ib.if_scope(i == 0):
                    counts_ptr[unique_idx] = unique_seq_indices_ptr[i]
                with ib.else_scope():
                    counts_ptr[unique_idx] = (
                        unique_seq_indices_ptr[i] - unique_seq_indices_ptr[i - 1]
                    )
    # calculate unique elements and inverse indices
    with ib.new_scope():
        with ib.for_range(0, data_length, kind="parallel") as i:
            data_idx = argsorted_indices_ptr[i]
            unique_idx = (
                inc_scan_ptr[i] if not index_converter_ptr else index_converter_ptr[inc_scan_ptr[i]]
            )
            inverse_indices_ptr[data_idx] = unique_idx
            with ib.if_scope(i == 0):
                unique_elements_ptr[unique_idx] = data_ptr[data_idx]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[i] != inc_scan_ptr[i - 1]):
                    unique_elements_ptr[unique_idx] = data_ptr[data_idx]
    return ib.get()


@hybrid.script
def _calc_first_occurence(argsorted_indices, inc_scan):
    """Hybrid script to calculate the first occurence of each unique element in the input data.

    Parameters
    ----------
    argsorted_indices : tvm.te.Tensor
        A tensor that stores the argsorted indices of the input data.

    inc_scan : tvm.te.Tensor
        A tensor that stores the inclusive scan of the binary tir.NE adjacent difference
        of the sorted data.

    first_occurence : tvm.te.Tensor
        A tensor that stores the first occurence of each unique element in the input data.
    """
    first_occurence = output_tensor(argsorted_indices.shape, "int32")
    for i in parallel(argsorted_indices.shape[0]):
        first_occurence[i] = argsorted_indices.shape[0]
    for i in parallel(argsorted_indices.shape[0]):
        if i == 0 or inc_scan[i] != inc_scan[i - 1]:
            first_occurence[inc_scan[i]] = argsorted_indices[i]
    return first_occurence


def unique(data, is_sorted=True, return_counts=False):
    """
    Find the unique elements of a 1-D tensor. Please note `output` and `counts` are all padded to
    have the same length of `data` and element with index >= num_unique[0] has undefined value.

    Parameters
    ----------
    data : tvm.te.Tensor
        A 1-D tensor of integers.

    sorted : bool
        Whether to sort the unique elements in ascending order before returning as output.

    return_counts : bool
        Whether to return the count of each unique element.

    Returns
    -------
    unique : tvm.te.Tensor
        A 1-D tensor containing the unique elements of the input data tensor. The same size as
        the input data. If there are less unique elements than input data, the end of the tensor
        is padded with zeros.

    indices : tvm.te.Tensor
        A 1-D tensor. The same size as output. For each entry in output, it contains
        the index of its first occurence in the input data. The end of the tensor is padded
        with the length of the input data.

    inverse_indices : tvm.te.Tensor
        A 1-D tensor. For each entry in data, it contains the index of that data element in
        the unique array. (Note that inverse_indices is very similar to indices if output is not
        sorted.)

    num_unique : tvm.te.Tensor
        A 1-D tensor with size=1 containing the number of unique elements in the input data tensor.

    counts (optional) : tvm.te.Tensor
        A 1-D tensor containing the count of each unique element in the output.

    Examples
    --------
    .. code-block:: python

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, False)
        output          =  [4, 5, 1, 2, 3, _, _, _]
        indices         =  [0, 1, 2, 3, 4, _, _, _]
        inverse_indices =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique      =  [5]

        [output, indices, num_unique, counts] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, True)
        output          =  [4, 5, 1, 2, 3, _, _, _]
        indices         =  [0, 1, 2, 3, 4, _, _, _]
        inverse_indices =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique      =  [5]
        counts          =  [2, 2, 1, 1, 2, _, _, _]

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], True)
        output          =  [1, 2, 3, 4, 5, _, _, _]
        indices         =  [2, 3, 4, 0, 1, _, _, _]
        inverse_indices =  [3, 4, 0, 1, 2, 2, 3, 4]
        num_unique      =  [5]
    """
    sorted_data = sort(data)
    argsorted_indices = argsort(data, dtype="int32")
    # adjacent difference
    adjacent_diff = _calc_adjacent_diff(sorted_data, "int32", tir.NE)
    # inclusive scan
    inc_scan = cumsum(adjacent_diff, dtype="int32", exclusive=0)
    # total number of unique elements
    num_unique_elements = _calc_num_unique(inc_scan)
    # prepare outputs
    if return_counts:
        out_data_shape = [data.shape] * 3
        out_dtypes = [data.dtype, "int32", "int32"]
    else:
        out_data_shape = [data.shape] * 2
        out_dtypes = [data.dtype, "int32"]
    # prepare inputs and fcompute

    first_occurence = _calc_first_occurence(argsorted_indices, inc_scan)
    if is_sorted:
        in_data = [data, argsorted_indices, inc_scan]
        if return_counts:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, None, *outs)
        else:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, None, *outs, None)

        indices = first_occurence
    else:
        # calculate index converter by sorting unique elements by their first occurence
        argsorted_first_occurence = argsort(first_occurence, dtype="int32")
        index_converter = argsort(argsorted_first_occurence, dtype="int32")
        in_data = [data, argsorted_indices, inc_scan, index_converter]
        if return_counts:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs)
        else:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs, None)
        # First occurence is in order of sorted unique output, if we sort the first_occurence array
        # we get the correct result
        indices = sort(first_occurence)

    outs = te.extern(
        out_data_shape,
        in_data,
        fcompute,
        dtype=out_dtypes,
        name="_calc_unique",
        tag="_calc_unique_cpu",
    )
    if return_counts:
        return [outs[0], indices, outs[1], num_unique_elements, outs[2]]
    return [outs[0], indices, outs[1], num_unique_elements]
