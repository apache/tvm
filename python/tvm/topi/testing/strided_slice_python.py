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
"""strided_slice/set in python"""


def strided_slice_python(data, begin, end, strides, slice_mode="end", axes=None):
    """Python version of strided slice operator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    begin : list
        Beginning of the slices.

    end : list
        End of the slices.

    strides : list
        The stride of each slice.

    slice_mode : str, optional
        The slice mode [end, size].
        end: The default slice mode, ending indices for the slice.
        size: The input strides will be ignored, input end in this mode indicates
              the sizeof a slice starting at the location specified by begin. If end[i] is -1,
              all remaining elements in that dimension are included in the slice.

    axes : list, optional
        Axes along which slicing is applied

    Returns
    -------
    result : numpy.ndarray
        The sliced result.
    """
    strides = [] if strides is None else strides
    if axes is not None:
        rank = len(data.shape)
        new_begin = [0] * rank
        new_end = [data.shape[i] for i in range(rank)]
        new_strides = [1] * rank

        for i, axis in enumerate(axes):
            new_begin[axis] = begin[i]
            new_end[axis] = end[i]
            if len(strides) > i:
                new_strides[axis] = strides[i]

        begin = new_begin
        end = new_end
        strides = new_strides

    slices = []
    for i in range(len(data.shape)):
        new_stride = None
        if slice_mode == "end" and i < len(strides):
            new_stride = strides[i]

        new_begin = begin[i] if i < len(begin) else None
        if i >= len(end):
            new_end = None
        elif slice_mode == "size":
            if end[i] < 0:
                new_end = None
            else:
                new_end = new_begin + end[i]
        else:
            new_end = end[i]

        slices.append(slice(new_begin, new_end, new_stride))

    return data[tuple(slices)]


def strided_set_python(data, v, begin, end, strides):
    """Python version of strided slice operator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    v : numpy.ndarray
        Value data

    begin : list
        Beginning of the slices.

    end : list
        End of the slices.

    strides : list
        The stride of each slice.

    Returns
    -------
    result : numpy.ndarray
        The updated result.
    """
    strides = [] if strides is None else strides
    slices = []
    res = data.copy()
    for i in range(len(data.shape)):
        slices.append(
            slice(
                begin[i] if i < len(begin) else None,
                end[i] if i < len(end) else None,
                strides[i] if i < len(strides) else None,
            )
        )
    res[tuple(slices)] = v
    return res
