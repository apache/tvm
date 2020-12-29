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
"""Slice axis in python"""


def slice_axis_python(data, axis, begin, end=None):
    """Slice input array along specific axis.

    Parameters
    ----------
    data : numpy.ndarray
        The source array to be sliced.

    axis : int
        Axis to be sliced.

    begin: int
        The index to begin with in the slicing.

    end: int, optional
        The index indicating end of the slice.

    Returns
    -------
    ret : numpy.ndarray
        The computed result.
    """
    dshape = data.shape
    if axis < 0:
        axis += len(dshape)
    if begin < 0:
        begin += dshape[axis]
    if end <= 0:
        end += dshape[axis]
    slc = [slice(None)] * len(dshape)
    slc[axis] = slice(begin, end)
    return data[tuple(slc)]
