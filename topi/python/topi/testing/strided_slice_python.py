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
"""gather_nd in python"""

def strided_slice_python(data, begin, end, strides):
    """Python version of strided slice operator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    begin : list
        Begining of the slices.

    end : list
        End of the slices.

    strides : list
        The stride of each slice.

    Returns
    -------
    result : numpy.ndarray
        The sliced result.
    """
    strides = [] if strides is None else strides
    slices = []
    for i in range(len(data.shape)):
        slices.append(slice(
            begin[i] if i < len(begin) else None,
            end[i] if i < len(end) else None,
            strides[i] if i < len(strides) else None))
    return data[tuple(slices)]
