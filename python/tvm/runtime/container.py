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
"""Runtime container structures."""

def getitem_helper(obj, elem_getter, length, idx):
    """Helper function to implement a pythonic getitem function.

    Parameters
    ----------
    obj: object
        The original object

    elem_getter : function
        A simple function that takes index and return a single element.

    length : int
        The size of the array

    idx : int or slice
        The argument passed to getitem

    Returns
    -------
    result : object
        The result of getitem
    """
    if isinstance(idx, slice):
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else length
        step = idx.step if idx.step is not None else 1
        if start < 0:
            start += length
        if stop < 0:
            stop += length
        return [elem_getter(obj, i) for i in range(start, stop, step)]

    if idx < -length or idx >= length:
        raise IndexError("Index out of range. size: {}, got index {}"
                         .format(length, idx))
    if idx < 0:
        idx += length
    return elem_getter(obj, idx)
