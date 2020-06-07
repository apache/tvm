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
# pylint: disable=invalid-name, unused-variable
"""NN operator common utilities"""
from tvm.ir import container


def get_pad_tuple1d(padding):
    """Common code to get the 1 dimensional pad option
    Parameters
    ----------
    padding : Union[int, Tuple[int, ...]]
        Padding size
    Returns
    -------
    pad_left : int
        Padding size on left
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, container.Array):
        padding = list(padding)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            pad_w = padding[0] * 2
        elif len(padding) == 2:
            return padding[0], padding[1]
        else:
            raise ValueError("Size of padding can only be 1 or 2")
    elif isinstance(padding, int):
        pad_w = padding * 2
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_left = (pad_w + 1) // 2
    return pad_left, pad_w - pad_left


def get_pad_tuple2d(padding):
    """Common code to get the pad option
    Parameters
    ----------
    padding : Union[int, Tuple[int, ...]]
        Padding size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, container.Array):
        padding = list(padding)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def get_pad_tuple3d(padding):
    """Common code to get the pad option
    Parameters
    ----------
    padding : Union[int, Tuple[int, ...]]
        Padding size
    Returns
    -------
    pad_front : int
        Padding size on front
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_back : int
        Padding size on back
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, container.Array):
        padding = list(padding)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 3:
            pad_d = padding[0] * 2
            pad_h = padding[1] * 2
            pad_w = padding[2] * 2
        elif len(padding) == 6:
            return padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]
        else:
            raise ValueError("Size of padding can only be 3 or 6")
    elif isinstance(padding, int):
        pad_d = pad_h = pad_w = padding * 2
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_front = (pad_d + 1) // 2
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_front, pad_top, pad_left, pad_d - pad_front, pad_h - pad_top, pad_w - pad_left
