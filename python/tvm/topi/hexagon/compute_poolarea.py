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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals

"""Compute PoolArea size which is used to exclude the zero-padding elements in the averaging
   calculation.
"""

from tvm import te, tir


def compute_PoolArea(i, j, ih, iw, kh, kw, sh, sw, dh, dw, pad_top, pad_left):
    """
    Parameters
    ----------
    i,j:
        index of output tensor along H and W axis
        This is equal to the starting point of the sliding window for which the average is computed
    ih, iw:
        input data size along H and W axis
    kh, kw:
        Kernel size along H and W axis
    sh, sw:
        Stride size along H and W axis
    dh, dw:
        Dilation size along H and W axis
    pad_top, pad_left:
        Pad size on Top and left side of input data

    # PoolArea refers to the area of that portion of each sliding window which only includes
    # the input data and not the padded area.

    # Motivation: The following example shows the location of the first sliding window (at i=0, j=0)
    # on a 6*6 array, with kernel=[3,3] and padding=[1, 1, 1, 1].
    # The input data elements are shown with (X) and padding data with (0).
    # As shown, the number of non-padding elements that should be used for computing
    # the average of values inside this window is 4, while the windows area is 3*3=9.
    # To compute the PoolArea, we have to move the top/left edge of the window down/right
    # to exclude zero-padding elements. The edge adjustment can be formulated as
    #    top_edge = max(i , pad_top)
    #    left_edge= max(j , pad_left)
    # Note that pad_top and pad_left represent point 0 of the input data along i and j direction.
    # In this example, bottom_edge and right_edge of the PoolArea do not need any adjustment,
    # because there is no padding data on those side of the window.
    # However, as we slide the window down and to the right, the window might go
    # beyond the input data boundaries (ih and iw). In these cases, bottom/right edge should be
    # moved up/left to be located inside the input data.
    # This can be formulated as
    #    bottom_edge = min(i + kh, ih + pad_top)
    #    left_edge   = min(j + kw, iw + pad_left)
    # Having all the edges,
    #    PoolArea = (bottom_edge - top_edge) * (right_edge - left_edge)

    #    _______
    #    |0 0 0|0 0 0 0 0                         0 0 0 0 0 0 0 0
    #    |     |                                 _______
    #    |0 X X|X X X X 0                        |0 X X|X X X X 0
    #    |     |                                 |     |
    #    |0 X X|X X X X 0        ====>           |0 X X|X X X X 0
    #    |_____|                                 |_____|
    #    0 X X X X X X 0                          0 X X X X X X 0
    #    0 X X X X X X 0                          0 X X X X X X 0
    #    0 X X X X X X 0                          0 X X X X X X 0
    #    0 X X X X X X 0                          0 X X X X X X 0
    #    0 0 0 0 0 0 0 0                          0 0 0 0 0 0 0 0


    # The above equations are derived under the assumption of having default value (1)
    # for stride and dilation. However, we need to expand them to support non-default
    # stride and dilation values.
    # Stride impacts the starting location of the sliding windows, so i and j should be
    # replaced by (i * sh) and j by (j * sw) in the equations.
    # Dilation changes the window size, making k kernel elements scattered into a d*(k - 1) + 1
    # window.
    # Non-1 dilation means that, we need to divide the adjusted window size by the dilation value
    # to find out how many kernel elements inside the sliding window are inside the input data
    # boundaries:
    #    top_edge= max(i * sh , pad_top)
    #    left_edge= max(j * sw , pad_left)
    #    bottom_edge = min(i * sh + (kh - 1) * dh + 1, ih + pad_top)
    #    left_edge   = min(j * sw + (kw - 1) * dw + 1, data_w + pad_left)
    #    PoolArea = ceil_div((bottom_edge - top_edge), dh) * ceil_div((right_edge - left_edge), dw)
    #
    # Finally, we need to address one corner case related to the non-default dilation:
    # Consider the following example along W axis, where iw = 3, kw = 3 and dw = 2.
    # The first figure on the left shows the sliding window of size 5 starting at index 0,
    # and the first figure on the right shows the same example with sliding window at index 1.
    # The second row of figures show the PoolArea after adjusting the edges
    # (both left_edge - right_edge = 3)
    # The third row of figures show the location of dialated kernel points(*).
    # As shown, although the distance between left and right edge in both cases is 3 and
    # dilation is 2 and ceil_div(3,2)=2, the right PoolArea only includes 1 kernel point.

    #  Sliding Window:                       |0 0 X X X |0                         0 |0 X X X  0|
    #  PoolArea(after edge adjustment):       0 0|X X X |0                         0  0|X X X| 0
    #  location of dilated kernel points:     * 0|* X * |0                         0  *|X * X| 0
    #  PoolArea (dilated_point_aware):        * 0|* X * |0                         0  * X|* X| 0

    # To address this issue, instead of moving the left_edge to bring it just inside the input
    # data boundary, we should move the edge to the right untill we get to the first dilated kernel
    # point inside the input data boundary.
    # The third row of figures shows how this row adjustment can solve the problem.
    # So the problem is reduced to finding the first dilated kernel point inside the data
    # boundary.# For that, we can find the number of dialted points which are mapped to the padded
    # area and find the location of the next one which should be inside the input data:
    #    num_of_prev_points = (pad_top - i * sh - 1) // dh
    #    next_point_index = i * sh + (num_prev_points + 1) * dh
    #
    # With that, Top_edge and left_edge can be reformulated as:
    #    if i*sh - pad_top < 0:
    #        top_edge = i * sh + ((pad_top - i * sh - 1) // dh + 1) * dh
    #    else:
    #        top_edge = i * sh
    #
    #    if j * sw - pad_left < 0:
    #        left_edge = j * sw + ((pad_left - j * sw - 1) // dw + 1) * dw
    #    else:
    #        left_edge= j * sw

    """
    top_edge = tir.if_then_else(
        tir.all(i * sh - pad_top < 0), i * sh + ((pad_top - i * sh - 1) // dh + 1) * dh, i * sh
    )
    bottom_edge = te.min(i * sh + (kh - 1) * dh + 1, ih + pad_top)
    left_edge = tir.if_then_else(
        tir.all(j * sw - pad_left < 0), j * sw + ((pad_left - j * sw - 1) // dw + 1) * dw, j * sw
    )
    right_edge = te.min(j * sw + (kw - 1) * dw + 1, iw + pad_left)
    return -((bottom_edge - top_edge) // -dh) * -((right_edge - left_edge) // -dw)
