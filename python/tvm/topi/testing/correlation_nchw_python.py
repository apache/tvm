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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Convolution 3D in python"""
import numpy as np


def correlation_nchw_python(
    data1, data2, kernel_size, max_displacement, stride1, stride2, padding, is_multiply
):
    """Correlationn operator in NCHW layout.

    Parameters
    ----------
    data1_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    data2_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel_size: int
        Kernel size for correlation, must be an odd number

    max_displacement: int
        Max displacement of Correlation

    stride1: int
        Stride for data1

    stride2: int
        Stride for data2 within the neightborhood centered around data1

    padding: int
        Padding for correlation

    is_multiply: bool
        operation type is either multiplication or substraction

    Returns
    -------
    c_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # compute output's dimension
    pad_data_height = data1.shape[2] + 2 * padding
    pad_data_width = data1.shape[3] + 2 * padding
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    out_width = (pad_data_width - border_size * 2) // stride1
    out_height = (pad_data_height - border_size * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    out_channel = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], out_channel, out_height, out_width))
    pad_data1 = np.zeros((data1.shape[0], data1.shape[1], pad_data_height, pad_data_width))
    pad_data2 = np.zeros((data1.shape[0], data1.shape[1], pad_data_height, pad_data_width))

    pad_data1[:, :, padding : padding + data1.shape[2], padding : padding + data1.shape[3]] = data1[
        :, :, :, :
    ]
    pad_data2[:, :, padding : padding + data2.shape[2], padding : padding + data2.shape[3]] = data2[
        :, :, :, :
    ]

    if is_multiply:
        corr_func = lambda x, y: x * y
    else:
        corr_func = lambda x, y: abs(x - y)

    # pylint: disable=too-many-nested-blocks
    for i in range(out_height):
        for j in range(out_width):
            for nbatch in range(data1.shape[0]):
                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for q in range(out_channel):
                    # location in data2
                    x2 = x1 + (q % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    y2 = y1 + (q // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                out[nbatch, q, i, j] += corr_func(
                                    pad_data1[nbatch, channel, y1 + h, x1 + w],
                                    pad_data2[nbatch, channel, y2 + h, x2 + w],
                                )

    out /= float(kernel_size**2 * data1.shape[1])
    return out
