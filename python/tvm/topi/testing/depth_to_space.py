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
"""Depth to space in python"""
import numpy as np


def depth_to_space_python(data, block_size, mode="DCR"):
    """Depth to Space operator in python for NCHW layout.

    Parameters
    ----------
    data : np.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    block_size : int
        Size of blocks to convert channel pixels into.

    Returns
    -------
    d2s_out : np.ndarray
        4-D with shape [batch, in_channel / (block_size * block_size),
                        out_height * block_size, out_width * block_size]
    """
    in_n, in_c, in_h, in_w = data.shape
    new_h = int(in_h * block_size)
    new_w = int(in_h * block_size)
    new_c = int(in_c / (block_size * block_size))

    if mode == "DCR":
        expanded = np.reshape(data, newshape=[in_n, block_size, block_size, new_c, in_h, in_w])
        transposed = np.transpose(expanded, axes=[0, 3, 4, 1, 5, 2])
    else:
        expanded = np.reshape(data, newshape=(in_n, new_c, block_size, block_size, in_h, in_w))
        transposed = np.transpose(expanded, axes=(0, 1, 4, 2, 5, 3))
    newshape = [in_n, new_c, new_h, new_w]
    d2s_out = np.reshape(transposed, newshape=newshape)
    return d2s_out
