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
"""Convolution in python"""
import numpy as np


def conv2d_backward_weight_nchw_python(dy_np, x_np, stride, kernel_size, padding):
    N, C, H, W = x_np.shape[1]
    _, K, P, Q = dy_np.shape[1]
    pad_h, pad_w = padding
    dw = np.zeros((K, C, kernel_size[0], kernel_size[1])).astype(dy_np.dtype)

    for k in range(K):
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    acc = 0
                    for n in range(N):
                        for p in range(P):
                            for q in range(Q):
                                coord = (n, c, p - pad_h + r, q - pad_w + s)

                                if (
                                    coord[2] < H
                                    and coord[2] >= 0
                                    and coord[3] < W
                                    and coord[3] >= 0
                                ):
                                    acc += x[coord]

                    dw[k, c, r, s] = acc

    return dw
