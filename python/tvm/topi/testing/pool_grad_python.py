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
# pylint: disable=invalid-name, unused-argument, unused-variable
"""Gradient of pooling in python"""
import numpy as np


def pool_grad_nchw(
    a_np, out_grad_np, pool_size, strides, padding, pool_type, ceil_mode, count_include_pad=True
):
    """pool_grad for NCHW layout in python"""
    dtype = a_np.dtype
    n, ic, ih, iw = a_np.shape
    kh, kw = pool_size
    sh, sw = strides
    pt, pl, pb, pr = padding

    pad_np = np.zeros(shape=(n, ic, ih + pt + pb, iw + pl + pr)).astype(dtype)
    no_zero = (range(n), range(ic), (range(pt, ih + pt)), (range(pl, iw + pl)))
    pad_np[np.ix_(*no_zero)] = a_np
    _, _, oh, ow = out_grad_np.shape
    pool_grad_np = np.zeros(shape=a_np.shape)
    pad_pool_grad_np = np.zeros(shape=pad_np.shape)

    if pool_type == "avg":
        for i in range(oh):
            for j in range(ow):
                if count_include_pad:
                    shape = pad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw].shape
                    # this can be different from kh*kw if input size cannot divide stride
                    pad_count = shape[2] * shape[3]
                else:
                    pad_count = np.sum(
                        pad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] > 0, axis=(2, 3)
                    )
                    # take the first element, as they are the same across batch and channel
                    pad_count = pad_count.ravel()[0]
                pad_pool_grad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += out_grad_np[
                    :, :, i, j
                ].reshape(n, ic, 1, 1) / np.maximum(pad_count, 1)
    elif pool_type == "max":
        for i in range(oh):
            for j in range(ow):
                a_patch = pad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
                a_patch = np.reshape(a_patch, (n, ic, -1))
                max_indices = np.argmax(a_patch, axis=2)
                c_idx, n_idx = np.meshgrid(range(ic), range(n), sparse=True)
                h_idx, w_idx = np.unravel_index(max_indices, (kh, kw))
                pad_pool_grad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw][
                    n_idx, c_idx, h_idx, w_idx
                ] += out_grad_np[n_idx, c_idx, i, j]
    for i in range(pool_grad_np.shape[2]):
        for j in range(pool_grad_np.shape[3]):
            pool_grad_np[:, :, i, j] = pad_pool_grad_np[:, :, i + pt, j + pl]

    return pool_grad_np
