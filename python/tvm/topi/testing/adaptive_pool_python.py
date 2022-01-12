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
"""adaptive pool in python"""
import numpy as np


def _start_index(index, odim, idim):
    return int(np.floor(index * idim / odim))


def _end_index(index, odim, idim):
    return int(np.ceil((index + 1) * idim / odim))


def _pool1d(in_size, out_size, np_data, np_op):
    out = np.zeros(out_size).astype(np_data.dtype)
    ow = out_size[0]
    for l in range(ow):
        l_start = _start_index(l, ow, in_size[0])
        l_end = _end_index(l, ow, in_size[0])
        l_sl = slice(l_start, l_end)
        out[l] = np_op(np_data[l_sl])
    return out


def _pool2d(in_size, out_size, np_data, np_op):
    out = np.zeros(out_size).astype(np_data.dtype)
    oh, ow = out_size
    for k in range(oh):
        k_start = _start_index(k, oh, in_size[0])
        k_end = _end_index(k, oh, in_size[0])
        k_sl = slice(k_start, k_end)
        for l in range(ow):
            l_start = _start_index(l, ow, in_size[1])
            l_end = _end_index(l, ow, in_size[1])
            l_sl = slice(l_start, l_end)
            out[k, l] = np_op(np_data[k_sl, l_sl])
    return out


def _pool3d(in_size, out_size, np_data, np_op):
    out = np.zeros(out_size).astype(np_data.dtype)
    od, oh, ow = out_size
    for m in range(od):
        m_start = _start_index(m, od, in_size[0])
        m_end = _end_index(m, od, in_size[0])
        m_sl = slice(m_start, m_end)
        for k in range(oh):
            k_start = _start_index(k, oh, in_size[1])
            k_end = _end_index(k, oh, in_size[1])
            k_sl = slice(k_start, k_end)
            for l in range(ow):
                l_start = _start_index(l, ow, in_size[2])
                l_end = _end_index(l, ow, in_size[2])
                l_sl = slice(l_start, l_end)
                out[m, k, l] = np_op(np_data[m_sl, k_sl, l_sl])
    return out


def adaptive_pool_channel_first(np_data, out_size, pool_op, np_op):
    """The reference function for adaptive pool, channel first layout"""
    ishape = np_data.shape
    n, c = ishape[:2]
    oshape = (n, c) + out_size
    np_out = np.zeros(oshape).astype(np_data.dtype)

    for i in range(n):
        for j in range(c):
            np_out[i, j] = pool_op(ishape[2:], out_size, np_data[i, j], np_op)

    return np_out


def adaptive_pool_channel_last(np_data, out_size, pool_op, np_op):
    """The reference function for adaptive pool, channel last layout"""
    ishape = np_data.shape
    n, c = ishape[0], ishape[-1]
    oshape = (n,) + out_size + (c,)
    np_out = np.zeros(oshape).astype(np_data.dtype)

    for i in range(n):
        for j in range(c):
            if len(out_size) == 1:
                np_out[i, :, j] = pool_op(ishape[1:-1], out_size, np_data[i, :, j], np_op)
            elif len(out_size) == 2:
                np_out[i, :, :, j] = pool_op(ishape[1:-1], out_size, np_data[i, :, :, j], np_op)
            else:
                np_out[i, :, :, :, j] = pool_op(
                    ishape[1:-1], out_size, np_data[i, :, :, :, j], np_op
                )

    return np_out


def adaptive_pool(np_data, out_size, pool_type, layout):
    """The reference function for adaptive pool, for 2d and 3d"""
    if isinstance(out_size, int):
        out_size = (out_size,)
    if len(out_size) == 1:
        pool_op = _pool1d
    elif len(out_size) == 2:
        pool_op = _pool2d
    else:
        assert len(out_size) == 3
        pool_op = _pool3d

    np_op = np.mean if pool_type == "avg" else np.max

    if layout in ["NCW", "NCHW", "NCDHW"]:
        return adaptive_pool_channel_first(np_data, out_size, pool_op, np_op)

    assert layout in ["NWC", "NHWC", "NDHWC"]
    return adaptive_pool_channel_last(np_data, out_size, pool_op, np_op)
