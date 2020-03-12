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


def adaptive_pool(np_data, out_size, pool_type):
    def start_index(index, odim, idim):
        return int(np.floor(index * idim / odim))

    def end_index(index, odim, idim):
        return int(np.ceil((index + 1) * idim / odim))

    def pool2d(i, j):
        out = np.zeros(out_size).astype(np_data.dtype)
        n, c, h, w = np_data.shape
        oh, ow = out_size
        for k in range(oh):
            k_start = start_index(k, oh, h)
            k_end = end_index(k, oh, h)
            k_sl = slice(k_start, k_end)
            for l in range(ow):
                l_start = start_index(l, ow, w)
                l_end = end_index(l, ow, w)
                l_sl = slice(l_start, l_end)
                out[k, l] = np_op(np_data[i, j, k_sl, l_sl])
        return out

    def pool3d(i, j):
        out = np.zeros(out_size).astype(np_data.dtype)
        n, c, d, h, w = np_data.shape
        od, oh, ow = out_size
        for m in range(od):
            m_start = start_index(m, od, d)
            m_end = end_index(m, od, d)
            m_sl = slice(m_start, m_end)
            for k in range(oh):
                k_start = start_index(k, oh, h)
                k_end = end_index(k, oh, h)
                k_sl = slice(k_start, k_end)
                for l in range(ow):
                    l_start = start_index(l, ow, w)
                    l_end = end_index(l, ow, w)
                    l_sl = slice(l_start, l_end)
                    out[m, k, l] = np_op(np_data[i, j, m_sl, k_sl, l_sl])
        return out

    if len(out_size) == 2:
        pool_op = pool2d
    else:
        assert len(out_size) == 3
        pool_op = pool3d

    n, c = np_data.shape[:2]
    oshape = (n, c) + out_size
    np_out = np.zeros(oshape).astype(np_data.dtype)
    np_op = np.mean if pool_type == "avg" else np.max
    for i in range(n):
        for j in range(c):
            np_out[i, j] = pool_op(i, j)
    return np_out
