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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Utility scheduling functions for the Bifrost schedules"""

from __future__ import absolute_import as _abs
import tvm
from tvm import te


def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """Fuse all the axis and bind to GPU threads"""
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    max_threads = tvm.target.Target.current(allow_none=False).max_num_threads
    bx, tx = s[tensor].split(fused, num_thread or max_threads)
    s[tensor].bind(bx, te.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, te.thread_axis("threadIdx.x"))
    return bx, tx


def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """Tile and bind to GPU threads"""
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """Tile and bind 3d"""
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, te.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    return zo, yo, xo, zi, yi, xi


def pack_tensor(s, tensor, factor, readers):
    """Do transform X[n, m] -> X[n / factor, m, factor]"""
    tmp = s.cache_read(tensor, "global", readers)
    y, x = s[tmp].op.axis
    yo, yi = s[tmp].split(y, factor)
    s[tmp].reorder(yo, x, yi)
    s[tmp].compute_inline()
    return s.cache_write(tmp, "global"), tmp


def transpose(s, tensor, y_index, x_index, readers):
    """Do transform X[n, m] -> X[m, n]"""
    tmp = s.cache_read(tensor, "global", readers)
    y, x = s[tmp].op.axis[y_index], s[tmp].op.axis[x_index]
    s[tmp].reorder(x, y)
    s[tmp].compute_inline()
    A_transpose = s.cache_write(tmp, "global")

    CR_A = s.cache_read(tensor, "local", [A_transpose])
    CW_A_transpose = s.cache_write(A_transpose, "local")

    y, x = s[A_transpose].op.axis[y_index], s[A_transpose].op.axis[x_index]
    yo, xo, yi, xi = s[A_transpose].tile(y, x, 4, 4)
    s[A_transpose].unroll(yi)
    s[A_transpose].vectorize(xi)
    _, _, _, xi = tile_and_bind(s, A_transpose, yo, xo, 32, 2)

    s[CW_A_transpose].compute_at(s[A_transpose], xi)
    y, x = s[CW_A_transpose].op.axis[y_index], s[CW_A_transpose].op.axis[x_index]
    s[CW_A_transpose].unroll(x)
    s[CW_A_transpose].unroll(y)

    s[CR_A].compute_at(s[A_transpose], xi)
    y, x = s[CR_A].op.axis[y_index], s[CR_A].op.axis[x_index]
    s[CR_A].unroll(y)
    s[CR_A].vectorize(x)

    return tmp


def interleave_transpose(s, tensor, width, y_index, x_index, readers, batched=False):
    """Interleave the tensor, then transpose it"""
    tmp = s.cache_read(tensor, "global", readers)
    y, x = s[tmp].op.axis[y_index], s[tmp].op.axis[x_index]
    xo, xi = s[tmp].split(x, width)
    s[tmp].reorder(xo, y, xi)
    s[tmp].fuse(y, xi)
    if batched:
        z = s[tmp].op.axis[0]
        s[tmp].fuse(z, xo)
    s[tmp].compute_inline()
    return s.cache_write(tmp, "global"), tmp


def transpose_interleave(s, tensor, width, y_index, x_index, readers, batched=False):
    """Transpose the tensor, then interleave it"""
    tmp = s.cache_read(tensor, "global", readers)
    y, x = s[tmp].op.axis[y_index], s[tmp].op.axis[x_index]
    yo, yi = s[tmp].split(y, width)
    s[tmp].reorder(yo, x, yi)
    s[tmp].fuse(x, yi)
    if batched:
        z = s[tmp].op.axis[0]
        s[tmp].fuse(z, yo)
    s[tmp].compute_inline()
    return s.cache_write(tmp, "global"), tmp
