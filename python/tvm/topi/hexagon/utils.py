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


"""Common hexagon specific utilities"""
from tvm import te


def n11c_1024c_2d(n, h, w, c):
    """Return index map for n11c_1024 2d layout"""
    return [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def n11c_1024c_1d(n, h, w, c):
    """Return index map for n11c_1024 1d layout"""
    return [n, h, w, c // 1024, c % 1024]


def nhwc_8h2w32c2w_2d(n, h, w, c):
    """Return index map for nhwc_8h2w32c2w 2d layout"""
    return [n, h // 8, w // 4, c // 32, te.AXIS_SEPARATOR, h % 8, (w % 4) // 2, c % 32, w % 2]


def nhwc_8h2w32c2w_1d(n, h, w, c):
    """Return index map for nhwc_8h2w32c2w 1d layout"""
    return [n, h // 8, w // 4, c // 32, h % 8, (w % 4) // 2, c % 32, w % 2]


def nhw_32h16w_2d(n, h, w):
    """Return index map for nhw_32h16w 2d layout"""
    return [n, h // 32, w // 16, te.AXIS_SEPARATOR, h % 32, w % 16]


def nhwc_4h4w32c_1d(n, h, w, c):
    """Return index map for nhwc_4h4232c 1d layout"""
    return [n, h // 4, w // 4, c // 32, h % 4, w % 4, c % 32]


def nhwc_4h4w32c_2d(n, h, w, c):
    """Return index map for nhwc_4h4w32c 2d layout"""
    return [n, h // 4, w // 4, c // 32, te.AXIS_SEPARATOR, h % 4, w % 4, c % 32]


def nc_512c_1d(n, c):
    """Return index map for nc_512c 1d layout"""
    return [n, c // 512, c % 512]


def nc_512c_2d(n, c):
    """Return index map for nc_512c 2d layout"""
    return [n, c // 512, te.AXIS_SEPARATOR, c % 512]


def nc_1024c_2d(n, c):
    """Return index map for nc_1024c 2d layout"""
    return [n, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def nhwc_4h2w32c2w_2d(n, h, w, c):
    """Return index map for nhwc_4h2w32c2w 2d layout"""
    return [n, h // 4, w // 4, c // 32, te.AXIS_SEPARATOR, h % 4, (w % 4) // 2, c % 32, w % 2]


def nhwc_1024c_2d(n, h, w, c):
    """Return index map for nhwc_1024 2d layout"""
    return [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def nc_1024_2d(n, c):
    """Return index map for nc_1024 2d layout"""
    return [n, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def iohw_16i32o2i_1d(height, width, in_channel, out_channel):
    return [
        in_channel // 32,
        out_channel // 32,
        height,
        width,
        (in_channel % 32) // 2,
        out_channel % 32,
        in_channel % 2,
    ]


def get_layout_transform_fn(layout):
    """Return index map function as per the layout string"""
    if layout == "nhwc-8h2w32c2w-2d":
        return nhwc_8h2w32c2w_2d
    if layout == "nhwc-8h2w32c2w-1d":
        return nhwc_8h2w32c2w_1d
    if layout == "n11c-1024c-2d":
        return n11c_1024c_2d
    if layout == "n11c-1024c-1d":
        return n11c_1024c_1d
    if layout == "nhwc-1024c-2d":
        return nhwc_1024c_2d
    if layout == "nc-1024-2d":
        return nc_1024_2d
    if layout == "nhw-32h16w-2d":
        return nhw_32h16w_2d
    if layout == "nhwc-4h4w32c-2d":
        return nhwc_4h4w32c_2d
    if layout == "nhwc-4h4w32c-1d":
        return nhwc_4h4w32c_1d
    if layout == "nc-512c-2d":
        return nc_512c_2d
    if layout == "nc-512c-1d":
        return nc_512c_1d
    if layout == "nhwc-4h2w32c2w-2d":
        return nhwc_4h2w32c2w_2d
    if layout == "nc-1024c-2d":
        return nc_1024c_2d
    if layout == "iohw-16i32o2i-1d":
        return iohw_16i32o2i_1d
    raise RuntimeError(f"Unexpected layout '{layout}'")
