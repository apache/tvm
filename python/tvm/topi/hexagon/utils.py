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

"""Common hexagon specific utilities"""
from tvm import te


def n11c_1024c_2d(batch, height, width, channel):
    """Return index map for n11c_1024 2d layout"""
    return [batch, height, width, channel // 1024, te.AXIS_SEPARATOR, channel % 1024]


def n11c_1024c_1d(batch, height, width, channel):
    """Return index map for n11c_1024 1d layout"""
    return [batch, height, width, channel // 1024, channel % 1024]


def nhwc_8h2w32c2w_2d(batch, height, width, channel):
    """Return index map for nhwc_8h2w32c2w 2d layout"""
    return [
        batch,
        height // 8,
        width // 4,
        channel // 32,
        te.AXIS_SEPARATOR,
        height % 8,
        (width % 4) // 2,
        channel % 32,
        width % 2,
    ]


def nhwc_8h2w32c2w_1d(batch, height, width, channel):
    """Return index map for nhwc_8h2w32c2w 1d layout"""
    return [
        batch,
        height // 8,
        width // 4,
        channel // 32,
        height % 8,
        (width % 4) // 2,
        channel % 32,
        width % 2,
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
    raise RuntimeError(f"Unexpected layout '{layout}'")
