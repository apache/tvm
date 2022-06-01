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

from tvm import te


def n11c_1024c_2d(n, h, w, c):
    return [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]


def n11c_1024c_1d(n, h, w, c):
    return [n, h, w, c // 1024, c % 1024]


def nhwc_8h2w32c2w_2d(n, h, w, c):
    return [n, h // 8, w // 4, c // 32, te.AXIS_SEPARATOR, h % 8, (w % 4) // 2, c % 32, w % 2]


def nhwc_8h2w32c2w_1d(n, h, w, c):
    return [n, h // 8, w // 4, c // 32, h % 8, (w % 4) // 2, c % 32, w % 2]


def get_layout_transform_fn(layout):
    if layout == "nhwc-8h2w32c2w-2d":
        return nhwc_8h2w32c2w_2d
    if layout == "nhwc-8h2w32c2w-1d":
        return nhwc_8h2w32c2w_1d
    elif layout == "n11c-1024c-2d":
        return n11c_1024c_2d
    elif layout == "n11c-1024c-1d":
        return n11c_1024c_1d
    else:
        raise RuntimeError(f"Unexpected layout '{layout}'")


def apply_transform(s, block, block_index: int, buffer_type: str, layout: str):
    """Apply transform layout on a buffer

    Parameters
    ----------
    s: Schedule
    block : BlockRV
        The block that accesses the target buffer
    buffer_index: int
        The index of the buffer in block's read or write region
    buffer_type : str
        Type of the buffer index, "read" or "write"
    layout : str
        Layout of the buffer
    """
    transform_fn = get_layout_transform_fn(layout)
    if layout == "nhwc-8h2w32c2w-1d":
        axis_separators = [4]
    elif layout == "n11c-1024c-1d":
        axis_separators = [2]
    else:
        raise RuntimeError(f"Unexpected layout '{layout}'")

    s.transform_layout(block, block_index, buffer_type, transform_fn)
    if axis_separators:
        s.set_axis_separator(block, block_index, buffer_type, axis_separators)