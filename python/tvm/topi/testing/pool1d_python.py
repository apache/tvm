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
"""max_pool1d and avg_pool1d in python"""
import math
import numpy as np


def pool1d_ncw_python(
    np_data,
    kernel,
    strides,
    padding,
    out_shape,
    pool_type,
    count_include_pad=True,
    ceil_mode=False,
    dtype="float32",
):
    """Baseline for max_pool1d and avg_pool1d, default layout is NCW"""
    in_n, in_c, in_w = in_shape = np_data.shape
    k_w = kernel[0]
    s_w = strides[0]
    pl, pr = padding

    if ceil_mode:
        assert out_shape[2] == int(math.ceil(float(in_shape[2] - k_w + pl + pr) / s_w) + 1)
    else:
        assert out_shape[2] == int(math.floor(float(in_shape[2] - k_w + pl + pr) / s_w) + 1)

    pad_np = np.zeros(shape=(in_n, in_c, in_w + pl + pr)).astype(dtype)

    no_zero = (range(in_n), range(in_c), range(pl, in_w + pl))
    pad_np[np.ix_(*no_zero)] = np_data
    ret_np = np.zeros(shape=out_shape).astype(dtype)

    if pool_type == "avg":
        for k in range(out_shape[2]):
            if count_include_pad:
                ret_np[:, :, k] = np.mean(pad_np[:, :, k * s_w : k * s_w + k_w], axis=(2,))
            else:
                pad_count = np.sum(pad_np[:, :, k * s_w : k * s_w + k_w] > 0, axis=(2,))
                ret_np[:, :, k] = np.sum(
                    pad_np[:, :, k * s_w : k * s_w + k_w], axis=(2,)
                ) / np.maximum(pad_count, 1)

    elif pool_type == "max":
        for k in range(out_shape[2]):
            ret_np[:, :, k] = np.max(pad_np[:, :, k * s_w : k * s_w + k_w], axis=(2,))

    else:
        raise ValueError("Pool type {} is not supported".format(pool_type))

    ret_np = np.maximum(ret_np, 0.0)
    return ret_np
