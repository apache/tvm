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
"""max_pool3d and avg_pool3d in python"""
import math
import numpy as np
import tvm

def pool3d_ncdhw_python(np_data, kernel,
                        strides, padding,
                        out_shape, pool_type,
                        count_include_pad=True,
                        ceil_mode=False, dtype="float32"):
    """baseline for max_pool3d and avg_pool3d, default layout is "NCDHW"""
    in_n, in_c, in_d, in_h, in_w = in_shape = np_data.shape
    if isinstance(kernel, int):
        k_d = k_h = k_w = kernel
    else:
        k_d, k_h, k_w = kernel
    if isinstance(strides, int):
        s_d = s_h = s_w = strides
    else:
        s_d, s_h, s_w = strides
    if isinstance(padding, int):
        pf = pt = pl = pk = pb = pr = padding
    else:
        pf, pt, pl, pk, pb, pr = padding

    if ceil_mode:
        assert out_shape[2] == int(math.ceil(float(in_shape[2] - k_d + pf + pk) / s_d) + 1)
        assert out_shape[3] == int(math.ceil(float(in_shape[3] - k_h + pt + pb) / s_h) + 1)
        assert out_shape[4] == int(math.ceil(float(in_shape[4] - k_w + pl + pr) / s_w) + 1)
    else:
        assert out_shape[2] == int(math.floor(float(in_shape[2] - k_d + pf + pk) / s_d) + 1)
        assert out_shape[3] == int(math.floor(float(in_shape[3] - k_h + pt + pb) / s_h) + 1)
        assert out_shape[4] == int(math.floor(float(in_shape[4] - k_w + pl + pr) / s_w) + 1)

    fill_value = tvm.tir.const(0.0, dtype).value
    if not(count_include_pad) and pool_type == 'max':
        fill_value = tvm.te.min_value(dtype).value

    pad_np = np.full(shape=(in_n, in_c,
                            in_d + pf + pk,
                            in_h + pt + pb,
                            in_w + pl + pr),
                     fill_value=fill_value,
                     dtype=dtype)

    no_zero = (range(in_n),
               range(in_c),
               (range(pf, in_d + pf)),
               (range(pt, in_h + pt)),
               (range(pl, in_w + pl)))
    pad_np[np.ix_(*no_zero)] = np_data
    ret_np = np.zeros(shape=out_shape).astype(dtype)

    if pool_type == 'avg':
        for k in range(out_shape[2]):
            for i in range(out_shape[3]):
                for j in range(out_shape[4]):
                    if count_include_pad:
                        ret_np[:, :, k, i, j] = \
                            np.mean(pad_np[:, :, k * s_d: k * s_d + k_d,
                                           i * s_h: i * s_h + k_h,
                                           j * s_w: j * s_w + k_w], axis=(2, 3, 4))
                    else:
                        pad_count = np.sum(pad_np[:, :,
                                                  k * s_d: k * s_d + k_d,
                                                  i * s_h: i * s_h + k_h,
                                                  j * s_w: j * s_w + k_w] > 0, axis=(2, 3, 4))
                        ret_np[:, :, k, i, j] = np.sum(pad_np[:, :,
                                                              k * s_d: k * s_d + k_d,
                                                              i * s_h: i * s_h + k_h,
                                                              j * s_w: j * s_w + k_w],
                                                       axis=(2, 3, 4)) / np.maximum(pad_count, 1)
    elif pool_type == 'max':
        for k in range(out_shape[2]):
            for i in range(out_shape[3]):
                for j in range(out_shape[4]):
                    ret_np[:, :, k, i, j] = np.max(
                        pad_np[:, :, k * s_d: k * s_d + k_d,
                               i * s_h: i * s_h + k_h,
                               j * s_w: j * s_w + k_w], axis=(2, 3, 4))
    else:
        raise ValueError("pool type {} is not supported".format(pool_type))

    ret_np = np.maximum(ret_np, fill_value)
    return ret_np
