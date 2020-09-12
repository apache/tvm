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
"""Find optimal scale for quantization by minimizing KL-divergence"""

import ctypes
import numpy as np

from . import _quantize


def _find_scale_by_kl(arr, quantized_dtype="int8", num_bins=8001, num_quantized_bins=255):
    """Given a tensor, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)
    min_val = np.min(arr)
    max_val = np.max(arr)
    thres = max(abs(min_val), abs(max_val))

    if min_val >= 0 and quantized_dtype in ["uint8"]:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1

    def get_pointer(arr, ctypes_type):
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes_type))
        return ctypes.cast(ptr, ctypes.c_void_p)

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-thres, thres))
    hist_ptr = get_pointer(hist.astype(np.int32), ctypes.c_int)
    hist_edges_ptr = get_pointer(hist_edges, ctypes.c_float)

    return _quantize.FindScaleByKLMinimization(
        hist_ptr, hist_edges_ptr, num_bins, num_quantized_bins
    )
