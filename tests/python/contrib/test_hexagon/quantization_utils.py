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

def quantize_uint8(val, minval, maxval):
    range = max(0.0001, maxval - minval)
    resize_amt = 255.0 / range
    value_f = (val - minval) * resize_amt
    value_i = round(value_f, 8)
    if value_i < 0:
        return 0
    elif value_i > 255:
        return 255
    else:
        return int(value_i)


def dequantize(val, minval, maxval):
    range = max(0.0001, maxval - minval)
    stepsize = range / 4294967296
    return val * stepsize


def quantize_array(in_f, size):
    in_q = []
    # 0 must lie in interval [min,max] for quantization to work correctly.
    in_min = min(0, min(in_f))
    in_max = max(0, max(in_f))
    for i in range(size):
        in_q.append(quantize_uint8(in_f[i], in_min, in_max))
    return in_q, in_min, in_max