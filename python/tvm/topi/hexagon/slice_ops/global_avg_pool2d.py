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

"""
Assumptions:
1) The input is in NCHW layout. Squeezenet is the only model that calls
   nn.global_avg_pool2d and the only layout it uses is 'NCHW'.
2) The op takes input data as an argument.
3) Both input and output dtype is float32 and
4) Input is assumed to always be multiple of fixed chunk 32c8h4w.
"""

from tvm import te
from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn


def global_avg_pool2d(
    data: te.Tensor,
):
    """global_avg_pool2d"""
    return topi.nn.global_pool(data, "avg", "NCHW")


def stir_global_avg_pool2d_schedule(outs: te.Tensor, ins: te.Tensor, input_layout: str):
    """Schedule"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)

    sum_block = s.get_block("adaptive_pool_sum")

    # Input is multiple of fixed chunk but output is NxCx1x1
    # Hence transform_layout is only applied on input
    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(sum_block, buffer=("read", 0), index_map=input_transformed_layout)

    return s
