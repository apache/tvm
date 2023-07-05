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
"""Hexagon slice softmax compute and schedule"""

import typing

from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def softmax_compute(in_tensor):
    """
    Compute for slice softmax op for hexagon.
    This op makes the following assumptions:
    1. This op is written for a sliced softmax operation.
    2. The input is assumed to be in NC layout.
    """
    return topi.nn.softmax(in_tensor, axis=1)


def softmax_stir_schedule(
    out: te.Tensor, inp: te.Tensor, out_layout: typing.Callable, in_layout: typing.Callable
):
    """
    STIR schedule definition for the compute of softmax
    """

    in_layout = get_layout_transform_fn(in_layout)
    out_layout = get_layout_transform_fn(out_layout)

    func = te.create_prim_func([inp, out])
    sch = tir.Schedule(func, debug_mask="all")

    max_tensor = sch.get_block("T_softmax_maxelem")
    exp_tensor = sch.get_block("T_softmax_exp")
    sum_tensor = sch.get_block("T_softmax_expsum")
    out_tensor = sch.get_block("T_softmax_norm")

    sch.transform_layout(max_tensor, inp.name, in_layout)
    sch.transform_layout(out_tensor, out.name, out_layout)

    _, c_inner = sch.get_loops(max_tensor)
    _, c_inner_i = sch.split(c_inner, [None, 64])
    rf_max = sch.rfactor(c_inner_i, 0)
    _, _, max_inner = sch.get_loops(rf_max)
    sch.vectorize(max_inner)

    _, loopi = sch.get_loops(exp_tensor)
    _, loopi_i = sch.split(loopi, [None, 512])
    sch.vectorize(loopi_i)

    _, c_sum_inner = sch.get_loops(sum_tensor)
    _, c_sum_inner_i = sch.split(c_sum_inner, [None, 64])
    rf_sum = sch.rfactor(c_sum_inner_i, 0)
    _, _, sum_inner = sch.get_loops(rf_sum)
    sch.vectorize(sum_inner)

    _, c_out_inner = sch.get_loops(out_tensor)
    _, c_out_inner_i = sch.split(c_out_inner, [None, 512])
    sch.vectorize(c_out_inner_i)

    return sch
