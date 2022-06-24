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

"""Hexagon slice batch flatten compute and schedule"""
from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def batch_flatten_compute(inp: te.Tensor) -> te.Tensor:
    """Compute for slice batch flatten op for hexagon.
    This op makes the following assumptions:
    1. This op is written for a sliced batch flatten operation.
    2. The input is assumed to be in NHWC layout.

    Parameters
    ----------
    Input : te.Tensor
        Input activations padded for inner dimension size
    Returns
    -------
    Output : te.Tensor
        Output of applying batch flatten operation on input
    """
    return topi.nn.flatten(inp)


def batch_flatten_stir_schedule(
    out: te.Tensor,
    inp: te.Tensor,
    out_layout: str,
    in_layout: str,
) -> tir.Schedule:
    """STIR schedule definition for the compute of batch flatten compute.
    Parameters
    ----------
    outputs : te.Tensor
        The output tensor as returned by a call to batch_flatten_compute
    input : te.Tensor
        Input tensor to batch_flatten
    out_layout: typing.Callable
        The transformation function definition for the expected output layout
    in_layout: typing.Callable
        The transformation function definition for the input layout
    Returns
    -------
    sch : tvm.tir.Schedule
        The STIR schedule for slice batch flatten compute
    """

    batch_flatten_func = te.create_prim_func([inp, out])
    sch = tir.Schedule(batch_flatten_func, debug_mask="all")
    compute = sch.get_block("compute")

    sch.transform_layout(compute, inp.name, get_layout_transform_fn(in_layout))
    sch.transform_layout(compute, out.name, get_layout_transform_fn(out_layout))
    i, j = sch.get_loops(compute)
    jout, channel = sch.split(j, [None, inp.shape[3]])
    height, width = sch.split(jout, [inp.shape[1], inp.shape[2]])
    channelo, channeli = sch.split(channel, [None, 1024])
    channelio, channelii = sch.split(channeli, [None, 64])
    sch.reorder(i, height, width, channelo, channelio, channelii)
    sch.vectorize(channelii)
    return sch
