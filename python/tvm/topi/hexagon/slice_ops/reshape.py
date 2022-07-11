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

"""Hexagon slice reshape compute and schedule"""
from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def reshape_compute(inp: te.Tensor, new_shape: tuple) -> te.Tensor:
    """Compute for slice reshape op for hexagon.
    This op makes the following assumptions:
    1. This op is written for a sliced reshape operation.
    2. The input is assumed to be in NHWC layout.

    Parameters
    ----------
    Input : te.Tensor
        Input tensor
    New Shape: tuple
        Output shape
    Returns
    -------
    Output : te.Tensor
        Output of applying reshape operation on input
    """
    return topi.transform.reshape(inp, new_shape)


def stir_sched_nhwc_2d_op(
    out: te.Tensor,
    inp: te.Tensor,
    out_layout: str,
    in_layout: str,
    c_split: int,
) -> tir.Schedule:
    """Schedule for output layout: nc-1024-2d, nc-2048-2d"""
    reshape_func = te.create_prim_func([inp, out])
    sch = tir.Schedule(reshape_func, debug_mask="all")
    compute = sch.get_block("T_reshape")

    sch.transform_layout(compute, inp.name, get_layout_transform_fn(in_layout))
    sch.transform_layout(compute, out.name, get_layout_transform_fn(out_layout))
    i, j = sch.get_loops(compute)
    jout, channel = sch.split(j, [None, inp.shape[3]])
    height, width = sch.split(jout, [inp.shape[1], inp.shape[2]])
    channelo, channeli = sch.split(channel, [None, 1024])
    channelio, channelii = sch.split(channeli, [None, c_split])
    sch.reorder(i, height, width, channelo, channelio, channelii)
    sch.vectorize(channelii)
    return sch


def stir_schedule_nhwc_8h2w32c2w(
    out: te.Tensor,
    inp: te.Tensor,
    out_layout: str,
    in_layout: str,
) -> tir.Schedule:
    """Schedule for input and output layout nhwc-8h2w32c2w"""
    reshape_func = te.create_prim_func([inp, out])
    sch = tir.Schedule(reshape_func, debug_mask="all")
    compute = sch.get_block("T_reshape")

    sch.transform_layout(compute, inp.name, get_layout_transform_fn(in_layout))
    sch.transform_layout(compute, out.name, get_layout_transform_fn(out_layout))
    return sch


def reshape_stir_schedule(
    out: te.Tensor,
    inp: te.Tensor,
    output_layout: str,
    input_layout: str,
) -> tir.Schedule:
    """STIR schedule definition for the compute of reshape compute.
    Parameters
    ----------
    outputs : te.Tensor
        The output tensor as returned by a call to reshape_compute
    input : te.Tensor
        Input tensor to reshape
    out_layout: str
        The transformation function definition for the expected output layout
    in_layout: str
        The transformation function definition for the input layout
    Returns
    -------
    sch : tvm.tir.Schedule
        The STIR schedule for slice reshape compute
    """
    if output_layout in ["nhwc-8h2w32c2w-2d", "nhwc-8h8w32c-2d"]:
        return stir_schedule_nhwc_8h2w32c2w(out, inp, output_layout, input_layout)
    if output_layout == "nc-1024-2d":
        return stir_sched_nhwc_2d_op(out, inp, output_layout, input_layout, 64)
    if output_layout == "nc-2048-2d":
        return stir_sched_nhwc_2d_op(out, inp, output_layout, input_layout, 128)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
