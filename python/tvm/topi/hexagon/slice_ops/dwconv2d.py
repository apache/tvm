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
# pylint: disable=line-too-long

"""Hexagon slice dwconv2d compute and schedule"""
import typing

import tvm
from tvm import te
from ..utils import get_layout_transform_fn


def dwconv2d_compute(
    activations: te.Tensor,
    weights: te.Tensor,
    out_shape: typing.Tuple,
    stride: typing.Tuple,
    dilation: typing.Tuple,
    dtype: str,
) -> te.Tensor:
    """Compute for slice dwconv2d op for hexagon.
    This op makes the following assumptions:
    1. This op is written for a sliced dw convolution with 2d physical buffers
    2. The input activations is assumed to be in NHWC layout and filter is in HWIO layout
    Parameters
    ----------
    activations : te.Tensor
        Input activations padded for inner dimension size
    weights : te.Tensor
        Weights without dilation
    out_shape : typing.Tuple
        The logical output shape without considering input padding
    stride : typing.Tuple
        stride
    dilation : typing.Tuple
        dilation
    dtype : str
        dtype
    Returns
    -------
    output : te.Tensor
        Output of applying 2D depthwise convolution of Weights on Input
    """

    filt_shape = weights.shape

    reduce_height = tvm.te.reduce_axis((0, filt_shape[0]), name="reduce_height")
    reduce_width = tvm.te.reduce_axis((0, filt_shape[1]), name="reduce_width")
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation
    output = tvm.te.compute(
        out_shape,
        lambda n, h, w, c: tvm.te.sum(
            (
                activations[
                    n,
                    h * stride_height + reduce_height * dilation_height,
                    w * stride_width + reduce_width * dilation_width,
                    c,
                ]
                * weights[reduce_height, reduce_width, 0, c]
            ).astype(dtype),
            axis=[reduce_height, reduce_width],
        ),
        name="Output",
    )
    return output


def dwconv2d_schedule(
    outs: te.Tensor,
    ins: typing.List[te.Tensor],
    transform_activation_layout: str,
    transform_weights: str,
) -> tvm.tir.Schedule:
    """STIR schedule definition for the compute defined above by dwconv2d_compute.
        - Auto-generated prim_func before applying schedule primitives for reference
        - The below TVMScript code is for dwconv2d with padded input dimensions and a stride of 1x1
    # from tvm.script import tir as T
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(InputTensor: T.Buffer[(1, 16, 8, 32), "float16"], Weights: T.Buffer[(3, 3, 1, 32), "float16"], Output: T.Buffer[(1, 8, 4, 32), "float16"]) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0, i1, i2, i3, i4, i5 in T.grid(1, 8, 4, 32, 3, 3):
                with T.block("Output"):
                    n, h, w, c, reduce_height, reduce_width = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(InputTensor[n, h + reduce_height, w + reduce_width, c], Weights[reduce_height, reduce_width, 0, c])
                    T.writes(Output[n, h, w, c])
                    with T.init():
                        Output[n, h, w, c] = T.float16(0)
                    Output[n, h, w, c] = Output[n, h, w, c] + InputTensor[n, h + reduce_height, w + reduce_width, c] * Weights[reduce_height, reduce_width, 0, c]
        Parameters
        ----------
        outs : te.Tensor
            The output Tensor as returned by a call to dwconv2d_compute
        ins : typing.List[te.Tensor]
            This is a list of 2 tensors - Input activations and Weights
        transform_activation_layout : str
            The transformation string representing the expected activations layout
        transform_weights : typing.Callable
            The transformation function definition for the expected weights layout
        Returns
        -------
        sch : tvm.tir.Schedule
            The STIR schedule for slice dwconv2d compute
    """
    assert len(ins) == 2, "This schedule expects only 2 inputs - Activations and Weights"
    source_expr = ins + [outs]
    prim_func = tvm.te.create_prim_func(source_expr)
    sch = tvm.tir.Schedule(prim_func)
    compute = sch.get_block("Output")
    transform_layout_fn = get_layout_transform_fn(transform_activation_layout)
    transform_layout_weights = get_layout_transform_fn(transform_weights)
    # Apply layout_transform for activation
    sch.transform_layout(compute, ins[0].name, transform_layout_fn)

    # Apply layout_transform for weights
    sch.transform_layout(compute, ins[1].name, transform_layout_weights)

    # Apply layout_transform for output
    sch.transform_layout(compute, outs.name, transform_layout_fn)

    batch, height, width, channel, reduce_height, reduce_width = sch.get_loops(
        compute
    )  # This still returns the original 6d loop
    h_outer, h_inner = sch.split(height, [None, 8])
    w_outer, w_inner = sch.split(width, [None, 4])
    w_inner_outer, w_inner_inner = sch.split(w_inner, [2, 2])
    c_outer, c_inner = sch.split(channel, [None, 32])
    sch.reorder(
        batch,
        h_outer,
        w_outer,
        c_outer,
        h_inner,
        w_inner_outer,
        reduce_height,
        reduce_width,
        c_inner,
        w_inner_inner,
    )
    sch.decompose_reduction(compute, reduce_height)
    # ci_wii = sch.fuse(c_inner, w_inner_inner)
    # sch.vectorize(ci_wii)
    return sch
