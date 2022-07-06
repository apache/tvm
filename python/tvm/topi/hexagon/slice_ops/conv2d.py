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

"""Hexagon slice conv2d compute and schedule"""
import typing

import tvm
from tvm import te

from ..utils import get_layout_transform_fn


def conv2d_compute(
    activations: te.Tensor,
    weights: te.Tensor,
    out_shape: typing.Tuple,
    stride: typing.Tuple,
    dilation: typing.Tuple,
    dtype: str,
    output_name: str,
    weights_width_reversed: bool = True,
) -> te.Tensor:
    """Compute for slice conv2d op for hexagon.

    This op makes the following assumptions:
    1. This op is written for a sliced convolution with 2d physical buffers
    2. The input activations is assumed to be in NHWC layout and filter is in HWIO layout
    3. Grouped convolutions are not supported. and there will be a separate compute definition for depthwise convolution
    4. In order to get grouped convolutions, it is assumed that the op will be sliced according to the groups and multiple calls to this compute would be placed.


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
    output_name : str
        The name to be given to output. This would become the block name for the corresponding STIR compute
    weights_width_reversed : bool
        The width axis of weights are expected in reverse order if weights_width_reversed is True

    Returns
    -------
    output : te.Tensor
        Output of applying 2D convolution of Weights on Input
    """

    filt_shape = weights.shape

    reduce_channel = tvm.te.reduce_axis((0, filt_shape[2]), name="reduce_channel")
    reduce_height = tvm.te.reduce_axis((0, filt_shape[0]), name="reduce_height")
    reduce_width = tvm.te.reduce_axis((0, filt_shape[1]), name="reduce_width")
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation

    if weights_width_reversed:
        weights_width_var = filt_shape[1] - reduce_width - 1
    else:
        weights_width_var = reduce_width

    output = tvm.te.compute(
        out_shape,
        lambda n, h, w, c: tvm.te.sum(
            (
                activations[
                    n,
                    h * stride_height + reduce_height * dilation_height,
                    w * stride_width + reduce_width * dilation_width,
                    reduce_channel,
                ]
                * weights[reduce_height, weights_width_var, reduce_channel, c]
            ).astype(dtype),
            axis=[reduce_channel, reduce_height, reduce_width],
        ),
        name=output_name,
    )
    return output


def conv2d_te_schedule(
    out: te.Tensor,
    ins: typing.List[te.Tensor],
    transform_activation_layout: str,
    transform_weights_layout: str,
    transform_output_layout: str,
) -> te.Schedule:
    """TE Schedule for the sliced conv2d op

    This schedule makes the following assumptions:
    1. There is only one output tensor
    2. The activations and weights have specific layouts defined by the last 2 arguments
    3. All transformation functions are expected to be a bijection for now

    Parameters
    ----------
    out : te.Tensor
        The output tensor returned by a call to conv2d_compute
    ins : typing.List[te.Tensor]
        The list of 2 Tensors which would be the input activations and weights
    transform_activation_layout : str
        The expected activations layout
    transform_weights_layout : str
        String representing the weights layout as defined in get_layout_transform_fn
    transform_output_layout: str
        String representing the output layout as defined in get_layout_transform_fn

    Returns
    -------
    sch : te.Schedule
        The TE schedule for slice conv2d
    """
    activations, weights = ins
    output = out
    sch = tvm.te.create_schedule(output.op)
    reduce_channel, reduce_height, reduce_width = sch[output].op.reduce_axis
    sch[activations].transform_layout(get_layout_transform_fn(transform_activation_layout))
    sch[weights].transform_layout(get_layout_transform_fn(transform_weights_layout))
    transformed_axis = sch[output].transform_layout(
        get_layout_transform_fn(transform_output_layout)
    )
    fused_out_axis = sch[output].fuse(transformed_axis[-1], transformed_axis[-2])
    sch[output].reorder(
        *[*transformed_axis[:-2], reduce_height, reduce_width, reduce_channel, fused_out_axis]
    )
    # The below code doesn't work yet as vectorization across 2D boundary is not yet supported
    # s[output].vectorize(fused_out_axis)
    return sch


def conv2d_schedule(
    outs: te.Tensor,
    ins: typing.List[te.Tensor],
    transform_activation_layout: str,
    transform_weights_layout: str,
    transform_output_layout: str,
    output_name: str,
) -> tvm.tir.Schedule:
    """STIR schedule definition for the compute defined above by conv2d_compute.

    - Auto-generated prim_func before applying schedule primitives for reference
    - The below TVMScript code is for conv2d with padded input dimensions and a stride of 1x1

    # from tvm.script import tir as T
    @T.prim_func
    def func(InputTensor: T.Buffer[(1, 24, 12, 32), "float16"], Weights: T.Buffer[(3, 3, 32, 32), "float16"], compute: T.Buffer[(1, 16, 8, 32), "float16"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 16, 8, 32, 32, 3, 3):
            with T.block("compute"):
                n, h, w, c, rc, rh, rw = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(InputTensor[n, h + rh, w + rw, rc], Weights[rh, rw, rc, c])
                T.writes(compute[n, h, w, c])
                with T.init():
                    compute[n, h, w, c] = T.float16(0)
                compute[n, h, w, c] = compute[n, h, w, c] + InputTensor[n, h + rh, w + rw, rc] * Weights[rh, rw, rc, c]

    Parameters
    ----------
    outs : te.Tensor
        The output Tensor as returned by a call to conv2d_compute
    ins : typing.List[te.Tensor]
        This is a list of 2 tensors - Input activations and Weights
    transform_activation_layout : str
        String representing the activations layout as defined in get_layout_transform_fn
    transform_weights_layout : str
        String representing the weights layout as defined in get_layout_transform_fn
    transform_output_layout: str
        String representing the output layout as defined in get_layout_transform_fn
    output_name : str
        The name that was given to the output compute and which can be used to get the block name

    Returns
    -------
    sch : tvm.tir.Schedule
        The STIR schedule for slice conv2d compute
    """

    assert len(ins) == 2, "This schedule expects only 2 inputs - Activations and Weights"
    source_expr = ins + [outs]
    prim_func = tvm.te.create_prim_func(source_expr)
    sch = tvm.tir.Schedule(prim_func)

    compute = sch.get_block(output_name)
    # Apply layout_transform for activation
    sch.transform_layout(compute, ins[0].name, get_layout_transform_fn(transform_activation_layout))

    # Apply layout_transform for weights
    sch.transform_layout(compute, ins[1].name, get_layout_transform_fn(transform_weights_layout))

    # Apply layout_transform for output
    sch.transform_layout(compute, outs.name, get_layout_transform_fn(transform_output_layout))

    batch, height, width, channel, reduce_channel, reduce_height, reduce_width = sch.get_loops(
        compute
    )  # This still returns the original 7d loop
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
        reduce_channel,
        c_inner,
        w_inner_inner,
    )
    sch.decompose_reduction(compute, reduce_height)
    # ci_wii = s.fuse(ci, wii)
    # s.vectorize(ci_wii)
    return sch
