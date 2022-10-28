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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals
"""
Please note the following assumptions made by the implementation:
1) The input must be padded in advance to account for 'padding'. In addition,
   both input and output must be padded as per the physical buffer layout.
2) 'padding' is ignored. It must be handled outside of the sliced op.
3) The weights are expected to be as per physical layout

The initial compute for quantized depthwise conv2d is as follows
where cm = channel_multiplier; assumed to be 1,
zp_a = Activation_zero_point,
zp_w = Weight_zero_point,
Qa = Quantized Activation,
Qw = Quantized Weights.

     a) Qc(n, oh, ow, oc) = (Sigma(r, s) (Qw(r, s, oc%cm, oc/cm) - zp_w)
                                      * (Qa(n, oh + r, ow + s, oc/cm) - zp_a))
                                      * scale_value
        where scale_value = (activation_scale * weight_scale) / output_scale

        This can be written as

     b) Qc(n, oh, ow, oc) = (t1 - t2 - t3 + t4) * scale_value

        where t1 = Sigma(r, s) Qw(r, s, oc%cm, oc/cm) * Qa(n, oh + r, ow + s, oc/cm)
              t2 = Sigma(r, s) zp_w * Qa(n, oh + r, ow + s, oc/cm)
              t3 = Sigma(r, s) zp_a * Qw(r, s, oc%cm, oc/cm)
              t4 = Sigma(r, s) zp_a * zp_w

     c) Qc(n, oh, ow, oc) = saturate(((t1 - t2 - t3 + t4) * fixed_scale_value)) >> rsh)

        where fixed_scale_value, rsh are fixed point values for scale_value.


Compute and schedule for quantized depthwise conv2d slice op"""

import typing
import tvm
from tvm import te
from ..utils import get_layout_transform_fn, get_fixed_point_value, saturate


def qdepthwise_conv2d_compute(
    activations: te.Tensor,
    weights: te.Tensor,
    out_shape: typing.Tuple,
    stride: typing.Tuple,
    dilation: typing.Tuple,
    dtype: str,
    # quantization params:
    activation_zero_point,
    activation_scale,
    weight_zero_point,
    weight_scale,
    output_zero_point,
    output_scale,
):
    """Compute for quantized depthwise conv2d"""
    filt_shape = weights.shape
    ob, oh, ow, oc = out_shape

    if dtype == "uint8":
        temp_dtype = "int32"
        big_dtype = "int64"
    elif dtype == "int8":
        temp_dtype = "int32"
        big_dtype = "int64"
    else:
        raise RuntimeError(f"Unsupported output dtype, {odtype}'")

    reduce_height = tvm.te.reduce_axis((0, filt_shape[0]), name="reduce_height")
    reduce_width = tvm.te.reduce_axis((0, filt_shape[1]), name="reduce_width")
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation

    scale_value = (activation_scale * weight_scale) / output_scale
    fixed_scale_value, rsh = get_fixed_point_value(scale_value, "int16")

    t1 = tvm.te.compute(
        out_shape,
        lambda n, h, w, c: tvm.te.sum(
            (
                (
                    activations[
                        n,
                        h * stride_height + reduce_height * dilation_height,
                        w * stride_width + reduce_width * dilation_width,
                        c,
                    ].astype(temp_dtype)
                )
                * (weights[reduce_height, reduce_width, 0, c].astype(temp_dtype))
            ).astype(temp_dtype),
            axis=[reduce_height, reduce_width],
        ),
        name="t1",
    )

    t2 = tvm.te.compute(
        out_shape,
        lambda n, h, w, c: tvm.te.sum(
            (
                (
                    activations[
                        n,
                        h * stride_height + reduce_height * dilation_height,
                        w * stride_width + reduce_width * dilation_width,
                        c,
                    ].astype(temp_dtype)
                )
                * weight_zero_point
            ).astype(temp_dtype),
            axis=[reduce_height, reduce_width],
        ),
        name="t2",
    )

    t3 = tvm.te.compute(
        (oc,),
        lambda c: tvm.te.sum(
            (
                ((weights[reduce_height, reduce_width, 0, c].astype(temp_dtype)))
                * activation_zero_point
            ).astype(temp_dtype),
            axis=[reduce_height, reduce_width],
        ),
        name="t3",
    )

    t4 = activation_zero_point * weight_zero_point * reduce_height * reduce_width

    output = tvm.te.compute(
        out_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    (
                        ((t1[n, h, w, c]).astype(big_dtype) - t2[n, h, w, c] - t3[c] + t4)
                        * fixed_scale_value
                    )
                    >> rsh
                )
                + (output_zero_point).astype(big_dtype)
            ),
            dtype,
        ).astype(dtype),
        name="output",
    )

    return output


def qdepthwise_conv2d_schedule(
    outs: te.Tensor,
    ins: typing.List[te.Tensor],
    transform_activation_layout: str,
    transform_weights: str,
):
    """
    Schedule for quantized depthwise conv2d for input layout nhwc-8h8w32c
    assert len(ins) == 2, "This schedule expects only 2 inputs - Activations and Weights
    """
    source_expr = ins + [outs]
    prim_func = tvm.te.create_prim_func(source_expr)
    sch = tvm.tir.Schedule(prim_func)

    compute = sch.get_block("output")
    compute1 = sch.get_block("t1")

    transform_layout_fn = get_layout_transform_fn(transform_activation_layout)
    transform_layout_weights = get_layout_transform_fn(transform_weights)

    # Apply layout_transform for activation
    sch.transform_layout(compute1, ins[0].name, transform_layout_fn)

    # Apply layout_transform for weights
    sch.transform_layout(compute1, ins[1].name, transform_layout_weights)

    # Apply layout_transform for output
    sch.transform_layout(compute, outs.name, transform_layout_fn)

    # This returns the original 6d loop
    batch, height, width, channel, reduce_height, reduce_width = sch.get_loops(compute1)
    h_outer, h_inner = sch.split(height, [None, 8])
    w_outer, w_inner = sch.split(width, [None, 8])
    c_outer, c_inner = sch.split(channel, [None, 32])
    sch.reorder(
        batch,
        h_outer,
        w_outer,
        c_outer,
        h_inner,
        reduce_height,
        reduce_width,
        w_inner,
        c_inner,
    )

    sch.decompose_reduction(compute1, reduce_height)
    # wi_ci = sch.fuse(w_inner,c_inner)
    # sch.vectorize(wi_ci)
    return sch
