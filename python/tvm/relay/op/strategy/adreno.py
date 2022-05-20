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
"""Definition of adreno operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_NCHWc_strategy.register("adreno")
@conv2d_strategy.register("adreno")
def conv2d_strategy_adreno(attrs, inputs, out_type, target):
    """conv2d adreno strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if (data_layout == "NCHW" and kernel_layout == "OIHW") or (
            data_layout == "NCHW4c" and kernel_layout == "OIHW4o"
        ):
            if out_type.dtype == "float16":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.conv2d_nchwc),
                    wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc),
                    name="conv2d_nchwc.image2d",
                    plevel=10,
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nchwc_acc32),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc_acc32),
                name="conv2d_nchwc_tpack.image2d",
                plevel=20,
            )
        elif (data_layout == "NHWC" and kernel_layout == "HWIO") or (
            data_layout == "NHWC4c" and kernel_layout == "HWIO4o"
        ):
            if out_type.dtype == "float16":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.conv2d_nhwc),
                    wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc),
                    name="conv2d_nhwc.image2d",
                    plevel=10,
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nhwc_acc32),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc_acc32),
                name="conv2d_nhwc_acc32.image2d",
                plevel=20,
            )
        else:
            raise RuntimeError(
                "Layout not supported: ("
                + data_layout
                + ", "
                + kernel_layout
                + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
            )
    else:
        # cannot use is_depthwise_conv2d because it does not know about NHWC4c/HWOI4o layouts
        if data_layout == "NCHW":
            ic = data.shape[1]
        elif data_layout == "NCHW4c":
            ic = data.shape[1] * data.shape[4]
        elif data_layout == "NHWC":
            ic = data.shape[3]
        elif data_layout == "NHWC4c":
            ic = data.shape[3] * data.shape[4]
        else:
            raise RuntimeError("Unsupported depthwise_conv2d data layout {}".format(data_layout))
        if kernel_layout == "OIHW":
            oc = kernel.shape[0]
        elif kernel_layout == "OIHW4o":
            oc = kernel.shape[0] * kernel.shape[4]
        elif kernel_layout == "HWOI":
            oc = kernel.shape[2]
        elif kernel_layout == "HWOI4o":
            oc = kernel.shape[2] * kernel.shape[4]
        else:
            raise RuntimeError(
                "Unsupported depthwise_conv2d kernel layout {}".format(kernel_layout)
            )

        if ic == oc == groups:
            if (data_layout == "NCHW" and kernel_layout == "OIHW") or (
                data_layout == "NCHW4c" and kernel_layout == "OIHW4o"
            ):
                if out_type.dtype == "float16":
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nchwc),
                        wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nchwc),
                        name="depthwise_conv2d_nchwc.image2d",
                        plevel=10,
                    )
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nchwc_acc32),
                    wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nchwc_acc32),
                    name="depthwise_conv2d_nchwc_acc32.image2d",
                    plevel=20,
                )
            elif (data_layout == "NHWC" and kernel_layout == "HWOI") or (
                data_layout == "NHWC4c" and kernel_layout == "HWOI4o"
            ):
                if data.shape[-1] >= 4:
                    if out_type.dtype == "float16":
                        strategy.add_implementation(
                            wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nhwc),
                            wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nhwc),
                            name="depthwise_conv2d_nhwc.image2d",
                            plevel=10,
                        )
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nhwc_acc32),
                        wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nhwc_acc32),
                        name="depthwise_conv2d_nhwc_acc32.image2d",
                        plevel=20,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                        wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                        name="depthwise_conv2d_nhwc.cuda",
                    )
            else:
                raise RuntimeError(
                    "Layout not supported: ("
                    + data_layout
                    + ", "
                    + kernel_layout
                    + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
                )
        else:
            raise RuntimeError("General group convolution is not currently supported")
    return strategy


@schedule_pool.register("adreno")
def schedule_pool_adreno(attrs, outs, target):
    """schedule pooling ops for adreno"""
    with target:
        if attrs.layout == "NCHW4c":
            return topi.adreno.schedule_pool(outs, attrs.layout)
        return topi.cuda.schedule_pool(outs, attrs.layout)
