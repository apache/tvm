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
"""Test code for upsampling"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
import math
from tvm.topi.utils import nchw_pack_layout


def verify_upsampling(
    batch,
    in_channel,
    in_height,
    in_width,
    scale_h,
    scale_w,
    layout="NCHW",
    method="nearest_neighbor",
    in_batch_block=0,
    in_channel_block=0,
):
    if layout == "NCHW":
        A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
        dtype = A.dtype
        out_shape = (
            batch,
            in_channel,
            int(round(in_height * scale_h)),
            int(round(in_width * scale_w)),
        )
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif nchw_pack_layout(layout):
        A = te.placeholder(
            (batch, in_channel, in_height, in_width, in_batch_block, in_channel_block), name="A"
        )
        dtype = A.dtype
        out_shape = (
            batch,
            in_channel,
            int(round(in_height * scale_h)),
            int(round(in_width * scale_w)),
            in_batch_block,
            in_channel_block,
        )
        a_np = np.random.uniform(
            size=(batch, in_channel, in_height, in_width, in_batch_block, in_channel_block)
        ).astype(dtype)
    elif layout == "NHWC":
        A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
        dtype = A.dtype
        out_shape = (
            batch,
            int(round(in_height * scale_h)),
            int(round(in_width * scale_w)),
            in_channel,
        )
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout))

    B = topi.nn.upsampling(A, scale_h, scale_w, layout=layout, method=method, align_corners=False)

    b_np = tvm.topi.testing.resize2d_python(
        a_np,
        (scale_h, scale_w),
        layout,
        method[2:] if method[0:2] == "bi" else method,
        "asymmetric",
    )

    def check_target(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), dev)
        f = tvm.build(s, [A, B], target)
        f(a, b)

        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

    for target, dev in tvm.testing.enabled_targets():
        check_target(target, dev)


@tvm.testing.uses_gpu
def test_upsampling():
    # nearest_neighbor - NCHW
    verify_upsampling(8, 16, 32, 32, 2.0, 2.0)
    verify_upsampling(2, 32, 64, 64, 3.0, 3.0)
    verify_upsampling(1, 64, 22, 32, 1.954545497894287, 2.0)

    ## nearest_neighbor - NHWC
    verify_upsampling(8, 16, 32, 32, 2.0, 2.0, layout="NHWC")
    verify_upsampling(2, 32, 64, 64, 3.0, 3.0, layout="NHWC")
    verify_upsampling(1, 64, 22, 32, 1.954545497894287, 2.0, layout="NHWC")

    # bilinear - NCHW
    verify_upsampling(2, 2, 32, 32, 2.0, 2.0, method="bilinear")
    verify_upsampling(2, 2, 32, 32, 3.0, 3.0, method="bilinear")
    verify_upsampling(1, 64, 22, 32, 1.954545497894287, 2.0, method="bilinear")

    # nearest_neighbor - NCHWinic
    verify_upsampling(2, 2, 32, 32, in_batch_block=4, in_channel_block=8, scale_h=2.0, scale_w=2.0)
    verify_upsampling(2, 2, 64, 64, in_batch_block=1, in_channel_block=16, scale_h=3.0, scale_w=3.0)
    verify_upsampling(
        1, 4, 22, 32, in_batch_block=1, in_channel_block=16, scale_h=1.954545497894287, scale_w=2.0
    )

    # bilinear - NCHWinic
    verify_upsampling(
        2,
        2,
        32,
        32,
        in_batch_block=1,
        in_channel_block=1,
        scale_h=2.0,
        scale_w=2.0,
        method="bilinear",
    )
    verify_upsampling(
        2,
        2,
        32,
        32,
        in_batch_block=1,
        in_channel_block=1,
        scale_h=3.0,
        scale_w=3.0,
        method="bilinear",
    )
    verify_upsampling(
        2,
        4,
        22,
        32,
        in_batch_block=1,
        in_channel_block=16,
        scale_h=1.954545497894287,
        scale_w=2.0,
        layout="NCHW1n16c",
        method="bilinear",
    )

    # bilinear - NHWC
    verify_upsampling(2, 2, 32, 32, 2.0, 2.0, layout="NHWC", method="bilinear")
    verify_upsampling(2, 2, 32, 32, 3.0, 3.0, layout="NHWC", method="bilinear")
    verify_upsampling(1, 64, 22, 32, 3.0, 3.0, layout="NHWC", method="bilinear")


def verify_upsampling3d(
    batch,
    in_channel,
    in_depth,
    in_height,
    in_width,
    scale_d,
    scale_h,
    scale_w,
    layout="NCDHW",
    method="nearest_neighbor",
):
    if layout == "NCDHW":
        A = te.placeholder((batch, in_channel, in_depth, in_height, in_width), name="A")
        dtype = A.dtype
        out_shape = (
            batch,
            in_channel,
            int(round(in_depth * scale_d)),
            int(round(in_height * scale_h)),
            int(round(in_width * scale_w)),
        )
        a_np = np.random.uniform(size=(batch, in_channel, in_depth, in_height, in_width)).astype(
            dtype
        )
    elif layout == "NDHWC":
        A = te.placeholder((batch, in_depth, in_height, in_width, in_channel), name="A")
        dtype = A.dtype
        out_shape = (
            batch,
            int(round(in_depth * scale_d)),
            int(round(in_height * scale_h)),
            int(round(in_width * scale_w)),
            in_channel,
        )
        a_np = np.random.uniform(size=(batch, in_depth, in_height, in_width, in_channel)).astype(
            dtype
        )
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout))

    B = topi.nn.upsampling3d(
        A,
        scale_d,
        scale_h,
        scale_w,
        layout=layout,
        method=method,
        coordinate_transformation_mode="asymmetric",
    )

    b_np = tvm.topi.testing.resize3d_python(
        a_np,
        (scale_d, scale_h, scale_w),
        layout,
        method[3:] if method[0:3] == "tri" else method,
        "asymmetric",
    )

    def check_target(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), dev)
        f = tvm.build(s, [A, B], target)
        f(a, b)

        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

    for target, dev in tvm.testing.enabled_targets():
        check_target(target, dev)


@tvm.testing.uses_gpu
def test_upsampling3d():
    # nearest_neighbor - NCDHW
    verify_upsampling3d(8, 8, 16, 16, 16, 2.0, 2.0, 2.0)
    verify_upsampling3d(2, 16, 32, 32, 32, 3.0, 3.0, 3.0)
    verify_upsampling3d(1, 8, 11, 16, 6, 1.954545497894287, 2.0, 1.5)

    ## nearest_neighbor - NDHWC
    verify_upsampling3d(8, 8, 16, 16, 16, 2.0, 2.0, 2.0, layout="NDHWC")
    verify_upsampling3d(2, 16, 32, 32, 32, 3.0, 3.0, 3.0, layout="NDHWC")
    verify_upsampling3d(1, 8, 11, 16, 6, 1.954545497894287, 2.0, 1.5, layout="NDHWC")

    # trilinear - NCDHW
    verify_upsampling3d(2, 2, 16, 16, 16, 2.0, 2.0, 2.0, method="trilinear")
    verify_upsampling3d(2, 2, 32, 32, 32, 3.0, 3.0, 3.0, method="trilinear")
    verify_upsampling3d(1, 2, 11, 16, 6, 1.954545497894287, 2.0, 1.5, method="trilinear")

    # trilinear - NDHWC
    verify_upsampling3d(2, 2, 16, 16, 16, 2.0, 2.0, 2.0, layout="NDHWC", method="trilinear")
    verify_upsampling3d(2, 2, 32, 32, 32, 3.0, 3.0, 3.0, layout="NDHWC", method="trilinear")
    verify_upsampling3d(
        1, 2, 11, 16, 6, 1.954545497894287, 2.0, 1.5, layout="NDHWC", method="trilinear"
    )


if __name__ == "__main__":
    test_upsampling()
    test_upsampling3d()
