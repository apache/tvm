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


def verify_resize2d(
    batch,
    in_channel,
    in_height,
    in_width,
    size,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
):
    if layout == "NCHW":
        A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
        dtype = A.dtype
        out_shape = (batch, in_channel, *size)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout == "NHWC":
        A = te.placeholder((batch, in_height, in_width, in_channel), name="A")
        dtype = A.dtype
        out_shape = (
            batch,
            *size,
            in_channel,
        )
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout))

    B = topi.image.resize2d(
        A, [0, 0, -1, -1], size, layout, method, coordinate_transformation_mode, rounding_method
    )

    b_np = tvm.topi.testing.resize2d_python(
        a_np,
        (size[0] / in_height, size[1] / in_width),
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
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
def test_image_resize2d():
    # nearest_neighbor - NCHW
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="floor",
    )
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="round",
    )
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="ceil",
    )

    # nearest_neighbor - NHWC
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        layout="NHWC",
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="floor",
    )
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        layout="NHWC",
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="round",
    )
    verify_resize2d(
        8,
        16,
        32,
        32,
        (64, 64),
        layout="NHWC",
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="ceil",
    )


if __name__ == "__main__":
    test_image_resize2d()
