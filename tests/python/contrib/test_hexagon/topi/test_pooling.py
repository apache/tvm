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
"""Test code for pooling"""
import numpy as np

import tvm
import tvm.testing
from tvm import topi
from tvm import te
from tvm.contrib.hexagon.session import Session
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple

from ..infrastructure import get_hexagon_target


class TestAdaptivePool:
    """Adaptive pool test class."""

    dshape, out_size, pool_type, layout = tvm.testing.parameters(
        ((1, 3, 112, 112), (1, 1), "max", "NCHW"),
        ((1, 3, 112, 112), (1, 1), "avg", "NCHW"),
        ((1, 14, 56, 78), (34, 13), "max", "NCHW"),
        ((1, 5, 46, 97), (4, 96), "avg", "NCHW"),
        ((1, 112, 112, 3), (1, 1), "max", "NHWC"),
        ((1, 5, 46, 97), (4, 96), "avg", "NHWC"),
        ((1, 16, 32, 32, 32), (1, 1, 1), "max", "NCDHW"),
        ((1, 16, 32, 32, 32), (1, 1, 1), "avg", "NCDHW"),
        ((1, 16, 32, 32, 32), (2, 2, 2), "avg", "NCDHW"),
        (
            (1, 16, 64, 32, 32),
            (7, 8, 9),
            "avg",
            "NCDHW",
        ),
        (
            (1, 16, 64, 32, 32),
            (8, 16, 16),
            "avg",
            "NCDHW",
        ),
        ((1, 16, 32, 32, 32), (1, 1, 1), "avg", "NDHWC"),
        ((1, 16, 32, 32, 32), (2, 2, 2), "max", "NDHWC"),
        ((1, 16, 32, 32, 32), (2, 4, 4), "max", "NDHWC"),
    )

    @tvm.testing.requires_hexagon
    def test_adaptive_pool(self, hexagon_session: Session, dshape, out_size, pool_type, layout):
        """Test adaptive pool."""
        dtype = "float32"
        np_data = np.random.uniform(low=0, high=255, size=dshape).astype(dtype)
        np_out = tvm.topi.testing.adaptive_pool(np_data, out_size, pool_type, layout)
        oshape = np_out.shape

        data = te.placeholder(dshape, name="data", dtype=dtype)
        if len(out_size) == 2:
            out = topi.nn.adaptive_pool(data, out_size, pool_type, layout)
        else:
            assert len(out_size) == 3
            out = topi.nn.adaptive_pool3d(data, out_size, pool_type, layout)

        with tvm.target.Target(get_hexagon_target("v68")):
            fschedule = topi.hexagon.schedule_adaptive_pool
            s = fschedule(out)

        func = tvm.build(
            s,
            [data, out],
            get_hexagon_target("v68"),
            name="adaptive-pool",
        )
        mod = hexagon_session.load_module(func)

        dev = hexagon_session.device
        a = tvm.nd.array(np_data, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(oshape), dtype=out.dtype), dev)
        mod["adaptive-pool"](a, b)

        tvm.testing.assert_allclose(b.numpy(), np_out, rtol=4e-5, atol=1e-6)


def verify_poolnd(
    hexagon_session,
    n,
    input_shape,
    kernel,
    stride,
    dilation,
    padding,
    pool_type,
    ceil_mode,
    count_include_pad=True,
    layout="NCW",
):
    """Pool test verification."""
    a_tensor = te.placeholder(input_shape, name="a_tensor")

    if n == 1:
        b_tensor = topi.nn.pool1d(
            a_tensor,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            pool_type=pool_type,
            ceil_mode=ceil_mode,
            layout=layout,
            count_include_pad=count_include_pad,
        )
    elif n == 2:
        b_tensor = topi.nn.pool2d(
            a_tensor,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            pool_type=pool_type,
            ceil_mode=ceil_mode,
            layout=layout,
            count_include_pad=count_include_pad,
        )
    elif n == 3:
        b_tensor = topi.nn.pool3d(
            a_tensor,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            pool_type=pool_type,
            ceil_mode=ceil_mode,
            layout=layout,
            count_include_pad=count_include_pad,
        )
    else:
        raise ValueError(f"PoolND only supports n=1, 2, 3 got n={n}")

    b_tensor = topi.nn.relu(b_tensor)
    dtype = a_tensor.dtype
    output_shape = [int(i) for i in b_tensor.shape]

    input_np = np.random.uniform(low=0.001, size=input_shape).astype(dtype)

    padding_before = padding[:n]
    padding_after = padding[n:]
    ref_np = tvm.topi.testing.poolnd_python(
        input_np,
        kernel,
        stride,
        dilation,
        padding_before,
        padding_after,
        pool_type,
        count_include_pad,
        ceil_mode,
        layout=layout,
    )

    np.testing.assert_equal(tuple(output_shape), tuple(ref_np.shape))

    with tvm.target.Target(get_hexagon_target("v68")):
        fschedule = topi.hexagon.schedule_pool
        s = fschedule(b_tensor, layout)

    func = tvm.build(s, [a_tensor, b_tensor], get_hexagon_target("v68"), name="pool")
    mod = hexagon_session.load_module(func)

    dev = hexagon_session.device
    a = tvm.nd.array(input_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(b_tensor.shape), dtype=dtype), dev)
    mod["pool"](a, b)

    tvm.testing.assert_allclose(b.numpy(), ref_np, rtol=1e-5)


class TestPool1D:
    """Pool1D test class."""

    (
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ) = tvm.testing.parameters(
        ([1, 16, 32], [2], [2], [1], [0, 0], "avg", False, True, "NCW"),
        ([1, 16, 31], [3], [3], [1], [1, 2], "avg", False, True, "NCW"),
        ([1, 16, 32], [2], [2], [1], [1, 2], "avg", False, False, "NCW"),
        ([1, 16, 31], [4], [4], [1], [3, 3], "avg", False, False, "NCW"),
        ([1, 16, 31], [4], [4], [1], [0, 0], "avg", False, False, "NCW"),
        ([1, 16, 32], [2], [2], [1], [0, 0], "max", False, True, "NCW"),
        ([1, 16, 31], [3], [3], [1], [2, 1], "max", False, True, "NCW"),
        ([1, 16, 31], [3], [3], [1], [2, 1], "max", True, True, "NCW"),
        ([1, 16, 31], [3], [3], [1], [2, 5], "avg", False, True, "NCW"),
        ([1, 16, 32], [2], [2], [1], [0, 3], "avg", False, False, "NCW"),
        ([1, 16, 31], [3], [3], [1], [1, 4], "max", False, True, "NCW"),
        ([1, 16, 31], [3], [3], [1], [3, 0], "max", True, True, "NCW"),
        # Test non-1 dilations
        ([1, 16, 31], [3], [3], [2], [2, 5], "avg", False, True, "NCW"),
        ([1, 16, 32], [2], [2], [3], [0, 3], "avg", False, False, "NCW"),
        ([1, 16, 31], [3], [3], [2], [1, 4], "max", False, True, "NCW"),
        ([1, 16, 31], [3], [3], [3], [3, 0], "max", True, True, "NCW"),
        # Test Channel last
        ([1, 32, 16], [2], [2], [1], [0, 0], "avg", False, True, "NWC"),
        ([1, 31, 16], [3], [3], [1], [1, 2], "avg", False, True, "NWC"),
        ([1, 32, 16], [2], [2], [1], [1, 2], "avg", False, False, "NWC"),
        ([1, 31, 16], [4], [4], [1], [3, 3], "avg", False, False, "NWC"),
        ([1, 31, 16], [4], [4], [1], [0, 0], "avg", False, False, "NWC"),
        ([1, 32, 16], [2], [2], [1], [0, 0], "max", False, True, "NWC"),
        ([1, 31, 16], [3], [3], [1], [2, 1], "max", False, True, "NWC"),
        ([1, 31, 16], [3], [3], [1], [2, 1], "max", True, True, "NWC"),
        ([1, 31, 16], [3], [3], [1], [2, 5], "avg", False, True, "NWC"),
        ([1, 31, 16], [2], [2], [1], [0, 3], "avg", False, False, "NWC"),
        ([1, 31, 16], [3], [3], [1], [1, 4], "max", False, True, "NWC"),
        ([1, 31, 16], [3], [3], [1], [3, 0], "max", True, True, "NWC"),
        ([1, 31, 16], [3], [3], [2], [2, 5], "avg", False, True, "NWC"),
        ([1, 32, 16], [2], [2], [3], [0, 3], "avg", False, False, "NWC"),
        ([1, 31, 16], [3], [3], [2], [1, 4], "max", False, True, "NWC"),
        ([1, 31, 16], [3], [3], [3], [3, 0], "max", True, True, "NWC"),
    )

    @tvm.testing.requires_hexagon
    def test_pool1d(
        self,
        hexagon_session: Session,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ):
        """Test Pool1D."""
        verify_poolnd(
            hexagon_session,
            1,
            input_shape,
            kernel,
            stride,
            dilation,
            padding,
            pool_type,
            ceil_mode,
            count_include_pad,
            layout,
        )


class TestPool2D:
    """Pool2D test class."""

    (
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ) = tvm.testing.parameters(
        ([1, 16, 32, 32], [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "avg", False, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [1, 2, 1, 2], "avg", False, True, "NCHW"),
        ([1, 16, 32, 32], [2, 2], [2, 2], [1, 1], [1, 2, 1, 2], "avg", False, False, "NCHW"),
        ([1, 16, 31, 31], [4, 4], [4, 4], [1, 1], [3, 3, 3, 3], "avg", False, False, "NCHW"),
        ([1, 16, 31, 31], [4, 4], [4, 4], [1, 1], [0, 0, 0, 0], "avg", False, False, "NCHW"),
        ([1, 16, 32, 32], [2, 3], [2, 2], [1, 1], [0, 0, 0, 0], "max", False, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [2, 1, 2, 1], "max", False, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [2, 1, 2, 1], "max", True, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [2, 1, 0, 3], "avg", False, True, "NCHW"),
        ([1, 16, 32, 32], [2, 3], [2, 2], [1, 1], [0, 3, 2, 1], "avg", False, False, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [1, 0, 3, 2], "max", False, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [1, 1], [3, 2, 1, 0], "max", True, True, "NCHW"),
        # Test non-1 dilations
        ([1, 16, 31, 31], [3, 3], [3, 3], [2, 1], [2, 1, 0, 3], "avg", False, True, "NCHW"),
        ([1, 16, 32, 32], [2, 3], [2, 2], [2, 3], [0, 3, 2, 1], "avg", False, False, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [3, 3], [1, 0, 3, 2], "max", False, True, "NCHW"),
        ([1, 16, 31, 31], [3, 3], [3, 3], [2, 2], [3, 2, 1, 0], "max", True, True, "NCHW"),
        # Test channel last
        ([1, 32, 32, 16], [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "avg", False, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [1, 2, 1, 2], "avg", False, True, "NHWC"),
        ([1, 32, 32, 16], [2, 2], [2, 2], [1, 1], [1, 2, 1, 2], "avg", False, False, "NHWC"),
        ([1, 31, 31, 16], [4, 4], [4, 4], [1, 1], [3, 3, 3, 3], "avg", False, False, "NHWC"),
        ([1, 31, 31, 16], [4, 4], [4, 4], [1, 1], [0, 0, 0, 0], "avg", False, False, "NHWC"),
        ([1, 32, 32, 16], [2, 3], [2, 2], [1, 1], [0, 0, 0, 0], "max", False, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [2, 1, 2, 1], "max", False, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [2, 1, 2, 1], "max", True, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [2, 1, 0, 3], "avg", False, True, "NHWC"),
        ([1, 32, 32, 16], [2, 3], [2, 2], [1, 1], [0, 3, 2, 1], "avg", False, False, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [1, 0, 3, 2], "max", False, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [1, 1], [3, 2, 1, 0], "max", True, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [2, 1], [2, 1, 0, 3], "avg", False, True, "NHWC"),
        ([1, 32, 32, 16], [2, 3], [2, 2], [2, 3], [0, 3, 2, 1], "avg", False, False, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [3, 3], [1, 0, 3, 2], "max", False, True, "NHWC"),
        ([1, 31, 31, 16], [3, 3], [3, 3], [2, 2], [3, 2, 1, 0], "max", True, True, "NHWC"),
    )

    @tvm.testing.requires_hexagon
    def test_pool2d(
        self,
        hexagon_session: Session,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ):
        """Test Pool2D."""
        verify_poolnd(
            hexagon_session,
            2,
            input_shape,
            kernel,
            stride,
            dilation,
            padding,
            pool_type,
            ceil_mode,
            count_include_pad,
            layout,
        )


class TestPool3D:
    """Pool3D test class."""

    (
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ) = tvm.testing.parameters(
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "avg",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            "avg",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [4, 4, 4],
            [4, 4, 4],
            [1, 1, 1],
            [3, 3, 3, 3, 3, 3],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [4, 4, 4],
            [4, 4, 4],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "max",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 2, 1, 1, 1, 2],
            "max",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 2, 1, 1, 1, 2],
            "max",
            True,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 1, 0, 5, 4, 3],
            "avg",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 5, 4, 3, 2, 1],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [1, 0, 5, 4, 3, 2],
            "max",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [3, 2, 1, 0, 5, 4],
            "max",
            True,
            True,
            "NCDHW",
        ),
        # Test non-1 dilation
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [2, 1, 0, 5, 4, 3],
            "avg",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [0, 5, 4, 3, 2, 1],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [2, 1, 3],
            [1, 0, 5, 4, 3, 2],
            "max",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [2, 2, 3],
            [3, 2, 1, 0, 5, 4],
            "max",
            True,
            True,
            "NCDHW",
        ),
        # Test channel last layouts
        (
            [1, 32, 32, 32, 16],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "avg",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            "avg",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 32, 32, 32, 16],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            "avg",
            False,
            False,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [4, 4, 4],
            [4, 4, 4],
            [1, 1, 1],
            [3, 3, 3, 3, 3, 3],
            "avg",
            False,
            False,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [4, 4, 4],
            [4, 4, 4],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "avg",
            False,
            False,
            "NDHWC",
        ),
        (
            [1, 32, 32, 32, 16],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            "max",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 2, 1, 1, 1, 2],
            "max",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 2, 1, 1, 1, 2],
            "max",
            True,
            True,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [2, 1, 0, 5, 4, 3],
            "avg",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 32, 32, 32, 16],
            [2, 2, 2],
            [2, 2, 2],
            [1, 1, 1],
            [0, 5, 4, 3, 2, 1],
            "avg",
            False,
            False,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [1, 0, 5, 4, 3, 2],
            "max",
            False,
            True,
            "NDHWC",
        ),
        (
            [1, 31, 31, 31, 16],
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 1],
            [3, 2, 1, 0, 5, 4],
            "max",
            True,
            True,
            "NDHWC",
        ),
        # Test non-1 dilation
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [2, 1, 0, 5, 4, 3],
            "avg",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 32, 32, 32],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [0, 5, 4, 3, 2, 1],
            "avg",
            False,
            False,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [2, 1, 3],
            [1, 0, 5, 4, 3, 2],
            "max",
            False,
            True,
            "NCDHW",
        ),
        (
            [1, 16, 31, 31, 31],
            [3, 3, 3],
            [3, 3, 3],
            [2, 2, 3],
            [3, 2, 1, 0, 5, 4],
            "max",
            True,
            True,
            "NCDHW",
        ),
    )

    @tvm.testing.requires_hexagon
    def test_pool3d(
        self,
        hexagon_session: Session,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        pool_type,
        ceil_mode,
        count_include_pad,
        layout,
    ):
        """Test Pool3D."""
        verify_poolnd(
            hexagon_session,
            3,
            input_shape,
            kernel,
            stride,
            dilation,
            padding,
            pool_type,
            ceil_mode,
            count_include_pad,
            layout,
        )


if __name__ == "__main__":
    tvm.testing.main()
