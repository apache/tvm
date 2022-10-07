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

""" Test conv2d HVX intrinsic implementation"""

import numpy as np

import tvm
import tvm.contrib.hexagon
from tvm.topi.testing import conv2d_nhwc_python

from ..infrastructure import get_hexagon_target


def build_conv2d(target):
    """Build and the return the conv2d module that calls the intrinsic implementation"""
    act_n, act_h, act_w, act_c = (
        tvm.te.var("act_n"),
        tvm.te.var("act_h"),
        tvm.te.var("act_w"),
        tvm.te.var("act_c"),
    )
    filt_h, filt_w, filt_o = tvm.te.var("filt_h"), tvm.te.var("fw"), tvm.te.var("filt_o")
    off_l, off_t = tvm.te.var("off_l"), tvm.te.var("off_t")
    stride_h, stride_w = tvm.te.var("stride_h"), tvm.te.var("stride_w")

    act_flat = tvm.te.placeholder(
        shape=(act_n, act_h, act_w, act_c), dtype="float16", name="act_flat"
    )
    wgt_flat = tvm.te.placeholder(
        shape=(filt_h, filt_w, act_c, filt_o), dtype="float16", name="wgt_flat"
    )

    out_flat = tvm.te.extern(
        shape=(act_n, (act_h - filt_h) // stride_h + 1, (act_w - filt_w) // stride_w + 1, filt_o),
        inputs=[act_flat, wgt_flat],
        fcompute=lambda ins, outs: tvm.tir.call_cpacked(
            "conv2d_packed_fp16",  # Function from TVM runtime
            ins[0],
            ins[1],
            off_t,
            off_l,
            stride_h,
            stride_w,
            outs[0],
            tvm.runtime.const(0),  # resource_handle (unused)
        ),
        dtype="float16",
    )

    s = tvm.te.create_schedule(out_flat.op)

    func_name = "extern_conv"
    with tvm.transform.PassContext(opt_level=3):
        module = tvm.build(
            s,
            [act_flat, wgt_flat, off_t, off_l, stride_h, stride_w, out_flat],
            target=target,
            name=func_name,
        )

    return module


shape_parameters = [
    (
        (1, 8, 4, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 10, 14, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 14, 6, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 14, 6, 3),
        (3, 3, 3, 64),
        (1, 1),
    ),
    (
        (1, 14, 6, 3),
        (5, 5, 3, 3),
        (1, 1),
    ),
    (
        (1, 8, 8, 3),
        (2, 2, 3, 3),
        (1, 1),
    ),
    (
        (1, 14, 6, 64),
        (3, 3, 64, 3),
        (1, 1),
    ),
    (
        (1, 4, 4, 40),
        (3, 3, 40, 3),
        (1, 1),
    ),
    (
        (1, 4, 4, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 5, 5, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 6, 6, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 7, 7, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 8, 8, 3),
        (3, 3, 3, 3),
        (1, 1),
    ),
    (
        (1, 8, 8, 3),
        (5, 5, 3, 3),
        (1, 1),
    ),
    (
        (1, 8, 8, 64),
        (2, 2, 64, 64),
        (1, 1),
    ),
    (
        (1, 8, 4, 3),
        (3, 3, 3, 3),
        (2, 2),
    ),
    (
        (1, 14, 6, 3),
        (3, 3, 3, 64),
        (2, 2),
    ),
    (
        (1, 14, 6, 3),
        (5, 5, 3, 3),
        (2, 2),
    ),
    (
        (1, 8, 8, 3),
        (2, 2, 3, 3),
        (2, 2),
    ),
]


def gen_config(params):
    """Utility function to generate useful ids for shape_parameters"""

    dims = lambda vals: "x".join(map(str, vals))

    config = {}
    for param in params:
        act_shape, wgt_shape, inp_stride = param
        name = f"nhwc{dims(act_shape)}-hwio{dims(wgt_shape)}-stride{dims(inp_stride)}"
        config[name] = param

    return config


class TestConv2dIntrin:
    """Test Conv2d Intrin class"""

    config = gen_config(shape_parameters)
    act_shape, wgt_shape, inp_stride = tvm.testing.parameters(*config.values(), ids=config.keys())
    inp_offset = tvm.testing.parameter((0, 0), ids=["offset0x0"])

    @tvm.testing.requires_hexagon
    def DISABLED_test_conv2d(self, act_shape, wgt_shape, inp_stride, inp_offset, hexagon_session):
        """Test conv2d intrinsic implementation"""
        assert act_shape[3] == wgt_shape[2]

        # Currently, input offset does not affect the output shape
        def get_out_shape(ash, wsh, inp_stride):
            assert ash[3] == wsh[2]
            osh = (
                ash[0],
                (ash[1] - wsh[0]) // inp_stride[0] + 1,
                (ash[2] - wsh[1]) // inp_stride[1] + 1,
                wsh[3],
            )
            assert tvm.tir.all([x > 0 for x in osh])
            return osh

        act = np.random.rand(*act_shape).astype("float16")
        wgt = np.random.rand(*wgt_shape).astype("float16")

        module = build_conv2d(get_hexagon_target("v68"))

        mod = hexagon_session.load_module(module)
        output = tvm.nd.array(
            np.zeros(get_out_shape(act_shape, wgt_shape, inp_stride), dtype="float16"),
            device=hexagon_session.device,
        )
        mod(
            tvm.nd.array(act, device=hexagon_session.device),
            tvm.nd.array(wgt, device=hexagon_session.device),
            inp_offset[0],  # off_t
            inp_offset[1],  # off_l
            inp_stride[0],  # stride_height
            inp_stride[1],  # stride_width
            output,
        )

        out = output.numpy()

        # Generate reference output and compare:
        ref_out = conv2d_nhwc_python(
            act.astype("float32"), wgt.astype("float32"), stride=inp_stride, padding="VALID"
        ).astype("float16")

        tvm.testing.assert_allclose(out, ref_out, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    tvm.testing.main()
