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

import numpy as np
import pytest
import sys

import tvm
import tvm.contrib.hexagon
from tvm.topi.testing import conv2d_nhwc_python


def build_conv2d(target):
    an, ah, aw, ac = (
        tvm.te.var("an"),
        tvm.te.var("ah"),
        tvm.te.var("aw"),
        tvm.te.var("ac"),
    )
    fh, fw, fo = tvm.te.var("fh"), tvm.te.var("fw"), tvm.te.var("fo")
    off_l, off_t = tvm.te.var("off_l"), tvm.te.var("off_t")
    stride_h, stride_w = tvm.te.var("stride_h"), tvm.te.var("stride_w")

    act_flat = tvm.te.placeholder(shape=(an, ah, aw, ac), dtype="float16", name="act_flat")
    wgt_flat = tvm.te.placeholder(shape=(fh, fw, ac, fo), dtype="float16", name="wgt_flat")

    out_flat = tvm.te.extern(
        shape=(an, (ah - fh) // stride_h + 1, (aw - fw) // stride_w + 1, fo),
        inputs=[act_flat, wgt_flat],
        fcompute=lambda ins, outs: tvm.tir.call_cpacked(
            "conv2d_packed",  # Function from TVM runtime
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
    {
        "act_shape": (1, 8, 4, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 10, 14, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 14, 6, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 14, 6, 3),
        "wgt_shape": (3, 3, 3, 64),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 14, 6, 3),
        "wgt_shape": (5, 5, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 8, 8, 3),
        "wgt_shape": (2, 2, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 14, 6, 64),
        "wgt_shape": (3, 3, 64, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 4, 4, 40),
        "wgt_shape": (3, 3, 40, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 4, 4, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 5, 5, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 6, 6, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 7, 7, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 8, 8, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 8, 8, 3),
        "wgt_shape": (5, 5, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 8, 8, 64),
        "wgt_shape": (2, 2, 64, 64),
        "inp_offset": (0, 0),
        "inp_stride": (1, 1),
    },
    {
        "act_shape": (1, 8, 4, 3),
        "wgt_shape": (3, 3, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (2, 2),
    },
    {
        "act_shape": (1, 14, 6, 3),
        "wgt_shape": (3, 3, 3, 64),
        "inp_offset": (0, 0),
        "inp_stride": (2, 2),
    },
    {
        "act_shape": (1, 14, 6, 3),
        "wgt_shape": (5, 5, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (2, 2),
    },
    {
        "act_shape": (1, 8, 8, 3),
        "wgt_shape": (2, 2, 3, 3),
        "inp_offset": (0, 0),
        "inp_stride": (2, 2),
    },
]


def gen_id(param):
    """Utility function to generate useful ids for shape_parameters"""

    dims = lambda vals: "x".join(map(str, vals))

    act_shape = param["act_shape"]
    wgt_shape = param["wgt_shape"]
    inp_stride = param["inp_stride"]
    return f"nhwc{dims(act_shape)}-hwio{dims(wgt_shape)}-stride{dims(inp_stride)}"


@tvm.testing.requires_hexagon
@pytest.mark.parametrize("shapes", shape_parameters, ids=map(gen_id, shape_parameters))
def test_conv2d(shapes, hexagon_session):
    act_shape = shapes["act_shape"]
    wgt_shape = shapes["wgt_shape"]
    inp_offset = shapes["inp_offset"]
    inp_stride = shapes["inp_stride"]
    assert act_shape[3] == wgt_shape[2]

    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

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

    module = build_conv2d(target)

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
