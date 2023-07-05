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

""" Test quantized conv2d HVX intrinsic implementation"""

import numpy as np

import tvm
import tvm.contrib.hexagon
from tvm.topi.hexagon.utils import get_fixed_point_value
from tvm.topi.testing import conv2d_nhwc_python

from ..infrastructure import get_hexagon_target, quantize_np


def build_conv2d(target):
    """Build and return the conv2d IRModule that calls the intrinsic implementation"""
    act_n, act_h, act_w, act_c = (
        tvm.te.var("an"),
        tvm.te.var("ah"),
        tvm.te.var("aw"),
        tvm.te.var("ac"),
    )
    filt_h, filt_w, filt_o = tvm.te.var("filt_h"), tvm.te.var("filt_w"), tvm.te.var("filt_o")
    act_scale, act_zp = tvm.te.var("act_scale", dtype="float32"), tvm.te.var("act_zp")
    wgt_scale, wgt_zp = tvm.te.var("wgt_scale", dtype="float32"), tvm.te.var("wgt_zp")
    out_scale, out_zp = tvm.te.var("out_scale", dtype="float32"), tvm.te.var("out_zp")
    fixed_final_scale, scale_factor = tvm.te.var("fixed_final_scale", dtype="int32"), tvm.te.var(
        "scale_factor"
    )
    stride_h, stride_w = tvm.te.var("stride_h"), tvm.te.var("stride_w")

    act_flat = tvm.te.placeholder(
        shape=(act_n, act_h, act_w, act_c), dtype="uint8", name="act_flat"
    )
    wgt_flat = tvm.te.placeholder(
        shape=(filt_h, filt_w, act_c, filt_o), dtype="int8", name="wgt_flat"
    )

    out_flat = tvm.te.extern(
        shape=(act_n, (act_h - filt_h) // stride_h + 1, (act_w - filt_w) // stride_w + 1, filt_o),
        inputs=[act_flat, wgt_flat],
        fcompute=lambda ins, outs: tvm.tir.call_cpacked(
            "conv2d_packed_quant",  # Function from TVM runtime
            ins[0],
            ins[1],
            act_scale,
            act_zp,
            wgt_scale,
            wgt_zp,
            out_scale,
            out_zp,
            stride_h,
            stride_w,
            fixed_final_scale,
            scale_factor,
            outs[0],
            tvm.runtime.const(0),  # resource_handle (unused)
        ),
        dtype="uint8",
    )

    s = tvm.te.create_schedule(out_flat.op)

    func_name = "conv2d_quant_hvx"
    module = tvm.build(
        s,
        [
            act_flat,
            wgt_flat,
            act_scale,
            act_zp,
            wgt_scale,
            wgt_zp,
            out_scale,
            out_zp,
            stride_h,
            stride_w,
            fixed_final_scale,
            scale_factor,
            out_flat,
        ],
        target=target,
        name=func_name,
    )

    return module


def gen_config(params):
    """Utility function to generate useful ids for shape_parameters"""

    dims = lambda vals: "x".join(map(str, vals))

    config = {}
    for param in params:
        act_shape, wgt_shape, inp_stride = param
        name = f"nhwc{dims(act_shape)}-hwio{dims(wgt_shape)}-stride{dims(inp_stride)}"
        config[name] = param

    return config


class TestQuantConv2dIntrin:
    """Test Quantized Conv2d Intrin class"""

    shape_parameters = [
        [
            (1, 5, 5, 33),
            (3, 3, 33, 33),
            (1, 1),
        ],
        [
            (1, 9, 8, 64),
            (3, 3, 64, 64),
            (1, 1),
        ],
        [
            (1, 11, 16, 64),
            (3, 3, 64, 32),
            (1, 1),
        ],
        [
            (1, 24, 8, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 4, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 5, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 6, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 7, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 8, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 9, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 10, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 11, 3),
            (3, 3, 3, 3),
            (1, 1),
        ],
        [
            (1, 4, 4, 5),
            (3, 3, 5, 3),
            (1, 1),
        ],
    ]

    config = gen_config(shape_parameters)
    act_shape, wgt_shape, inp_stride = tvm.testing.parameters(*config.values(), ids=config.keys())
    inp_offset = tvm.testing.parameter((0, 0), ids=["offset0x0"])

    @tvm.testing.requires_hexagon
    def test_conv2d_quant(self, act_shape, wgt_shape, inp_stride, hexagon_session):
        """Test quantized conv2d intrinsic implementation"""
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

        act_f = np.random.uniform(-1.5, 1.0, size=act_shape).astype("float32")
        wgt_f = np.random.uniform(-1.5, 1.0, size=wgt_shape).astype("float32")

        # Quanize activations using onnxruntime
        act_q, act_scale, act_zp = quantize_np(act_f, dtype="uint8")
        act_q = act_q.reshape(act_f.shape)

        # Quanize weights using onnxruntime
        wgt_q, wgt_scale, wgt_zp = quantize_np(wgt_f, dtype="int8")
        wgt_q = wgt_q.reshape(wgt_f.shape)

        # Generate reference output
        ref_out = conv2d_nhwc_python(act_f, wgt_f, stride=inp_stride, padding="VALID")

        ref_out_q, out_scale, out_zp = quantize_np(ref_out, dtype="uint8")
        ref_out_q = ref_out_q.reshape(ref_out.shape)

        final_scale = act_scale * wgt_scale / out_scale
        fixed_final_scale, scale_factor = get_fixed_point_value(final_scale)

        module = build_conv2d(get_hexagon_target("v69"))
        mod = hexagon_session.load_module(module)

        output_shape = get_out_shape(act_shape, wgt_shape, inp_stride)

        output = tvm.nd.array(
            np.zeros(output_shape, dtype="uint8"),
            device=hexagon_session.device,
        )
        mod(
            tvm.nd.array(act_q, device=hexagon_session.device),
            tvm.nd.array(wgt_q, device=hexagon_session.device),
            act_scale,
            act_zp,
            wgt_scale,
            wgt_zp,
            out_scale,
            out_zp,
            inp_stride[0],  # stride_height
            inp_stride[1],  # stride_width
            fixed_final_scale,
            scale_factor,
            output,
        )

        out_q = output.numpy()

        tvm.testing.assert_allclose(out_q, ref_out_q, rtol=0, atol=2)


if __name__ == "__main__":
    tvm.testing.main()
