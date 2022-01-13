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
"""Test DNNL integration conv2d tests."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.dnnl import partition_for_dnnl

from common import (
    requires_dnnl,
    parametrized,
    check_result,
    check_fully_annotated,
    Builder,
    filler_uni,
)
from common import ConvProfile, ArgConstConfig, QuantizationConfig
from common import permute, expand_dim

acp_regular = ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=None)
acp_no_bias = ArgConstConfig(Data=False, Weights=True, Bias=None, Sum=None)
acp_with_sum = ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=False)
acp_no_bias_with_sum = ArgConstConfig(Data=False, Weights=True, Bias=None, Sum=False)

# Basic convolution 3x3. More trivial, symmetric
base_conv = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=1,
    PW=1,
    DH=1,
    DW=1,
    GR=1,
    D_LAYOUT="NCHW",
    K_LAYOUT="OIHW",
)

# same as Basic but with NHWC data layout
base_conv_nhwc = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=1,
    PW=1,
    DH=1,
    DW=1,
    GR=1,
    D_LAYOUT="NHWC",
    K_LAYOUT="HWIO",
)

base_conv_no_pad = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=0,
    PW=0,
    DH=1,
    DW=1,
    GR=1,
    D_LAYOUT="NCHW",
    K_LAYOUT="OIHW",
)

base_conv_no_pad_nhwc = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=0,
    PW=0,
    DH=1,
    DW=1,
    GR=1,
    D_LAYOUT="NHWC",
    K_LAYOUT="HWIO",
)

# same as Basic but with groups
base_conv_group_no_pad = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=0,
    PW=0,
    DH=1,
    DW=1,
    GR=2,
    D_LAYOUT="NCHW",
    K_LAYOUT="OIHW",
)

# same as Basic but with group == IC == OC
base_conv_dw_no_pad = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=16,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=0,
    PW=0,
    DH=1,
    DW=1,
    GR=16,
    D_LAYOUT="NCHW",
    K_LAYOUT="OIHW",
)

base_conv_dilated = ConvProfile(
    N=1,
    IH=5,
    IW=5,
    IC=8,
    OC=16,
    KH=3,
    KW=3,
    SH=1,
    SW=1,
    PH=2,
    PW=2,
    DH=2,
    DW=2,
    GR=1,
    D_LAYOUT="NCHW",
    K_LAYOUT="OIHW",
)

conv_profiles = [
    ("Base", base_conv, acp_regular),
    ("NHWC", base_conv_nhwc, acp_regular),
    ("Group", base_conv_group_no_pad, acp_regular),
    ("DW", base_conv_dw_no_pad, acp_regular),
    ("Dilated", base_conv_dilated, acp_regular),
]


@requires_dnnl
@parametrized("profile", conv_profiles)
def test_conv2d(profile):
    def generate_model(p, c):
        np.random.seed(0)

        d_shape = [p.N, p.IC, p.IH, p.IW]
        w_shape = [p.OC, p.IC, p.KH, p.KW]
        b_shape = [p.OC]
        s_shape = [
            p.N,
            p.OC,
            (p.IH + 2 * p.PH - (p.KH - 1) * p.DH - 1) // p.SH + 1,
            (p.IW + 2 * p.PW - (p.KW - 1) * p.DW - 1) // p.SW + 1,
        ]

        if p.GR != 1:
            w_shape[1] //= p.GR

        d_shape = permute(d_shape, l_from="NCHW", l_to=p.D_LAYOUT)
        s_shape = permute(s_shape, l_from="NCHW", l_to=p.D_LAYOUT)
        w_shape = permute(w_shape, l_from="OIHW", l_to=p.K_LAYOUT)

        c_dim = p.D_LAYOUT.find("C")
        # b_shape = expand_dim(b_shape, rank=len(p.D_LAYOUT) - c_dim)

        bld = Builder()

        op = bld.arg(shape=d_shape, dtype="float32", is_const=c.Data)
        wgh = bld.arg(shape=w_shape, dtype="float32", is_const=c.Weights)
        op = tvm.relay.nn.conv2d(
            op,
            wgh,
            kernel_size=[p.KH, p.KW],
            padding=[p.PH, p.PW],
            strides=[p.SH, p.SW],
            dilation=[p.DH, p.DW],
            groups=p.GR,
            channels=p.OC,
            out_dtype="float32",
            data_layout=p.D_LAYOUT,
            kernel_layout=p.K_LAYOUT,
        )

        if c.Bias is not None:
            bias = bld.arg(shape=b_shape, dtype="float32", is_const=c.Bias)
            op = tvm.relay.nn.bias_add(op, bias, axis=c_dim)
            # op = tvm.relay.add(op, bias)

        if c.Sum is not None:
            sum_in = bld.arg(shape=s_shape, dtype="float32", is_const=c.Sum)
            op = tvm.relay.op.add(op, sum_in)

        return bld.finalize(op)

    conv_p, arg_p = profile
    ref_mod, args = generate_model(conv_p, arg_p)
    mod = partition_for_dnnl(ref_mod)

    # atol=1 means int values should match with +-1 quantum value tolerance
    check_result(mod, ref_mod, args, tol=1e-10, atol=1)


# Regular and simple quantization scheme. All tensors are quantized per tensor.
# Data and weights quantized symmetrically (zp == 0).
qp_regular = QuantizationConfig(
    d_zp=0,
    d_scl=0.2,
    d_pc=False,
    k_zp=0,
    k_scl=0.1,
    k_pc=False,
    rq_zp=30,
    rq_scl=0.2,
    rq_pc=False,
    sum_zp=15,
    sum_scl=0.3,
    sum_pc=False,
    o_zp=5,
    o_scl=0.2,
    o_pc=False,
)

# Like a Regular quantization scheme but with asymmetric data quantization.
qp_asymmetric_data = QuantizationConfig(
    d_zp=3,
    d_scl=0.2,
    d_pc=False,
    k_zp=0,
    k_scl=0.1,
    k_pc=False,
    rq_zp=10,
    rq_scl=0.1,
    rq_pc=False,
    sum_zp=5,
    sum_scl=0.3,
    sum_pc=False,
    o_zp=4,
    o_scl=0.2,
    o_pc=False,
)

qnn_conv_profiles = [
    #  Pattern Conv2d + Requantize
    ("Base", base_conv, acp_regular, qp_regular),
    ("NHWC", base_conv_nhwc, acp_regular, qp_regular),
    #  Asymmetric input. NOTE: No pad! Input ZP is not compatible with PAD
    ("Group", base_conv_group_no_pad, acp_regular, qp_asymmetric_data),
    ("DW", base_conv_dw_no_pad, acp_regular, qp_asymmetric_data),
    ("NoBias", base_conv, acp_no_bias, qp_regular),
    ("AsymmetricInput", base_conv_no_pad, acp_regular, qp_asymmetric_data),
    ("AsymmetricInput_NHWC", base_conv_no_pad_nhwc, acp_regular, qp_asymmetric_data),
    #  Pattern Conv2d + Requantize + Sum
    ("WithSum", base_conv_no_pad, acp_with_sum, qp_asymmetric_data),
    ("WithSum_NHWC", base_conv_no_pad_nhwc, acp_with_sum, qp_asymmetric_data),
    ("WithSum_NoBias", base_conv_no_pad, acp_no_bias_with_sum, qp_asymmetric_data),
]


@requires_dnnl
@parametrized("profile", qnn_conv_profiles)
def test_qnn_conv2d(profile):
    def generate_model(p, c, q):
        np.random.seed(0)

        d_shape = [p.N, p.IC, p.IH, p.IW]
        w_shape = [p.OC, p.IC, p.KH, p.KW]
        b_shape = [p.OC]
        s_shape = [
            p.N,
            p.OC,
            (p.IH + 2 * p.PH - (p.KH - 1) * p.DH - 1) // p.SH + 1,
            (p.IW + 2 * p.PW - (p.KW - 1) * p.DW - 1) // p.SW + 1,
        ]

        if p.GR != 1:
            w_shape[1] //= p.GR

        d_shape = permute(d_shape, l_from="NCHW", l_to=p.D_LAYOUT)
        s_shape = permute(s_shape, l_from="NCHW", l_to=p.D_LAYOUT)
        w_shape = permute(w_shape, l_from="OIHW", l_to=p.K_LAYOUT)

        c_dim = p.D_LAYOUT.find("C")
        b_shape = expand_dim(b_shape, rank=len(p.D_LAYOUT) - c_dim)

        bld = Builder(qnn_profile=q)

        # Start build a test graph
        data = bld.arg(shape=d_shape, dtype="uint8", is_const=c.Data, filler=filler_uni(0, 20))
        d_zp, d_scl = bld.make_zp_and_scl("d", p.IC)

        # Convolution
        wgh = bld.arg(shape=w_shape, dtype="int8", is_const=c.Weights, filler=filler_uni(-20, 20))
        w_zp, w_scl = bld.make_zp_and_scl("k")

        op = tvm.relay.qnn.op.conv2d(
            data,
            wgh,
            d_zp,
            w_zp,
            d_scl,
            w_scl,
            kernel_size=[p.KH, p.KW],
            padding=[p.PH, p.PW],
            strides=[p.SH, p.SW],
            dilation=[p.DH, p.DW],
            groups=p.GR,
            channels=p.OC,
            out_dtype="int32",
            data_layout=p.D_LAYOUT,
            kernel_layout=p.K_LAYOUT,
        )
        # Optional bias
        if c.Bias is not None:
            bias = bld.arg(
                shape=b_shape, dtype="int32", is_const=c.Bias, filler=filler_uni(-50, 50)
            )
            op = tvm.relay.add(op, bias)

        # Re-quantization
        rq_in_zp = bld.make_zp(0)
        rq_in_scl = bld.make_scl(q.d_scl * q.k_scl)  # in real cases that should be a vector
        rq_out_zp, rq_out_scl = bld.make_zp_and_scl("rq")

        op = tvm.relay.qnn.op.requantize(
            op, rq_in_scl, rq_in_zp, rq_out_scl, rq_out_zp, out_dtype="int32"
        )
        op = tvm.relay.clip(
            op, a_min=0.0, a_max=255.0
        )  # pytorch frontend specific, I guess it's redundant
        op = tvm.relay.cast(op, dtype="uint8")

        # Optional sum (ResNet like)
        if c.Sum is not None:
            sum_in = bld.arg(dtype="uint8", shape=s_shape, filler=filler_uni(0, 10), is_const=c.Sum)

            lhs_zp, lhs_scl = bld.make_zp_and_scl("rq")
            rhs_zp, rhs_scl = bld.make_zp_and_scl("sum")
            out_zp, out_scl = bld.make_zp_and_scl("o")

            op = tvm.relay.qnn.op.add(op, sum_in, lhs_scl, lhs_zp, rhs_scl, rhs_zp, out_scl, out_zp)
            op = tvm.relay.clip(op, a_min=0.0, a_max=255.0)

        return bld.finalize(op)

    conv_p, arg_p, quant_p = profile
    ref_mod, args = generate_model(conv_p, arg_p, quant_p)
    mod = partition_for_dnnl(ref_mod)

    # atol=1 means int values should match with +-1 quantum value tolerance
    check_result(mod, ref_mod, args, tol=1e-10, atol=1)
