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
"""Test DNNL integration dense tests."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.dnnl import partition_for_dnnl, get_dnnl_version

from .common import requires_dnnl, parametrized, check_result, Builder, filler_uni
from .common import DenseProfile, ArgConstConfig, QuantizationConfig

base_dense_profile = DenseProfile(N=2, IC=10, OC=16)
regular_const_arg_prof = ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=None)
cp_with_sum = ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=False)

dense_profiles = [
    ("Base", base_dense_profile, ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=None)),
    ("WithSum", base_dense_profile, ArgConstConfig(Data=False, Weights=True, Bias=True, Sum=False)),
]


@requires_dnnl
@parametrized("profile", dense_profiles)
def test_dense(profile):
    def generate_model(p, c):
        np.random.seed(0)

        d_shape = [p.N, p.IC]
        w_shape = [p.OC, p.IC]
        b_shape = [p.OC]
        s_shape = [p.N, p.OC]

        c_dim = 1

        bld = Builder()

        op = bld.arg(shape=d_shape, dtype="float32", is_const=c.Data)
        wgh = bld.arg(shape=w_shape, dtype="float32", is_const=c.Weights)
        op = tvm.relay.nn.dense(op, wgh, out_dtype="float32")

        if c.Bias is not None:
            bias = bld.arg(shape=b_shape, dtype="float32", is_const=c.Bias)
            op = tvm.relay.nn.bias_add(op, bias, axis=c_dim)

        if c.Sum is not None:
            sum_in = bld.arg(shape=s_shape, dtype="float32", is_const=c.Sum)
            op = tvm.relay.op.add(op, sum_in)

        return bld.finalize(op)

    dense_p, arg_p = profile
    ref_mod, args = generate_model(dense_p, arg_p)
    mod = partition_for_dnnl(ref_mod)
    check_result(mod, ref_mod, args, tol=1e-10, atol=1)


qp_regular = QuantizationConfig(
    d_zp=0,
    d_scl=0.2,
    d_pc=False,
    k_zp=0,
    k_scl=0.1,
    k_pc=False,
    rq_zp=30,
    rq_scl=0.2,
    rq_pc=False,  # asymmetric
    sum_zp=15,
    sum_scl=0.3,
    sum_pc=False,  # asymmetric
    o_zp=5,
    o_scl=0.2,
    o_pc=False,  # asymmetric
)
qp_asymmetric_all = QuantizationConfig(
    d_zp=3,
    d_scl=0.2,
    d_pc=False,  # asymmetric
    k_zp=0,
    k_scl=0.1,
    k_pc=False,
    rq_zp=10,
    rq_scl=0.1,
    rq_pc=False,  # asymmetric
    sum_zp=5,
    sum_scl=0.3,
    sum_pc=False,  # asymmetric
    o_zp=4,
    o_scl=0.2,
    o_pc=False,  # asymmetric
)

qnn_dense_profiles = [
    #  Pattern Dense + Requantize
    ("Base", base_dense_profile, regular_const_arg_prof, qp_regular),
    ("AsymmetricInput", base_dense_profile, regular_const_arg_prof, qp_asymmetric_all),
    #  Pattern Dense + Requantize + Sum
    ("AsymmetricInput_Sum", base_dense_profile, cp_with_sum, qp_asymmetric_all),
]


@requires_dnnl
@parametrized("profile", qnn_dense_profiles)
def test_qnn_dense(profile):
    def generate_model(p, c, q):
        np.random.seed(0)

        d_shape = [p.N, p.IC]
        w_shape = [p.OC, p.IC]
        b_shape = [p.OC]
        s_shape = [p.N, p.OC]

        bld = Builder(qnn_profile=q)

        # Start build a test graph
        data = bld.arg(shape=d_shape, dtype="uint8", is_const=c.Data, filler=filler_uni(0, 20))
        d_zp, d_scl = bld.make_zp_and_scl("d", p.IC)

        # Convolution
        wgh = bld.arg(shape=w_shape, dtype="int8", is_const=c.Weights, filler=filler_uni(-20, 20))
        w_zp, w_scl = bld.make_zp_and_scl("k")

        op = tvm.relay.qnn.op.dense(
            data, wgh, d_zp, w_zp, d_scl, w_scl, units=p.OC, out_dtype="int32"
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

    # WA. Old DNNL versions don't support dense+sum int8 pattern(dst_zp and per channel o_scale are not supported).
    # desired_compiler == None skip verification of full dnnl offload.
    desired_compiler = None if arg_p.Sum is not None and get_dnnl_version() < (2, 2) else "dnnl"

    # atol=1 means int values should match with +-1 quantum value tolerance
    check_result(mod, ref_mod, args, tol=1e-10, atol=1, desired_compiler=desired_compiler)
