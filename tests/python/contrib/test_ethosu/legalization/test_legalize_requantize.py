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
# pylint: disable=invalid-name, unused-argument

import pytest

pytest.importorskip("ethosu.vela")

import math

import tvm
from tvm.relay.backend.contrib.ethosu import legalize
from tvm import relay
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tests.python.contrib.test_ethosu.legalization import legalize_infra


@pytest.mark.parametrize(
    "ifm_shape,ifm_scale,ifm_zp,ofm_scale,ofm_zp",
    [[(1, 8, 8, 3), 1.0, 0, 1.0, 0], [(1, 20, 30, 3), 1.345, 34, 0.32, -23]],
)
def test_ethosu_requantize(ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    dtype = "int8"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        requantize = relay.qnn.op.requantize(
            ifm,
            relay.const(ifm_scale, dtype="float32"),
            relay.const(ifm_zp, dtype="int32"),
            relay.const(ofm_scale, dtype="float32"),
            relay.const(ofm_zp, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    def verify(ext_func):
        op = ext_func.body

        # Check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype

        # Check OFM
        ofm = op.checked_type
        assert list(ofm.shape) == list(ifm_shape)
        assert str(ofm.dtype) == dtype

        # Check quantization params
        assert math.isclose(op.attrs.ifm_scale, ifm_scale, abs_tol=1e-7)
        assert op.attrs.ifm_zero_point == ifm_zp
        assert math.isclose(op.attrs.ofm_scale, ofm_scale, abs_tol=1e-7)
        assert op.attrs.ofm_zero_point == ofm_zp

    rewriter = legalize.RequantizeRewriter()
    pattern_table = [
        (
            ethosu.RequantizeParams.composite_name,
            ethosu.requantize_pattern(),
            lambda pat: ethosu.RequantizeParams(pat).is_valid(),
        ),
    ]

    mod = create_model()
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
