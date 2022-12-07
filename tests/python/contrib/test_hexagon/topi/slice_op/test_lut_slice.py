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

"""Test for LUT"""

import math

import tvm
from tvm import te
import tvm.testing
from tvm.contrib.hexagon.session import Session
import tvm.contrib.hexagon
import numpy as np
from ...infrastructure import quantize_np, get_hexagon_target

HEX_TARGET = get_hexagon_target("v68")


class TestLUT:
    """Class containing test params for LUT testing"""

    shape, func, dtype = tvm.testing.parameters(
        ([1, 8, 8, 32], {"py": np.sqrt, "tvm": tvm.topi.hexagon.qnn.injective.qsqrt}, "uint8"),
        ([1024], {"py": np.sqrt, "tvm": tvm.topi.hexagon.qnn.injective.qsqrt}, "uint8"),
        ([1, 8, 8, 32], {"py": math.exp, "tvm": tvm.topi.hexagon.qnn.injective.qexp}, "uint8"),
        ([1024], {"py": math.exp, "tvm": tvm.topi.hexagon.qnn.injective.qexp}, "uint8"),
        ([1, 8, 8, 32], {"py": math.erf, "tvm": tvm.topi.hexagon.qnn.injective.qerf}, "uint8"),
        ([1024], {"py": math.erf, "tvm": tvm.topi.hexagon.qnn.injective.qerf}, "uint8"),
        ([1, 8, 8, 32], {"py": np.tanh, "tvm": tvm.topi.hexagon.qnn.injective.qtanh}, "uint8"),
        ([1024], {"py": np.tanh, "tvm": tvm.topi.hexagon.qnn.injective.qtanh}, "uint8"),
    )

    @tvm.testing.requires_hexagon
    def test_lut(
        self,
        hexagon_session: Session,
        shape,
        func,
        dtype,
    ):
        """Main test function. Compares py function output to hexagon output."""

        # Make input
        a_np = np.random.random(shape)
        a_np_quant, in_scale, in_zero = quantize_np(a_np, dtype)

        # Get golden
        golden = np.vectorize(func["py"])(in_scale * (a_np_quant - in_zero))
        golden_quant, out_scale, out_zero = quantize_np(golden, dtype)

        a_tensor = te.placeholder(shape, name="a_tensor", dtype=dtype)
        o_tensor = func["tvm"](a_tensor, in_scale, in_zero, out_scale, out_zero, dtype=dtype)
        sch = tvm.topi.hexagon.lut.lutize(o_tensor, a_tensor)

        # Lower hexagon code
        with tvm.transform.PassContext(opt_level=3):
            hex_lowered = tvm.build(
                sch,
                [a_tensor, o_tensor],
                tvm.target.Target(HEX_TARGET, host=HEX_TARGET),
                name="LUT",
            )

        # Run hexagon code
        mod = hexagon_session.load_module(hex_lowered)
        dev = hexagon_session.device
        a_buf = tvm.nd.array(a_np_quant.astype(dtype), dev)
        o_buf = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
        mod(a_buf, o_buf)

        np.testing.assert_allclose(golden_quant, o_buf.numpy(), rtol=0, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
