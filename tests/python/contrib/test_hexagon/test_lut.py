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

import pytest
import math

import tvm
from tvm import te
import tvm.testing
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.build import HexagonLauncherRPC
import tvm.contrib.hexagon
from tvm import relay
import numpy as np
from .infrastructure import allocate_hexagon_array, transform_numpy, quantize_np

hex_target = tvm.target.hexagon("v68", link_params=True)

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
    hexagon_server_process,
    hexagon_launcher: HexagonLauncherRPC,
    hexagon_session: Session,
    shape,
    func,
    dtype,
):

    # Make input
    a_np = np.random.random(shape)
    a_np_quant, in_scale, in_zero = quantize_np(a_np, dtype)

    # Get golden
    golden = np.vectorize(func["py"])(in_scale * (a_np_quant - in_zero))
    golden_quant, out_scale, out_zero = quantize_np(golden, dtype)

    A = te.placeholder(shape, name="A", dtype=dtype)
    O = func["tvm"](A, in_scale, in_zero, out_scale, out_zero, dtype=dtype)
    s = tvm.topi.hexagon.lut.lutize(O, A)

    # Lower hexagon code
    with tvm.transform.PassContext(opt_level=3):
        hex_lowered = tvm.build(
            s, [A, O], tvm.target.Target(hex_target, host=hex_target), name="LUT"
        )

    # Run hexagon code
    mod = hexagon_session.load_module(hex_lowered)
    dev = hexagon_session.device
    a = tvm.nd.array(a_np_quant.astype(dtype), dev)
    o = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
    mod(a, o)

    np.testing.assert_allclose(golden_quant, o.numpy(), rtol=0, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
