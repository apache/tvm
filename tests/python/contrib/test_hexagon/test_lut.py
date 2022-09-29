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

import tvm
from tvm import te
import tvm.testing
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.build import HexagonLauncherRPC
import tvm.contrib.hexagon
from tvm import relay

import numpy as np

hex_target = tvm.target.hexagon("v68", link_params=True)

shape, func, dtype = tvm.testing.parameters(
    ([1, 8, 8, 32], {"py": np.sqrt, "tvm": tvm.topi.hexagon.injective.sqrt}, "uint8"),
    ([1024], {"py": np.sqrt, "tvm": tvm.topi.hexagon.injective.sqrt}, "uint8"),
    ([1, 8, 8, 32], {"py": lambda x: -x, "tvm": tvm.topi.hexagon.injective.negative}, "int8"),
    ([1024], {"py": lambda x: -x, "tvm": tvm.topi.hexagon.injective.negative}, "int8"),
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
    if dtype == "int8":
        a_np = np.random.randint(-128, 127, size=shape).astype(dtype)
    elif dtype == "uint8":
        a_np = np.random.randint(0, 255, size=shape).astype(dtype)
    else:
        raise ValueError(f"dtype {dtype} not supported for LUTs!")

    A = te.placeholder(shape, name="A", dtype=dtype)
    O = func["tvm"](A, dtype=dtype)
    s = tvm.topi.hexagon.lut.lutize(O, A)

    # Lower hexagon code
    with tvm.transform.PassContext(opt_level=3):
        hex_lowered = tvm.build(
            s, [A, O], tvm.target.Target(hex_target, host=hex_target), name="LUT"
        )

    # Run hexagon code
    mod = hexagon_session.load_module(hex_lowered)
    dev = hexagon_session.device
    a = tvm.nd.array(a_np, dev)
    o = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
    mod(a, o)

    # Get golden
    golden = func["py"](a_np).astype(dtype)

    np.testing.assert_allclose(golden, o.numpy(), rtol=0, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
