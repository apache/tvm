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
"""Verilator codegen tests"""

import os
import numpy as np

import tvm
from tvm import relay

from test_verilator.infrastructure import (
    _register_verilator_op,
    skip_test,
    compile_hardware,
    compile_module,
    run_module,
    offload,
)


_register_verilator_op("add")


def create_module_add(shape, dtype):
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def run_check_add(exe, shape, dtype):
    x_data = np.random.randint(5, size=shape, dtype=dtype)
    y_data = np.random.randint(5, size=shape, dtype=dtype)
    ref = x_data + y_data
    inputs = {"x": x_data, "y": y_data}
    out = run_module(exe, inputs)
    tvm.testing.assert_allclose(out.asnumpy(), ref, rtol=1e-5, atol=1e-5)


def test_add():
    if skip_test():
        return
    dtype = "int32"
    shape = (8, 4)
    mod = create_module_add(shape, dtype)
    mod = offload(mod)
    exe = compile_module(mod)
    run_check_add(exe, shape, dtype)


if __name__ == "__main__":
    test_add()
