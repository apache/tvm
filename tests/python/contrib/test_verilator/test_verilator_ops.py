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

import numpy as np

import tvm
import tvm.testing
from tvm import relay
import pytest

from test_verilator.infrastructure import (
    skip_test,
    compile_hardware,
    compiler_opts,
    run_module,
    offload,
    clear_stats,
    stats,
)


def create_module_add(shape, dtype):
    """Create add module.

    Paramters
    ---------
    shape : Tuple
        The shape tuple.

    dtype : Str
        The data type.

    Returns
    -------
    mod: Module
        The relay module.
    """
    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def create_module_bias_add(xshape, yshape, dtype):
    """Create bias_add module.

    Paramters
    ---------
    xshape : Tuple
        The x shape tuple.

    yshape : Tuple
        The y shape tuple.

    dtype : Str
        The data type.

    Returns
    -------
    mod: Module
        The relay module.
    """
    x = relay.var("x", shape=xshape, dtype=dtype)
    y = relay.var("y", shape=yshape, dtype=dtype)
    z = relay.nn.bias_add(x, y, axis=3)
    f = relay.Function([x, y], z)
    mod = tvm.IRModule()
    mod["main"] = f
    return mod


def run_and_check(xshape, yshape, dtype, mod, opts):
    """Run and check values.

    Paramters
    ---------
    xshape : Tuple
        The x shape tuple.

    yshape : Tuple
        The y shape tuple.

    dtype : Str
        The data type.

    mod: Module
        The relay module.

    opts: Dict
        The compiler options.

    Returns
    -------
    cycles: Int
        The number of cycles.
    """
    x_data = np.random.randint(5, size=xshape, dtype=dtype)
    y_data = np.random.randint(5, size=yshape, dtype=dtype)
    ref = x_data + y_data
    inp = {"x": x_data, "y": y_data}
    clear_stats()
    out = run_module(inp, mod, params=None, opts=opts)
    values = stats()
    tvm.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)
    return values["cycle_counter"]


def print_test_info(test, lanes, cycles):
    """Print counter

    Paramters
    ---------
    test : Str
        The name of the test.

    lanes : Int
        The number of vector lanes.

    cycles : Int
        The number of cycles.
    """
    print("test:{} vector-lanes:{} number of cycles:{}".format(test, lanes, cycles))


@pytest.mark.skipif(skip_test(), reason="Skip because Verilator codegen is not available")
def tadd(lanes):
    """Print counter

    Paramters
    ---------
    lanes : Int
        The number of vector lanes.
    """
    if skip_test():
        return
    dtype = "int32"
    shape = (8, 4)
    mod = create_module_add(shape, dtype)
    mod = offload(mod)
    lib = compile_hardware(lanes)
    opts = compiler_opts(lib)
    cycles = run_and_check(shape, shape, dtype, mod, opts)
    print_test_info("add", lanes, cycles)


@pytest.mark.skipif(skip_test(), reason="Skip because Verilator codegen is not available")
def tbias(lanes):
    """Print counter

    Paramters
    ---------
    lanes : Int
        The number of vector lanes.
    """
    if skip_test():
        return
    dtype = "int32"
    xshape = (1, 112, 112, 32)
    yshape = (32,)
    mod = create_module_bias_add(xshape, yshape, dtype)
    mod = offload(mod)
    lib = compile_hardware(lanes)
    opts = compiler_opts(lib)
    cycles = run_and_check(xshape, yshape, dtype, mod, opts)
    print_test_info("nn.bias_add", lanes, cycles)


def test_add():
    """add tests."""
    tadd(1)
    tadd(4)


def test_bias_add():
    """bias_add tests."""
    tbias(1)
    tbias(32)


if __name__ == "__main__":
    tvm.testing.main()
