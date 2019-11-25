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
import os

import numpy as np
import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.relay.testing import resnet

# Use the host emulated micro device.
DEV_CONFIG = micro.device.host.default_config()

def relay_micro_build(func, dev_config, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : Dict[str, Any]
        MicroTVM config dict for the target device

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device
    """
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target="c", params=params)
    micro_mod = create_micro_mod(c_mod, dev_config)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def test_alloc():
    """Test tensor allocation on the device."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"
    with micro.Session(DEV_CONFIG):
        ctx = tvm.micro_dev(0)
        np_tensor = np.random.uniform(size=shape).astype(dtype)
        micro_tensor = tvm.nd.array(np_tensor, ctx)
        tvm.testing.assert_allclose(np_tensor, micro_tensor.asnumpy())


def test_add():
    """Test a module which performs addition."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd"
    c_mod = tvm.build(s, [A, B, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG):
        micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, b, c)
        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    """Test a module which uses a workspace to compute an intermediate value."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd_two_workspace"
    c_mod = tvm.build(s, [A, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG):
        micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, c)

        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + 2.0)


def test_graph_runtime():
    """Test a program which uses the graph runtime."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(1.0))
    func = relay.Function([x], z)

    with micro.Session(DEV_CONFIG):
        mod = relay_micro_build(func, DEV_CONFIG)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        mod.run(x=x_in)
        result = mod.get_output(0).asnumpy()

        tvm.testing.assert_allclose(
                result, x_in * x_in + 1.0)


def test_multiple_modules():
    """Test loading multiple modules on the device simultaneously."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)
    # Construct Relay subtract program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.subtract(x, relay.const(1.0))
    sub_const_func = relay.Function([x], ret)

    with micro.Session(DEV_CONFIG):
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG)
        sub_const_mod = relay_micro_build(sub_const_func, DEV_CONFIG)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        add_const_mod.run(x=x_in)
        add_result = add_const_mod.get_output(0).asnumpy()
        sub_const_mod.run(x=x_in)
        sub_result = sub_const_mod.get_output(0).asnumpy()

        tvm.testing.assert_allclose(
                add_result, x_in + 1.0)
        tvm.testing.assert_allclose(
                sub_result, x_in - 1.0)


def test_interleave_sessions():
    """Test closing and reopening sessions."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG)
    sess_b = micro.Session(DEV_CONFIG)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
    with sess_b:
        np_tensor_b = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_b = tvm.nd.array(np_tensor_b, tvm.micro_dev(0))
    with sess_a:
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG)
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)
    with sess_b:
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG)
        add_const_mod.run(x=micro_tensor_b)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_b + 1.0)


def test_nested_sessions():
    """Test entering and exiting nested session contexts."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG)
    sess_b = micro.Session(DEV_CONFIG)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
        with sess_b:
            np_tensor_b = np.random.uniform(size=shape).astype(dtype)
            micro_tensor_b = tvm.nd.array(np_tensor_b, tvm.micro_dev(0))
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG)
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)


def test_inactive_session_use():
    """Test the use of objects allocated in a session that is no longer active."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG)
    sess_b = micro.Session(DEV_CONFIG)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG)

    with sess_b:
        # These objects belong to `sess_a`.
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)


if __name__ == "__main__":
    test_alloc()
    test_add()
    test_workspace_add()
    test_graph_runtime()
    test_multiple_modules()
    test_interleave_sessions()
    test_nested_sessions()
    test_inactive_session_use()
