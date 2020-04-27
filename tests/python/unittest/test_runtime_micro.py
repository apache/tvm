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
from tvm import te
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.relay.testing import resnet

# # Use the host emulated micro device.
DEV_CONFIG_A = micro.device.host.generate_config()
DEV_CONFIG_B = micro.device.host.generate_config()
TARGET = 'c -device=micro_dev'

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
    mod : tvm.runtime.Module
        graph runtime module for the target device
    """
    disable_vectorize = tvm.target.build_config(disable_vectorize=True)
    disable_fusion = relay.build_config(disabled_pass={'FuseOps'})
    with disable_vectorize, disable_fusion:
        graph, c_mod, params = relay.build(func, target=TARGET, params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


GDB_INIT_TEMPLATE = """
layout asm
target remote localhost:{gdb_port}
set $pc = UTVMInit
break UTVMDone
"""


def reset_gdbinit():
    if 'server_port' not in DEV_CONFIG_A:
        return
    gdb_init_dir = os.environ['MICRO_GDB_INIT_DIR']
    with open(f'{gdb_init_dir}/.gdbinit', 'w') as f:
        gdb_port = DEV_CONFIG_A['server_port'] - 3333
        f.write(GDB_INIT_TEMPLATE.format(gdb_port=gdb_port))


def test_alloc():
    """Test tensor allocation on the device."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"
    with micro.Session(DEV_CONFIG_A):
        ctx = tvm.micro_dev(0)
        np_tensor = np.random.uniform(size=shape).astype(dtype)
        micro_tensor = tvm.nd.array(np_tensor, ctx)
        tvm.testing.assert_allclose(np_tensor, micro_tensor.asnumpy())


def test_add():
    """Test a module which performs addition."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    reset_gdbinit()

    # Construct TVM expression.
    tvm_shape = tvm.runtime.convert(shape)
    A = te.placeholder(tvm_shape, name="A", dtype=dtype)
    B = te.placeholder(tvm_shape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)

    func_name = "fadd"
    c_mod = tvm.build(s, [A, B, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG_A) as sess:
        micro_mod = micro.create_micro_mod(c_mod, DEV_CONFIG_A)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)

        a_np = np.random.uniform(size=shape).astype(dtype)
        a = tvm.nd.array(a_np, ctx)
        b_np = np.random.uniform(size=shape).astype(dtype)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, b, c)

        # ensure inputs weren't corrupted
        tvm.testing.assert_allclose(
                a.asnumpy(), a_np)
        tvm.testing.assert_allclose(
                b.asnumpy(), b_np)
        # ensure output is correct
        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    """Test a module which uses a workspace to compute an intermediate value."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    reset_gdbinit()

    # Construct TVM expression.
    tvm_shape = tvm.runtime.convert(shape)
    A = te.placeholder(tvm_shape, name="A", dtype=dtype)
    B = te.placeholder(tvm_shape, name="B", dtype=dtype)
    B = te.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = te.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = te.create_schedule(C.op)

    func_name = "fadd_two_workspace"
    c_mod = tvm.build(s, [A, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG_A) as sess:
        micro_mod = micro.create_micro_mod(c_mod, DEV_CONFIG_A)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a_np = np.random.uniform(size=shape).astype(dtype)
        a = tvm.nd.array(a_np, ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        micro_func(a, c)

        # ensure input wasn't corrupted
        tvm.testing.assert_allclose(
                a.asnumpy(), a_np)
        # ensure output is correct
        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + 2.0)


def test_graph_runtime():
    """Test a program which uses the graph runtime."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(1.0))
    func = relay.Function([x], z)

    with micro.Session(DEV_CONFIG_A):
        mod = relay_micro_build(func, DEV_CONFIG_A)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        mod.run(x=x_in)
        result = mod.get_output(0).asnumpy()

        tvm.testing.assert_allclose(
                mod.get_input(0).asnumpy(), x_in)
        tvm.testing.assert_allclose(
                result, x_in * x_in + 1.0)


def test_conv2d():
    if not tvm.runtime.enabled("micro_dev"):
        return

    from tvm.relay import create_executor
    from tvm.relay import transform

    dshape = (1, 4, 16, 16)
    dtype = 'int8'
    func_name = 'fused_nn_conv2d'

    reset_gdbinit()

    # Construct Relay program.
    x = relay.var("x", shape=dshape, dtype=dtype)
    conv_expr = relay.nn.conv2d(
            x, relay.var("w"),
            kernel_size=(3, 3),
            padding=(1, 1),
            channels=4)
    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)

    x_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    out_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))

    with tvm.target.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(mod, target="c")

    with micro.Session(DEV_CONFIG_A):
        micro_mod = micro.create_micro_mod(c_mod, DEV_CONFIG_A)
        candidate_func_name = func_name
        for i in range(100):
            try:
                micro_func = micro_mod[candidate_func_name]
                break
            except tvm.TVMError as e:
                candidate_func_name = f'{func_name}_{i}'
        else:
            assert False
        ctx = tvm.micro_dev(0)

        x_data = tvm.nd.array(np.random.uniform(size=x_shape).astype(dtype), ctx)
        w_data = tvm.nd.array(np.random.uniform(size=w_shape).astype(dtype), ctx)
        result = tvm.nd.array(np.zeros(shape=out_shape, dtype=dtype), ctx)
        micro_func(x_data, w_data, result)

        out_data = np.zeros(out_shape, dtype=dtype)
        params = { 'x': x_data.asnumpy(), 'w': w_data.asnumpy() }
        intrp = create_executor('debug')
        expected_result = intrp.evaluate(mod['main'])(x_data, w_data)

        tvm.testing.assert_allclose(result.asnumpy(), expected_result.asnumpy())


def test_interleave_sessions():
    """Test closing and reopening sessions."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG_A)
    sess_b = micro.Session(DEV_CONFIG_B)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
    with sess_b:
        np_tensor_b = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_b = tvm.nd.array(np_tensor_b, tvm.micro_dev(0))
    with sess_a:
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG_A)
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)
    with sess_b:
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG_B)
        add_const_mod.run(x=micro_tensor_b)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_b + 1.0)


def test_nested_sessions():
    """Test entering and exiting nested session contexts."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG_A)
    sess_b = micro.Session(DEV_CONFIG_B)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
        with sess_b:
            np_tensor_b = np.random.uniform(size=shape).astype(dtype)
            micro_tensor_b = tvm.nd.array(np_tensor_b, tvm.micro_dev(0))
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG_A)
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)


def test_inactive_session_use():
    """Test the use of objects allocated in a session that is no longer active."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    # Construct Relay add program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    ret = relay.add(x, relay.const(1.0))
    add_const_func = relay.Function([x], ret)

    sess_a = micro.Session(DEV_CONFIG_A)
    sess_b = micro.Session(DEV_CONFIG_B)
    with sess_a:
        np_tensor_a = np.random.uniform(size=shape).astype(dtype)
        micro_tensor_a = tvm.nd.array(np_tensor_a, tvm.micro_dev(0))
        add_const_mod = relay_micro_build(add_const_func, DEV_CONFIG_A)

    with sess_b:
        # These objects belong to `sess_a`.
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)


# TODO add workspace alloc/free stress test

if __name__ == "__main__":
    test_alloc()
    print()
    print('finished alloc test')
    input('[press enter to continue]')
    test_add()
    print()
    print('finished add test')
    input('[press enter to continue]')
    test_workspace_add()
    print()
    print('finished workspace add test')
    input('[press enter to continue]')
    test_graph_runtime()
    print()
    print('finished graph runtime test')
    input('[press enter to continue]')
    test_conv2d()
    print()
    print('finished conv2d test')
    input('[press enter to continue]')
    test_multiple_modules()
    print()
    print('finished multiple modules test')
    input('[press enter to continue]')
    test_interleave_sessions()
    print()
    print('finished interleaved sessions test')
    input('[press enter to continue]')
    test_nested_sessions()
    print()
    print('finished nested sessions test')
    input('[press enter to continue]')
    test_inactive_session_use()
    print()
    print('finished use inactive session test')
    input('[press enter to continue]')
