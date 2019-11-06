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
from tvm.relay.testing import resnet

# Use the host emulated micro device.
DEV_CONFIG = micro.device.host.default_config()
#DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)

#def create_micro_mod(c_mod, dev_config):
#    """Produces a micro module from a given module.
#
#    Parameters
#    ----------
#    c_mod : tvm.module.Module
#        module with "c" as its target backend
#
#    toolchain_prefix : str
#        toolchain prefix to be used (see `tvm.micro.Session` docs)
#
#    Return
#    ------
#    micro_mod : tvm.module.Module
#        micro module for the target device
#    """
#    print('[create_micro_mod]')
#    temp_dir = util.tempdir()
#    lib_obj_path = temp_dir.relpath("dev_lib.obj")
#    c_mod.export_library(
#            lib_obj_path,
#            fcompile=tvm.micro.cross_compiler(dev_config['binutil'], micro.LibType.OPERATOR))
#    micro_mod = tvm.module.load(lib_obj_path)
#    return micro_mod


def relay_micro_build(func, sess, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device
    """
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target="c", params=params)
    micro_mod = sess.create_micro_mod(c_mod)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def reset_gdbinit():
    with open('/home/pratyush/Code/nucleo-interaction-from-scratch/.gdbinit', 'w') as f:
        gdbinit_contents = (
"""layout asm
target remote localhost:3333

#print "(*((TVMValue*) utvm_task.arg_values)).v_handle"
#print (*((TVMValue*) utvm_task.arg_values)).v_handle
#print "*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))"
#print *((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))
#print "((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[0]"
#print ((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[0]
#print "((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[1]"
#print ((float*) (*((TVMArray*) ((*((TVMValue*) utvm_task.arg_values)).v_handle))).data)[1]
#break UTVMMain
#break UTVMDone
#jump UTVMMain""")
        f.write(gdbinit_contents)


# TODO(weberlo): Add example program to test scalar double/int TVMValue serialization.
# TODO(weberlo): How can we test the OpenOCD device?  The CI would need to have OpenOCD
# and Spike installed.

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

    reset_gdbinit()

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd"
    c_mod = tvm.build(s, [A, B, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG) as sess:
        micro_mod = sess.create_micro_mod(c_mod)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        print(a)
        print(b)
        print(c)
        micro_func(a, b, c)
        print(c)
        tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_workspace_add():
    """Test a module which uses a workspace to compute an intermediate value."""
    if not tvm.module.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    reset_gdbinit()

    # Construct TVM expression.
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A", dtype=dtype)
    B = tvm.placeholder(tvm_shape, name="B", dtype=dtype)
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name="C")
    s = tvm.create_schedule(C.op)

    func_name = "fadd_two_workspace"
    c_mod = tvm.build(s, [A, C], target="c", name=func_name)

    with micro.Session(DEV_CONFIG) as sess:
        micro_mod = sess.create_micro_mod(c_mod)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)
        a = tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx)
        print(a)
        c = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
        print(c)
        micro_func(a, c)
        print(c)

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

    with micro.Session(DEV_CONFIG) as sess:
        mod = relay_micro_build(func, sess)

        x_in = np.random.uniform(size=shape[0]).astype(dtype)
        print(x_in)
        mod.run(x=x_in)
        result = mod.get_output(0).asnumpy()
        print(result)

        tvm.testing.assert_allclose(
                result, x_in * x_in + 1.0)


def test_conv2d():
    if not tvm.module.enabled("micro_dev"):
        return

    from tvm.relay import create_executor
    from tvm.relay import transform

    dshape = (1, 4, 16, 16)
    dtype = 'float32'
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
    mod = relay.Module.from_expr(func)
    mod = transform.InferType()(mod)

    x_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    out_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))

    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(mod, target="c")

    with micro.Session(DEV_CONFIG) as sess:
        micro_mod = sess.create_micro_mod(c_mod)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)

        x_data = tvm.nd.array(np.random.uniform(size=x_shape).astype(dtype), ctx)
        w_data = tvm.nd.array(np.random.uniform(size=w_shape).astype(dtype), ctx)
        result = tvm.nd.array(np.zeros(shape=out_shape, dtype=dtype), ctx)
        micro_func(x_data, w_data, result)

        out_data = np.zeros(out_shape, dtype=dtype)
        params = { 'x': x_data.asnumpy(), 'w': w_data.asnumpy() }
        intrp = create_executor('debug')
        expected_result = intrp.evaluate(mod['main'])(x_data, w_data).data

        tvm.testing.assert_allclose(result.asnumpy(), expected_result.asnumpy())


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

    with micro.Session(DEV_CONFIG) as sess:
        add_const_mod = relay_micro_build(add_const_func, sess)
        sub_const_mod = relay_micro_build(sub_const_func, sess)

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
        add_const_mod = relay_micro_build(add_const_func, sess_a)
        add_const_mod.run(x=micro_tensor_a)
        add_result = add_const_mod.get_output(0).asnumpy()
        tvm.testing.assert_allclose(
                add_result, np_tensor_a + 1.0)
    with sess_b:
        add_const_mod = relay_micro_build(add_const_func, sess_b)
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
        add_const_mod = relay_micro_build(add_const_func, sess_a)
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
        add_const_mod = relay_micro_build(add_const_func, sess_a)

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
    test_conv2d()
    test_multiple_modules()
    test_interleave_sessions()
    test_nested_sessions()
    test_inactive_session_use()
