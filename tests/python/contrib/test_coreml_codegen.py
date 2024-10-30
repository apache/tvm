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
import numpy as np
import pytest
from unittest import mock

import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib.target import coreml as _coreml

requires_coremltools = tvm.testing.requires_package("coremltools")


def _has_xcode():
    try:
        tvm.contrib.xcode.xcrun([])
        return True
    except FileNotFoundError:
        pass

    return False


def _create_graph():
    shape = (10, 10)
    mod = tvm.IRModule()

    x = relay.var("x", shape=shape)
    y = relay.var("y", shape=shape)
    z = x + x
    p = y * y
    func = relay.Function([x, y], p - z)
    mod["main"] = func

    return mod


def _create_graph_annotated():
    shape = (10, 10)
    target = "coremlcompiler"
    mod = tvm.IRModule()

    # function 0
    f0_i0 = relay.var(target + "_0_i0", shape=shape)
    func0 = relay.Function([f0_i0], f0_i0 * f0_i0)

    func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func0 = func0.with_attr("Compiler", target)
    func0 = func0.with_attr("global_symbol", target + "_0")
    gv0 = relay.GlobalVar(target + "_0")
    mod[gv0] = func0

    # function 2
    f2_i0 = relay.var(target + "_2_i0", shape=shape)
    func2 = relay.Function([f2_i0], f2_i0 + f2_i0)

    func2 = func2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func2 = func2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func2 = func2.with_attr("Compiler", target)
    func2 = func2.with_attr("global_symbol", target + "_2")
    gv2 = relay.GlobalVar(target + "_2")
    mod[gv2] = func2
    mod = relay.transform.InferType()(mod)

    # body
    x = relay.var("x", shape=shape)
    y = relay.var("y", shape=shape)
    func = relay.Function([x, y], gv0(y) - gv2(x))
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    return mod


@pytest.mark.xfail(
    reason="Currently failing test.  See tracking issue https://github.com/apache/tvm/issues/8901"
)
@tvm.testing.uses_gpu
@requires_coremltools
def test_annotate():
    mod = _create_graph()
    mod = transform.AnnotateTarget("coremlcompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    expected = _create_graph_annotated()
    tvm.ir.assert_structural_equal(mod, expected, map_free_vars=True)


@pytest.mark.skipif(not _has_xcode(), reason="Xcode is not available")
@tvm.testing.uses_gpu
@requires_coremltools
def test_compile_and_run():
    dev = tvm.cpu()
    target = "llvm"
    tol = 1e-3

    with relay.build_config(opt_level=3):
        lib = relay.build(_create_graph_annotated(), target=target)
    m = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    shape = (10, 10)
    x_data = np.random.rand(*shape).astype("float32")
    y_data = np.random.rand(*shape).astype("float32")

    m.set_input("x", x_data)
    m.set_input("y", y_data)
    m.run()
    out = tvm.nd.empty(shape, device=dev)
    out = m.get_output(0, out)

    expected = (y_data * y_data) - (x_data + x_data)
    tvm.testing.assert_allclose(out.numpy(), expected, rtol=tol, atol=tol)


@mock.patch("tvm.contrib.coreml_runtime.create")
@mock.patch("tvm.contrib.xcode.compile_coreml")
def _construct_model(func, m1, m2):
    mod = tvm.IRModule()
    mod["main"] = func
    mod = transform.AnnotateTarget("coremlcompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    fcompile = tvm._ffi.get_global_func("relay.ext.coremlcompiler")

    for var, func in mod.functions.items():
        if "Compiler" in func.attrs and func.attrs["Compiler"] == "coremlcompiler":
            fcompile(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_add():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = x + x
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_multiply():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = x * x
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_clip():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = relay.clip(x, a_min=0.0, a_max=1.0)
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_batch_flatten():
    shape = (10, 10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.batch_flatten(x)
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_expand_dims():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = relay.expand_dims(x, axis=0)
    func = relay.Function([x], y)
    _construct_model(func)

    y = relay.expand_dims(x, axis=-1)
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_relu():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.relu(x)
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_softmax():
    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.softmax(x, axis=1)
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_conv2d():
    x = relay.var("x", shape=(1, 3, 224, 224))
    w = relay.const(np.zeros((16, 3, 3, 3), dtype="float32"))
    y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
    func = relay.Function([x], y)
    _construct_model(func)


@tvm.testing.uses_gpu
@requires_coremltools
def test_global_avg_pool2d():
    shape = (10, 10, 10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.global_avg_pool2d(x)
    func = relay.Function([x], y)
    _construct_model(func)


if __name__ == "__main__":
    test_annotate()
    test_compile_and_run()
    test_add()
    test_multiply()
    test_clip()
    test_expand_dims()
    test_relu()
    test_batch_flatten()
    test_softmax()
    test_conv2d()
    test_global_avg_pool2d()
