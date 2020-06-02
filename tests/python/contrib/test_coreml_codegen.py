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
from tvm import relay
from tvm.relay import transform
from tvm.contrib.target import coreml as _coreml

pytest.importorskip("coremltools")


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

    x = relay.var('x', shape=shape)
    y = relay.var('y', shape=shape)
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

    # body
    x = relay.var('x', shape=shape)
    y = relay.var('y', shape=shape)
    func = relay.Function([x, y], gv0(y) - gv2(x))
    mod["main"] = func

    return mod


def test_annotate():
    mod = _create_graph()
    mod = transform.AnnotateTarget("coremlcompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    expected = _create_graph_annotated()
    assert tvm.ir.structural_equal(mod, expected, map_free_vars=True)


@mock.patch('tvm.contrib.coreml_runtime.create')
@mock.patch('tvm.contrib.xcode.compile_coreml')
def test_construct_model(m1, m2):
    mod = _create_graph_annotated()

    fcompile = tvm._ffi.get_global_func("relay.ext.coremlcompiler")

    for var, func in mod.functions.items():
        if func.attrs and 'Compiler' in func.attrs and \
           func.attrs['Compiler'] == 'coremlcompiler':
            fcompile(tvm.IRModule.from_expr(func.body))


@pytest.mark.skipif(not _has_xcode(), reason="Xcode is not available")
def test_compile_and_run():
    ctx=tvm.cpu()
    target="llvm"
    tol=1e-3

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(_create_graph_annotated(), target=target)
    m = tvm.contrib.graph_runtime.create(json, lib, ctx)

    shape = (10, 10)
    x_data = np.random.rand(*shape).astype('float32')
    y_data = np.random.rand(*shape).astype('float32')

    m.set_input("x", x_data)
    m.set_input("y", y_data)
    m.set_input(**params)
    m.run()
    out = tvm.nd.empty(shape, ctx=ctx)
    out = m.get_output(0, out)

    expected = (y_data * y_data) - (x_data + x_data)
    tvm.testing.assert_allclose(out.asnumpy(), expected, rtol=tol, atol=tol)


if __name__ == "__main__":
    test_annotate()
    test_construct_model()
    test_compile_and_run()
