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

import tvm
from tvm.contrib import graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.compiler.build_module import _run_graph, precompute_prune

def test_compile():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    def verify(graph, lib):
        m = graph_runtime.create(graph, lib, tvm.cpu(0))
        # get member functions
        set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
        na = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
        nb = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
        # set inputs
        set_input("x", na)
        set_input("y", nb)
        # execute
        run()
        # get outputs
        out = tvm.nd.empty(shape, dtype)
        get_output(0, out)
        tvm.testing.assert_allclose(
            out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))

    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    assert graph.index.num_nodes == 3
    verify(graph, lib)

    with nnvm.compiler.build_config(opt_level=0):
        graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
        # print(graph.ir())
        assert graph.index.num_nodes == 4
        verify(graph, lib)

def test_run():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    res = _run_graph(z, {"x": nx, "y": ny})
    tvm.testing.assert_allclose(
        res[0].asnumpy(), np.exp(nx.asnumpy() + ny.asnumpy()))


def test_precompute_prune():
    x = sym.Variable("x") + 1
    a = sym.Variable("a")
    y = sym.Variable("y")
    z = y + x + a
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    na = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx, "a": na}
    graph, lib, params = nnvm.compiler.build(
        z, "llvm", shape={"y": ny.shape}, params=params)
    assert graph.index.num_nodes == 4
    m = graph_runtime.create(graph, lib, tvm.cpu(0))
    params["y"] = ny
    res = tvm.nd.empty(shape)
    m["load_params"](nnvm.compiler.save_param_dict(params))
    m.run()
    out = m.get_output(0, out=res)
    tvm.testing.assert_allclose(
        res.asnumpy(), nx.asnumpy() + 1 + ny.asnumpy() + na.asnumpy())


def test_dtypes():
    x = sym.Variable("x")
    y = sym.relu(x)
    dshape = (1, 3, 32, 32)
    oshape = dshape
    for dtype in ['float32', 'float64', 'int32', 'int16', 'int8', 'int64']:
        graph, lib, _ = nnvm.compiler.build(y, 'llvm', {"x": dshape}, dtype=dtype)
        m = graph_runtime.create(graph, lib, tvm.cpu())
        if 'float' in dtype:
          data = np.random.uniform(size=dshape).astype(dtype)
        elif 'int' in dtype:
          data = np.random.randint(-127, 127, dshape).astype(dtype)
        m.run(x=data)
        data = (data > 0) * data
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        tvm.testing.assert_allclose(out.asnumpy(), data, atol=1e-5, rtol=1e-5)

def test_ndarray_output():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = x + y
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx, "ny": ny}
    graph, lib, params = nnvm.compiler.build(
        z, "llvm", shape={"y": ny.shape, "x": nx.shape}, params=params)
    m = graph_runtime.create(graph, lib, tvm.cpu(0))
    m.set_input("x", nx)
    m.set_input("y", ny)
    m.run()
    out = m.get_output(0)
    tvm.testing.assert_allclose(
        out.asnumpy(), nx.asnumpy() + ny.asnumpy())

def test_ndarray_input():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = x + y
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    ny = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx, "ny": ny}
    graph, lib, params = nnvm.compiler.build(
        z, "llvm", shape={"y": ny.shape, "x": nx.shape}, params=params)
    m = graph_runtime.create(graph, lib, tvm.cpu(0))
    m.set_input("x", nx)
    m.set_input("y", ny)
    in_x = tvm.nd.empty(shape, dtype)
    in_y = tvm.nd.empty(shape, dtype)
    m.get_input("x", in_x)
    m.get_input("y", in_y)
    tvm.testing.assert_allclose(nx.asnumpy(), in_x.asnumpy())
    tvm.testing.assert_allclose(ny.asnumpy(), in_y.asnumpy())
    in_nx = m.get_input("x")
    in_ny = m.get_input("y")
    tvm.testing.assert_allclose(nx.asnumpy(), in_nx.asnumpy())
    tvm.testing.assert_allclose(ny.asnumpy(), in_ny.asnumpy())

def test_num_outputs():
    x = sym.Variable('x')
    z = sym.split(x, indices_or_sections=5, axis=1)
    shape = (10, 10)
    dtype = tvm.float32
    nx = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
    params = {"x": nx}
    graph, lib, params = nnvm.compiler.build(
        z, "llvm", shape={"x": nx.shape}, params=params)
    m = graph_runtime.create(graph, lib, tvm.cpu(0))
    assert m.get_num_outputs() == 5

if __name__ == "__main__":
    test_precompute_prune()
    test_compile()
    test_run()
    test_dtypes()
    test_ndarray_output()
    test_ndarray_input()
    test_num_outputs()
