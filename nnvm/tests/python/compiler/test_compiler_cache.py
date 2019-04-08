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

def test_compile_cache():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 1)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    def verify(graph, lib):
        m = graph_runtime.create(graph, lib, tvm.cpu(0))
        # get member functions
        na = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
        nb = tvm.nd.array(np.random.uniform(size=shape).astype(dtype))
        m.run(x=na, y=nb)
        # get outputs
        out = m.get_output(0, tvm.nd.empty(shape, dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))

    engine = nnvm.compiler.engine
    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    inputs = [tvm.placeholder((10,)), tvm.placeholder((10,))]

    gkey = nnvm.compiler.graph_key(nnvm.graph.create(z), inputs, "llvm")
    gkey2 = nnvm.compiler.graph_key(nnvm.graph.create(z), inputs + inputs, "llvm")
    gf = engine[gkey]
    assert gf is not None
    assert engine[gkey2] is None
    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    assert graph.index.num_nodes == 3
    verify(graph, lib)
    # Test various set external cache
    engine.clear_cache()
    engine[gkey] = gf

if __name__ == "__main__":
    test_compile_cache()
