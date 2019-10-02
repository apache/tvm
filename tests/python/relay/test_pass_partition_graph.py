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
"""Unit tests for graph partitioning."""
import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.transform
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.annotation import subgraph_begin, subgraph_end

class GCCAnnotator(ExprMutator):
    """
    A simple annotator that creates the following subgraph:
           |
      -- begin --
           |
          add
           |
        subtract
           |
        multiply
           |
       -- end --
           |
    """
    def __init__(self):
        super(GCCAnnotator, self).__init__()
        self.in_subgraph = 0

    def visit_call(self, call):
        if call.op.name == "add": # Annotate begin at args
            if self.in_subgraph == 1:
                lhs = subgraph_begin(super().visit(call.args[0]), "gcc")
                rhs = subgraph_begin(super().visit(call.args[1]), "gcc")
                op = relay.add(lhs, rhs)
                self.in_subgraph = 2
                return op
        elif call.op.name == "subtract":
            if self.in_subgraph == 1:
                lhs = super().visit(call.args[0])
                rhs = super().visit(call.args[1])
                if isinstance(lhs, relay.expr.Var):
                    lhs = subgraph_begin(lhs, "gcc")
                if isinstance(rhs, relay.expr.Var):
                    rhs = subgraph_begin(rhs, "gcc")
                return relay.subtract(lhs, rhs)
        elif call.op.name == "multiply": # Annotate end at output
            self.in_subgraph = 1
            lhs = super().visit(call.args[0])
            rhs = super().visit(call.args[1])
            if isinstance(lhs, relay.expr.Var):
                lhs = subgraph_begin(lhs, "gcc")
            if isinstance(rhs, relay.expr.Var):
                rhs = subgraph_begin(rhs, "gcc")
            op = relay.multiply(lhs, rhs)
            if self.in_subgraph == 2:
                op = subgraph_end(op, "gcc")
            self.in_subgraph = 0
            return op
        return super().visit_call(call)


class WholeGraphAnnotator(ExprMutator):
    """
    An annotator that creates a subgraph for an entire graph.
    """

    def __init__(self, compiler):
        super(WholeGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = subgraph_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = subgraph_end(new_call, self.compiler)
        return new_call


class MobileNetAnnotator(ExprMutator):
    """
    Annotate mobilenet until global_avg_pool.
    """

    def __init__(self, compiler):
        super(MobileNetAnnotator, self).__init__()
        self.compiler = compiler
        self.subgraph_open = False

    def visit_call(self, call):

        if call.op.name == 'nn.global_avg_pool2d':
            self.subgraph_open = True
        subgraph_open = self.subgraph_open

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if call.op.name == 'nn.global_avg_pool2d':
                param = subgraph_end(param, self.compiler)
            if subgraph_open and isinstance(param, relay.expr.Var):
                param = subgraph_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        return new_call


def test_multi_node_subgraph():
    x = relay.var('x', shape=(10, 10))
    w0 = relay.var('w', shape=(10, 10))
    w1 = relay.var('w', shape=(10, 10))
    w2 = relay.var('w', shape=(10, 10))
    w3 = relay.var('w', shape=(10, 10))
    w4 = relay.var('w', shape=(10, 10))
    w5 = relay.var('w', shape=(10, 10))
    w6 = relay.var('w', shape=(10, 10))
    w7 = relay.var('w', shape=(10, 10))

    # Subgraph on GCC
    # FIXME: We generate two subgraphs for this case but they should be merged to one
    # due to the common input (x).
    z0 = relay.add(x, w0)
    p0 = relay.subtract(z0, w1)
    q0 = relay.multiply(p0, w2)

    z1 = relay.add(x, w3)
    p1 = relay.subtract(z1, w4)
    q1 = relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((q0, q1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = relay.Module()
    ann = GCCAnnotator()
    mod["main"] = ann.visit(f)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype('float32')
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype('float32'))

    for kind in ["debug", "vm"]:
        ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
        res = ex.evaluate()(x_data, *w_data)
        tvm.testing.assert_allclose(
            res.asnumpy(),
            np.concatenate(
                (((x_data + w_data[0]) - w_data[1]) * w_data[2],
                 ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                 x_data + w_data[6] - w_data[7]),
                axis=0))


def test_extern_gcc_single_op():
    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))
    z = x + y
    f = relay.Function([x, y], z)
    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    mod = relay.Module()
    mod["main"] = f
    mod = relay.transform.ExternOp("gcc")(mod)
    mod = relay.transform.PartitionGraph()(mod)

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        res = ex.evaluate()(x_data, y_data)
        tvm.testing.assert_allclose(res.asnumpy(), (x_data + y_data))


def test_extern_gcc():
    x = relay.var('x', shape=(2, 2))
    y = relay.var('y', shape=(2, 2))
    z = x + x
    p = y * y
    f = relay.Function([x, y], p - z)
    x_data = np.random.rand(2, 2).astype('float32')
    y_data = np.random.rand(2, 2).astype('float32')
    mod = relay.Module()
    mod["main"] = f
    mod = relay.transform.ExternOp("gcc")(mod)
    mod = relay.transform.PartitionGraph()(mod)

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        res = ex.evaluate()(x_data, y_data)
        tvm.testing.assert_allclose(res.asnumpy(),
                                    (y_data * y_data) - (x_data + x_data))


def test_extern_dnnl():
    dtype = 'float32'
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    data = relay.var('data', shape=(ishape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1shape), dtype=dtype)
    depthwise_conv2d_1 = relay.nn.conv2d(data,
                                         weight1,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         groups=32)
    depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                         weight1,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         groups=32)
    out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

    f = relay.Function([data, weight1], out)

    mod = relay.Module()
    mod['main'] = WholeGraphAnnotator('dnnl').visit(f)
    mod = relay.transform.PartitionGraph()(mod)

    ref_mod = relay.Module()
    ref_mod['main'] = f

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu())
        res = ex.evaluate()(i_data, w1_data)

        ref_ex = relay.create_executor(kind, mod=ref_mod, ctx=tvm.cpu(0))
        ref_res = ref_ex.evaluate()(i_data, w1_data)

        tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)


def test_extern_dnnl_mobilenet():
    # FIXME: This test is only for demo purpose and supposed to be removed.
    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')

    #mod['main'] = MobileNetAnnotator('dnnl').visit(mod['main'])
    mod = relay.transform.ExternOp('dnnl')(mod)
    mod = relay.transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(0))
        res = ex.evaluate()(i_data, **params)

    # FIXME: When subgraph has only one op, Relay executor will use the cache value instead
    # of re-computing, so the following checking logic does not work.
    #ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')
    #ref_ex = relay.create_executor("debug", mod=ref_mod, ctx=tvm.cpu(0))
    #ref_res = ref_ex.evaluate()(i_data, **params)

    #tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)


if __name__ == "__main__":
    test_multi_node_subgraph()
    test_extern_gcc_single_op()
    test_extern_gcc()
    test_extern_dnnl()
    #test_extern_dnnl_mobilenet()
