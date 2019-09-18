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

class MyAnnotator(ExprMutator):
    def visit_call(self, call):
        #print(call.op.name)
        if call.op.name == "add": # Annotate begin at args
            lhs = subgraph_begin(call.args[0], "gcc")
            rhs = subgraph_begin(call.args[1], "gcc")
            op = relay.add(lhs, rhs)
            return op
        elif call.op.name == "concatenate": # Annotate end at output
            op = super().visit_call(call)
            return subgraph_end(op, "gcc")
        return super().visit_call(call)

    def visit_function(self, func):
        return relay.Function(func.params, self.visit(func.body))

def annotate(expr):
    ann = MyAnnotator()
    return ann.visit(expr)

def test_partition_graph():
    x = relay.var('x', shape=(10, 10))
    z0 = relay.add(x, relay.const(0, dtype='float32'))
    z1 = relay.add(x, relay.const(5, dtype='float32'))
    z2 = relay.multiply(x, relay.const(2, dtype='float32'))
    p0 = relay.subtract(z0, relay.const(3, dtype='float32'))
    p1 = relay.subtract(z1, relay.const(4, dtype='float32'))
    p2 = relay.add(z2, relay.const(7, dtype='float32'))
    q = relay.concatenate((p0, p1, p2), axis=0)
    f = relay.Function([x], q)
    mod = relay.Module()
    mod["main"] = annotate(f)
    print(mod['main'])
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    print(mod['main'])
    #x_data = np.random.rand(10, 10).astype('float32')
    #y_data = np.random.rand(10, 10).astype('float32')
    # ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    # res = ex.evaluate()(x_data)
    # tvm.testing.assert_allclose(res.asnumpy(), y_data - (x_data + x_data))

def test_extern_gcc():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    z = x + x
    p = y * y
    f = relay.Function([x, y], p - z)
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')
    mod = relay.Module()
    mod["main"] = f
    mod = relay.transform.ExternOp("gcc")(mod)
    mod = relay.transform.PartitionGraph()(mod)
    print(mod['main'])
    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), (y_data * y_data) - (x_data + x_data))

def test_extern_cblas():
    m = 16
    n = 224
    k = 224
    dtype = 'float64'
    x = relay.var('x', shape=(m, k), dtype=dtype)
    y = relay.var('y', shape=(n, k), dtype=dtype)
    f = relay.Function([x, y], relay.op.nn.dense(x, y))
    mod = relay.Module()
    mod['main'] = f
    mod = relay.transform.ExternOp('cblas')(mod)
    mod = relay.transform.PartitionGraph()(mod)

    x_data = np.random.uniform(0, 1, (m, k)).astype(dtype)
    y_data = np.random.uniform(0, 1, (n, k)).astype(dtype)
    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(x_data, y_data)
    tvm.testing.assert_allclose(
        res.asnumpy(), np.dot(x_data, y_data.T), rtol=1e-5)

def test_extern_dnnl():
    dtype = 'float32'
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    w2shape = (100, 32 * 14 * 14)

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
    out1 = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)
    out2 = relay.nn.batch_flatten(data=out1)
    weight2 = relay.var('weight2', shape=(w2shape), dtype=dtype)
    out3 = relay.nn.dense(out2, weight2)

    f = relay.Function([data, weight1, weight2], out3)

    mod = relay.Module()
    mod['main'] = f
    mod = relay.transform.ExternOp('dnnl')(mod)
    mod = relay.transform.PartitionGraph()(mod)

    ref_mod = relay.Module()
    ref_mod['main'] = f

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    w2_data = np.random.uniform(0, 1, w2shape).astype(dtype)

    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(i_data, w1_data, w2_data)

    ref_ex = relay.create_executor("debug", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, w1_data, w2_data)

    tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)


def test_extern_dnnl_mobilenet():
    # FIXME: This test is only for demo purpose and supposed to be removed.
    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')

    mod = relay.transform.ExternOp('dnnl')(mod)
    mod = relay.transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(0))
    res = ex.evaluate()(i_data, **params)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')
    ref_ex = relay.create_executor("debug", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)


if __name__ == "__main__":
    test_partition_graph()
    test_extern_gcc()
    test_extern_cblas()
    test_extern_dnnl()
    #test_extern_dnnl_mobilenet()
