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
import mxnet as mx
from tvm import relay
from tvm.relay import transform
import model_zoo
import numpy as np
from mxnet import gluon

def compare_graph(lhs_mod, rhs_mod):
    lhs_mod = transform.InferType()(lhs_mod)
    rhs_mod = transform.InferType()(rhs_mod)
    assert relay.analysis.alpha_equal(lhs_mod["main"], rhs_mod["main"])

def test_mlp():
    shape = {"data": (1, 1, 28, 28)}
    mx_fun = model_zoo.mx_mlp()
    mod, _ = relay.frontend.from_mxnet(mx_fun, shape=shape)
    relay_fun = model_zoo.relay_mlp()
    compare_graph(mod, relay_fun)


def test_vgg():
    shape = {"data": (1, 3, 224, 224)}
    for n in [11, 13, 16, 19]:
        mx_sym = model_zoo.mx_vgg(n)
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape=shape)
        relay_mod = model_zoo.relay_vgg(n)
        compare_graph(mod, relay_mod)


def test_resnet():
    shape = {"data": (1, 3, 224, 224)}
    for n in [18, 34, 50, 101]:
        mx_sym = model_zoo.mx_resnet(n)
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape=shape)
        relay_mod = model_zoo.relay_resnet(n)
        compare_graph(mod, relay_mod)


def test_squeezenet():
    shape = {"data": (1, 3, 224, 224)}
    for version in ['1.0', '1.1']:
        mx_sym = model_zoo.mx_squeezenet(version)
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape)
        relay_mod = model_zoo.relay_squeezenet(version)
        compare_graph(mod, relay_mod)


def test_inception_v3():
    shape = {"data": (1, 3, 299, 299)}
    mx_sym = model_zoo.mx_inception_v3()
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_mod = model_zoo.relay_inception_v3()
    compare_graph(mod, relay_mod)


def test_dqn():
    shape = {"data": (1, 4, 84, 84)}
    mx_sym = model_zoo.mx_dqn()
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_mod = model_zoo.relay_dqn()
    compare_graph(mod, relay_mod)


def test_dcgan():
    shape = {"data": (2, 100)}
    mx_sym = model_zoo.mx_dcgan()
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_mod = model_zoo.relay_dcgan(batch_size=2)
    compare_graph(mod, relay_mod)


def test_multi_outputs():
    xshape = (10, 27)
    yshape = (10, 9)

    def mx_compose(F, **kwargs):
        x = F.sym.Variable("x")
        y = F.sym.Variable("y")
        z = F.sym.split(x, **kwargs)
        return F.sym.broadcast_sub(F.sym.broadcast_add(z[0], z[2]), y)

    def relay_compose(F, **kwargs):
        x = F.var("x", shape=xshape)
        y = F.var("y", shape=yshape)
        z = F.split(x, **kwargs)
        z = F.subtract(F.add(z[0], z[2]), y)
        func = relay.Function(relay.analysis.free_vars(z), z)
        return relay.Module.from_expr(func)

    mx_sym = mx_compose(mx, num_outputs=3, axis=1)
    mod, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"x":xshape, "y":yshape})
    relay_mod = relay_compose(relay, indices_or_sections=3, axis=1)
    compare_graph(mod, relay_mod)

def to_list(mod, relay_list):
    py_list = []

    while mod.get_constructor(relay_list.tag).name_hint == 'cons':
        py_list.append(relay_list.fields[0].asnumpy())
        relay_list = relay_list.fields[1]

    return py_list

def test_while_loop():
    class Loop(gluon.HybridBlock):
        def hybrid_forward(self, F, data):
            def sum(state, i):
                s = state + F.take(data, i)
                return [], [s, i + 1]

            def sum_cond(state, i):
                return i < 4

            out, state = F.contrib.while_loop(
              sum_cond,
              sum,
              [F.zeros((1)), F.zeros((1))],
              max_iterations=5)
            return out, state

    data = mx.nd.arange(5)
    loop_layer = Loop()
    out, state = loop_layer(data)
    mod, _ = relay.frontend.from_mxnet(loop_layer, shape={'data': (5,)})

def test_foreach_map():
    def add1(data, _):
        return data + 1, []

    class Map(gluon.HybridBlock):
        def hybrid_forward(self, F, data):
            out, _ = F.contrib.foreach(add1, data, [])
            return out

    data = mx.nd.arange(5)
    map_layer = Map()
    mxnet_out = map_layer(data)
    mod, _ = relay.frontend.from_mxnet(map_layer, shape={'data': (5,)})
    ex = relay.create_executor('debug', mod=mod)
    relay_out = ex.evaluate()(data.asnumpy())
    relay_out = to_list(mod, relay_out)
    relay_out = np.array(relay_out)
    mxnet_out = mxnet_out.reshape((5,)).asnumpy()
    np.testing.assert_allclose(mxnet_out, relay_out)

def test_foreach_fold():
    def sum(data, state):
        return [], state + data

    class Scan(gluon.HybridBlock):
        def hybrid_forward(self, F, data):
            _, state = F.contrib.foreach(sum, data, F.zeros((1)))
            return state

    scan_layer = Scan()
    data = mx.nd.arange(5)
    mxnet_out = scan_layer(data)
    mod, _ = relay.frontend.from_mxnet(scan_layer, shape={'data': (5,)})
    ex = relay.create_executor('debug', mod=mod)
    relay_out = ex.evaluate()(data.asnumpy())
    relay_out = relay_out[0].asnumpy()
    mxnet_out = mxnet_out.asnumpy()
    np.testing.assert_allclose(mxnet_out, relay_out)

if __name__ == "__main__":
    # test_mlp()
    # test_resnet()
    # test_vgg()
    # test_multi_outputs()
    # test_dqn()
    # test_dcgan()
    # test_squeezenet()
    # test_inception_v3()
    # test_while_loop()
    test_foreach_map()
    test_foreach_fold()
