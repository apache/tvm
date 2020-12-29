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

import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
import model_zoo


def compare_graph(lhs_mod, rhs_mod):
    lhs_mod = transform.InferType()(lhs_mod)
    rhs_mod = transform.InferType()(rhs_mod)
    assert tvm.ir.structural_equal(lhs_mod["main"], rhs_mod["main"])


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
    for version in ["1.0", "1.1"]:
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
        return tvm.IRModule.from_expr(func)

    mx_sym = mx_compose(mx, num_outputs=3, axis=1)
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape={"x": xshape, "y": yshape})
    relay_mod = relay_compose(relay, indices_or_sections=3, axis=1)
    compare_graph(mod, relay_mod)


if __name__ == "__main__":
    test_mlp()
    test_resnet()
    test_vgg()
    test_multi_outputs()
    test_dqn()
    test_dcgan()
    test_squeezenet()
    test_inception_v3()
