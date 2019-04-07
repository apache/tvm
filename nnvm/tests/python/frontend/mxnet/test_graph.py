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
import nnvm
from nnvm.compiler import graph_util, graph_attr
import model_zoo

def compare_graph(sym1, sym2, ishape=(2, 3, 224, 224)):
    g1 = nnvm.graph.create(sym1)
    g2 = nnvm.graph.create(sym2)
    graph_attr.set_shape_inputs(g1, {'data':ishape})
    graph_attr.set_shape_inputs(g2, {'data':ishape})
    g1 = g1.apply("InferShape").apply("SimplifyInference")
    g2 = g2.apply("InferShape").apply("SimplifyInference")
    graph_util.check_graph_equal(g1, g2)

def test_mlp():
    mx_sym = model_zoo.mx_mlp
    from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym = model_zoo.nnvm_mlp
    compare_graph(from_mx_sym, nnvm_sym)

def test_vgg():
    for n in [11, 13, 16, 19]:
        mx_sym = model_zoo.mx_vgg[n]
        from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
        nnvm_sym = model_zoo.nnvm_vgg[n]
        compare_graph(from_mx_sym, nnvm_sym)

def test_resnet():
    for n in [18, 34, 50, 101]:
        mx_sym = model_zoo.mx_resnet[n]
        from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
        nnvm_sym = model_zoo.nnvm_resnet[n]
        compare_graph(from_mx_sym, nnvm_sym)

def test_squeezenet():
    for version in ['1.0', '1.1']:
        mx_sym = model_zoo.mx_squeezenet[version]
        from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
        nnvm_sym = model_zoo.nnvm_squeezenet[version]
        compare_graph(from_mx_sym, nnvm_sym)

def test_inception_v3():
    mx_sym = model_zoo.mx_inception_v3
    from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym = model_zoo.nnvm_inception_v3
    compare_graph(from_mx_sym, nnvm_sym, ishape=(2, 3, 299, 299))

def test_dqn():
    mx_sym = model_zoo.mx_dqn
    from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym = model_zoo.nnvm_dqn
    compare_graph(from_mx_sym, nnvm_sym, ishape=(2, 4, 84, 84))

def test_dcgan():
    mx_sym = model_zoo.mx_dcgan
    from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym = model_zoo.nnvm_dcgan
    compare_graph(from_mx_sym, nnvm_sym, ishape=(2, 100))

def test_multi_outputs():
    def compose(F, **kwargs):
        x = F.sym.Variable('x')
        y = F.sym.Variable('y')
        z = F.sym.split(x, **kwargs)
        return F.sym.broadcast_sub(F.sym.broadcast_add(z[0], z[2]), y)
    mx_sym = compose(mx, num_outputs=3, axis=1)
    from_mx_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym = compose(nnvm, indices_or_sections=3, axis=1)
    compare_graph(from_mx_sym, nnvm_sym)

if __name__ == '__main__':
    test_mlp()
    test_vgg()
    test_resnet()
    test_multi_outputs()
    test_dqn()
    test_dcgan()
    test_squeezenet()
    test_inception_v3()
