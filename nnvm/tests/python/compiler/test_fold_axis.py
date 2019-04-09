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
"""Unittest cases for fold_axis"""
import tvm
import nnvm
import nnvm.testing.resnet
import numpy as np
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_fold_axis_conv():
    # Before simplify
    def before(x, conv_weight, conv_bias, in_scale, out_scale, channels):
        x = x * sym.expand_dims(in_scale, axis=1, num_newaxis=2)
        y = sym.conv2d(x, conv_weight, conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       name="conv")
        y = sym.relu(y)
        y = y * sym.expand_dims(out_scale, axis=1, num_newaxis=2)
        return y

    def expected(x, conv_weight, conv_bias, in_scale, out_scale, channels):
        conv_weight = conv_weight * sym.expand_dims(out_scale, axis=1, num_newaxis=3)
        conv_weight = conv_weight * sym.expand_dims(in_scale, axis=1, num_newaxis=2)
        conv_bias = conv_bias * out_scale
        y = sym.conv2d(x,
                       conv_weight,
                       conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       name="conv")
        y = sym.relu(y)
        return y

    def check(shape, channels):
        x = sym.Variable("x") + 1
        weight = sym.Variable("weight")
        bias = sym.Variable("bias")
        in_scale = sym.Variable("in_scale")
        out_scale = sym.Variable("out_scale")
        y1 = before(x, weight, bias, in_scale, out_scale, channels)
        y2 = expected(x, weight, bias, in_scale, out_scale, channels)
        ishape = {"x": shape, "out_scale": (channels,), "in_scale": (shape[1],)}
        g1 = nnvm.graph.create(y1)
        g2 = nnvm.graph.create(y2)
        graph_attr.set_shape_inputs(g1, ishape)
        g1 = g1.apply("InferShape").apply("FoldScaleAxis")
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check((2, 4, 10, 10), 2)

def test_fold_axis_depthwise_conv():
    # Before simplify
    def before(x, conv_weight, conv_bias, in_scale, out_scale, channels):
        x = x * sym.expand_dims(in_scale, axis=1, num_newaxis=2)
        y = sym.conv2d(x, conv_weight, conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       groups=54,
                       name="depthiwise_conv")
        y = sym.relu(y)
        y = y * sym.expand_dims(out_scale, axis=1, num_newaxis=2)
        return y

    def expected(x, conv_weight, conv_bias, in_scale, out_scale, channels):
        conv_weight = conv_weight * sym.expand_dims(out_scale, axis=1, num_newaxis=3)
        conv_weight = conv_weight * sym.expand_dims(in_scale, axis=1, num_newaxis=3)
        conv_bias = conv_bias * out_scale
        y = sym.conv2d(x,
                       conv_weight,
                       conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       groups=54,
                       name="depthiwise_conv")
        y = sym.relu(y)
        return y

    def check(shape, channels):
        x = sym.Variable("x") + 1
        weight = sym.Variable("weight")
        bias = sym.Variable("bias")
        in_scale = sym.Variable("in_scale")
        out_scale = sym.Variable("out_scale")
        y1 = before(x, weight, bias, in_scale, out_scale, channels)
        y2 = expected(x, weight, bias, in_scale, out_scale, channels)
        ishape = {"x": shape, "out_scale": (channels,), "in_scale": (shape[1],)}
        g1 = nnvm.graph.create(y1)
        g2 = nnvm.graph.create(y2)
        graph_attr.set_shape_inputs(g1, ishape)
        g1 = g1.apply("InferShape").apply("FoldScaleAxis")
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check((1, 54, 63, 127), 54)

def test_fold_fail():
    # Before simplify
    def before(x, scale, channels):
        y = sym.conv2d(x,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       name="conv")
        y = y * sym.expand_dims(scale, axis=1, num_newaxis=1)
        return y

    def check(shape, channels):
        x = sym.Variable("x")
        bias = sym.Variable("bias")
        scale = sym.Variable("scale")
        y1 = before(x, scale, channels)
        ishape = {"x": shape, "scale": (channels,), "bias": (channels,)}
        g1 = nnvm.graph.create(y1)
        graph_attr.set_shape_inputs(g1, ishape)
        g2 = g1.apply("InferShape").apply("FoldScaleAxis")
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check((2, 10, 10, 10), 10)


def test_fold_resnet():
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) +image_shape
    net, params = nnvm.testing.resnet.get_workload(
        batch_size=1, image_shape=image_shape)
    ishape = {"data" : data_shape}
    graph = nnvm.graph.create(net)
    data = np.random.uniform(size=data_shape).astype("float32")
    # Initial pass do shape type inference
    shape, _ = graph_util.infer_shape(graph, **ishape)
    ishape.update(zip(graph.index.input_names, shape))

    def run_prune(graph, params, opt_level):
        # Apply optimization
        with nnvm.compiler.build_config(opt_level=0):
            graph = nnvm.compiler.optimize(graph, ishape)
        graph, params = nnvm.compiler.build_module.precompute_prune(graph, params)
        params["data"] = data
        return nnvm.compiler.build_module._run_graph(graph, params)

    x = run_prune(graph, params, 0)
    y = run_prune(graph, params, 3)
    tvm.testing.assert_allclose(y[0].asnumpy(), x[0].asnumpy())


if __name__ == "__main__":
    test_fold_resnet()
    test_fold_axis_conv()
    test_fold_fail()
    test_fold_axis_depthwise_conv()
