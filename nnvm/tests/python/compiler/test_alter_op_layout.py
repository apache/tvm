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
"""Unittest cases for AlterOpLayout pass"""
from nnvm import symbol as sym
from nnvm.compiler import graph_attr
from nnvm.top import registry as reg
import nnvm.graph as graph

def get_layouts(g):
    ldict = {}
    vlayout = g.json_attr("layout")
    entry_ptr = g.index.entry_ptr
    for i, n in enumerate(g.index.nodes):
        begin, end = entry_ptr[i], entry_ptr[i + 1]
        ldict[n["name"]] = vlayout[begin:end]
    return ldict


def test_alter_conv2d_layout():
    data = sym.Variable("data", shape=(1, 32, 512, 512))
    conv = sym.conv2d(data, name="conv", channels=16,
                      kernel_size=(3,3), padding=(1,1),
                      use_bias=False, layout="NCHW")
    # split here
    convs = sym.split(conv, indices_or_sections=2)
    relus = [sym.relu(x, name="relu") for x in convs]
    relu = sym.concatenate(*relus)
    flatten = sym.flatten(relu, name="flatten")
    softmax = sym.softmax(flatten, name="softmax")
    g = graph.create(softmax)

    g = g.apply("CorrectLayout")
    g = graph_attr.set_dtype_inputs(g, "float32")
    g = g.apply(["InferShape", "InferType"])
    layouts_origin = get_layouts(g)

    @reg.register_alter_op_layout("conv2d", level=100)
    def alter_conv2d_layout(attrs, inputs, tinfos):
        new_attrs = {k : attrs[k] for k in attrs.keys()}
        new_attrs["layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "NCHW16c"
        new_attrs["name"] = "conv_alter"
        return sym.conv2d(inputs[0], inputs[1], **new_attrs)

    g = g.apply("AlterOpLayout")
    layouts = get_layouts(g)

    # check copy layouts
    for node in ["data", "relu", "flatten", "softmax", "conv_weight"]:
        assert layouts[node] == layouts_origin[node]
    assert layouts["conv_alter"] == layouts_origin["conv"]


def test_consecutive_alter_layout():
    data = sym.Variable("data", shape=(1, 32, 512, 512))
    pool1 = sym.global_avg_pool2d(data, name="global_avg_pool2d_1", layout="NCHW")
    pool2 = sym.global_avg_pool2d(pool1, name="global_avg_pool2d_2", layout="NCHW")
    relu = sym.relu(pool2, name="relu")

    g = graph.create(relu)
    g = g.apply("CorrectLayout")
    g = graph_attr.set_dtype_inputs(g, "float32")
    g = g.apply(["InferShape", "InferType"])
    assert g.json_attr("layout") == ['NCHW', 'NCHW', 'NCHW', 'NCHW']

    @reg.register_alter_op_layout("global_avg_pool2d", level=100)
    def alter_global_avg_pool2d_layout(attrs, inputs, tinfos):
        new_attrs = {k : attrs[k] for k in attrs.keys()}
        new_attrs["layout"] = "NCHW16c"
        return sym.global_avg_pool2d(inputs[0], **new_attrs)

    g = g.apply("AlterOpLayout")

    # pool1 get replaced - output layout of pool1 is not recorded
    # pool2 get replaced - input layout of pool2 is not recorded
    # thus the second entry must be undefined - it can neither recover from pool1's output,
    # nor from pool2's input.
    assert g.json_attr("layout") == ['NCHW', '__undef__', 'NCHW', 'NCHW']


def test_alter_func_return_none():
    data = sym.Variable("data", shape=(1, 32, 512, 512))
    pool1 = sym.global_max_pool2d(data, name="pool1", layout="NCHW")
    pool2 = sym.global_max_pool2d(pool1, name="pool2", layout="NCHW")
    relu = sym.relu(pool2, name="relu")

    g = graph.create(relu)
    g = g.apply("CorrectLayout")
    g = graph_attr.set_dtype_inputs(g, "float32")
    g = g.apply(["InferShape", "InferType"])
    assert g.json_attr("layout") == ['NCHW', 'NCHW', 'NCHW', 'NCHW']

    @reg.register_alter_op_layout("global_max_pool2d", level=100)
    def alter_global_max_pool2d_layout(attrs, inputs, tinfos):
        return None

    g = g.apply("AlterOpLayout")

    # alter func return none, nothing get replaced,
    # the layouts should remain the same
    assert g.json_attr("layout") == ['NCHW', 'NCHW', 'NCHW', 'NCHW']


if __name__ == "__main__":
    test_alter_conv2d_layout()
    test_consecutive_alter_layout()
    test_alter_func_return_none()
