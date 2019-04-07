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
import nnvm
import nnvm.symbol as sym
import nnvm.graph as graph
from nnvm.compiler import graph_attr

def correct_layout(g, layout=None):
    if isinstance(g, nnvm.symbol.Symbol):
        g = graph.create(g)
    if layout:
        graph_attr.set_layout_inputs(g, layout)
    g = g.apply("CorrectLayout")
    ldict = {}
    vlayout = g.json_attr("layout")
    entry_ptr = g.index.entry_ptr
    for i, n in enumerate(g.index.nodes):
        begin, end = entry_ptr[i], entry_ptr[i + 1]
        ldict[n["name"]] = vlayout[begin:end]
    return g, ldict


# Level 1
def test_dense():
    x = sym.Variable("data", shape=(10, 20))
    y = sym.dense(x, units=30, name="fc")
    g, ldict = correct_layout(y, "HW")
    assert(ldict["data"][0] == "HW")
    assert(ldict["fc"][0] == "HW")
    assert(ldict["fc_bias"][0] == "__undef__")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, "HW16w")
    assert(ldict["data"][0] == "HW16w")
    assert(ldict["data_HW"][0] == "HW")
    assert(ldict["fc"][0] == "HW")
    assert(ldict["fc_bias"][0] == "__undef__")


def test_matmul():
    a = sym.Variable("a", shape=(10, 20))
    b = sym.Variable("b", shape=(20, 30))
    c = sym.matmul(a, b, name="matmul")
    g, ldict = correct_layout(c, {"a" : "HW", "b" : "WC"})
    assert(ldict["a"][0] == "HW")
    assert(ldict["b"][0] == "WC")
    assert(ldict["matmul"][0] == "HC")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, {"a" : "HW16w", "b" : "WC16c"})
    assert(ldict["a"][0] == "HW16w")
    assert(ldict["a_HW"][0] == "HW")
    assert(ldict["b"][0] == "WC16c")
    assert(ldict["b_WC"][0] == "WC")
    assert(ldict["matmul"][0] == "HC")
    a = sym.Variable("a", shape=(20, 10))
    c = sym.matmul(a, b, name="matmul", transpose_a=True)
    g, ldict = correct_layout(c, {"a" : "HW", "b" : "HC"})
    assert(ldict["a"][0] == "HW")
    assert(ldict["b"][0] == "HC")
    assert(ldict["matmul"][0] == "WC")
    b = sym.Variable("b", shape=(30, 20))
    c = sym.matmul(a, b, name="matmul", transpose_b=True)
    g, ldict = correct_layout(c, {"a" : "HW", "b" : "CW"})
    assert(ldict["a"][0] == "HW")
    assert(ldict["b"][0] == "CW")
    assert(ldict["matmul"][0] == "HC")
    a = sym.Variable("a", shape=(20, 10))
    b = sym.Variable("b", shape=(30, 20))
    c = sym.matmul(a, b, name="matmul", transpose_a=True, transpose_b=True)
    g, ldict = correct_layout(c, {"a" : "HW", "b" : "CH"})
    assert(ldict["a"][0] == "HW")
    assert(ldict["b"][0] == "CH")
    assert(ldict["matmul"][0] == "WC")


def test_concatenate():
    x1 = sym.Variable("x", shape=(10, 20))
    x2 = sym.Variable("y", shape=(10, 30))
    z = sym.concatenate(x1, x2, name="concat")
    g, ldict = correct_layout(z, {"x": "HW", "y": "HW"})
    assert(ldict["x"][0] == "HW")
    assert(ldict["y"][0] == "HW")
    assert(ldict["concat"][0] == "HW")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, {"x": "HW16w", "y": "HW16w"})
    assert(ldict["x"][0] == "HW16w")
    assert(ldict["y"][0] == "HW16w")
    assert(ldict["concat"][0] == "HW16w")

    x1 = sym.Variable("x", shape=(10, 20, 60))
    x2 = sym.Variable("y", shape=(10, 20, 40))
    z = sym.concatenate(x1, x2, axis=2, name="concat")
    g, ldict = correct_layout(z, {"x": "H20wW", "y": "H20wW"})
    assert(ldict["x"][0] == "H20wW")
    assert(ldict["y"][0] == "H20wW")
    assert(ldict["concat"][0] == "H20wW")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, {"x": "HW", "y": "HW"})
    assert(ldict["x_H20wW"][0] == "H20wW")
    assert(ldict["x_H20wW"][0] == "H20wW")
    assert(ldict["concat"][0] == "H20wW")


def test_expand_dims():
    x = sym.Variable("x", shape=(10, 20))
    y = sym.expand_dims(x, axis=1, name="y")
    g, ldict = correct_layout(y, "HW")
    assert(ldict["x"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, "HW16w")
    assert(ldict["x"][0] == "HW16w")
    assert(ldict["x_HW"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")


def test_split():
    x = sym.Variable("x", shape=(10, 20))
    y = sym.split(x, indices_or_sections=[11], name="y")
    g, ldict = correct_layout(y, "HW")
    assert(ldict["x"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, "HW16w")
    assert(ldict["x"][0] == "HW16w")
    assert(ldict["x_HW"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")


def test_batchnorm():
    x = sym.Variable("data", shape=(10, 20, 30, 40))
    y = sym.batch_norm(x, axis=1, epsilon=2e-5, name="bn")
    g, ldict = correct_layout(y, "NCHW")
    assert(ldict["data"][0] == "NCHW")
    assert(ldict["bn"][0] == "NCHW")
    assert(ldict["bn"][1] == "C")
    assert(ldict["bn"][2] == "C")
    assert(ldict["bn_beta"][0] == "C")
    assert(ldict["bn_gamma"][0] == "C")
    assert(ldict["bn_moving_mean"][0] == "C")
    assert(ldict["bn_moving_var"][0] == "C")
    # batch_norm can deal with sub-dim of C at the last dim.
    g, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["data"][0] == "NCHW16c")
    assert(ldict["bn"][0] == "NCHW16c")
    assert(ldict["bn"][1] == "C16c")
    assert(ldict["bn"][2] == "C16c")
    assert(ldict["bn_beta"][0] == "C")
    assert(ldict["bn_beta_C16c"][0] == "C16c")
    assert(ldict["bn_gamma"][0] == "C")
    assert(ldict["bn_gamma_C16c"][0] == "C16c")
    assert(ldict["bn_moving_mean"][0] == "C")
    assert(ldict["bn_moving_mean_C16c"][0] == "C16c")
    assert(ldict["bn_moving_var"][0] == "C")
    assert(ldict["bn_moving_var_C16c"][0] == "C16c")
    # but for other layout, it does a layout transform for data
    g, ldict = correct_layout(g, "NCH16cW")
    assert(ldict["data"][0] == "NCH16cW")
    assert(ldict["data_NCHW16c"][0] == "NCHW16c")
    assert(ldict["bn"][0] == "NCHW16c")
    assert(ldict["bn"][1] == "C16c")
    assert(ldict["bn"][2] == "C16c")
    assert(ldict["bn_beta"][0] == "C")
    assert(ldict["bn_beta_C16c"][0] == "C16c")
    assert(ldict["bn_gamma"][0] == "C")
    assert(ldict["bn_gamma_C16c"][0] == "C16c")
    assert(ldict["bn_moving_mean"][0] == "C")
    assert(ldict["bn_moving_mean_C16c"][0] == "C16c")
    assert(ldict["bn_moving_var"][0] == "C")
    assert(ldict["bn_moving_var_C16c"][0] == "C16c")


def test_flatten():
    x = sym.Variable("x", shape=(10, 20, 10, 10))
    y = sym.flatten(x, name="y")
    g, ldict = correct_layout(y, "NCHW")
    assert(ldict["x"][0] == "NCHW")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["x"][0] == "NCHW16c")
    assert(ldict["x_NCHW"][0] == "NCHW")
    assert(ldict["y"][0] == "__undef__")


def test_softmax():
    x = sym.Variable("x", shape=(10, 20, 10, 10))
    y = sym.softmax(x, name="y")
    g, ldict = correct_layout(y, "NCHW")
    assert(ldict["x"][0] == "NCHW")
    assert(ldict["y"][0] == "NCHW")
    # second pass will insert layout transform
    _, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["x"][0] == "NCHW16c")
    assert(ldict["x_NCHW"][0] == "NCHW")
    assert(ldict["y"][0] == "NCHW")


# Level 2
def test_conv2d():
    x = sym.Variable("data", shape=(1, 32, 512, 512))
    y = sym.conv2d(x, name="conv", channels=12,
                   kernel_size=(3,3), padding=(1,1), layout="NCHW")
    _, ldict = correct_layout(y)
    assert(ldict["data"][0] == "NCHW")
    assert(ldict["conv_weight"][0] == "OIHW")
    assert(ldict["conv_bias"][0] == "C")
    assert(ldict["conv"][0] == "NCHW")
    y = sym.conv2d(x, name="conv", channels=12,
                   kernel_size=(3,3), padding=(1,1), layout="NCHW16c",
                   kernel_layout="OIHW16i16o", out_layout="NCHW8c")
    _, ldict = correct_layout(y)
    assert(ldict["data"][0] == "NCHW16c")
    assert(ldict["conv_weight"][0] == "OIHW16i16o")
    assert(ldict["conv_bias"][0] == "C8c")
    assert(ldict["conv"][0] == "NCHW8c")
    y = sym.conv2d(x, name="conv", channels=12,
                   kernel_size=(3,3), padding=(1,1), layout="N16cHWC")
    _, ldict = correct_layout(y)
    assert(ldict["data"][0] == "N16cHWC")
    assert(ldict["conv_weight"][0] == "OIHW")
    assert(ldict["conv_bias"][0] == "16cC")
    assert(ldict["conv"][0] == "N16cHWC")


def test_conv2d_transpose():
    x = sym.Variable("data", shape=(1, 32, 512, 512))
    y = sym.conv2d_transpose(x, name="conv", channels=12,
                             kernel_size=(3,3), padding=(1,1), layout="NCHW")
    _, ldict = correct_layout(y)
    assert(ldict["data"][0] == "NCHW")
    assert(ldict["conv_weight"][0] == "OIHW")
    assert(ldict["conv_bias"][0] == "C")
    assert(ldict["conv"][0] == "NCHW")


def test_max_pool2d():
    x = sym.Variable("data", shape=(1, 32, 512, 512))
    y = sym.max_pool2d(x, name="pool", pool_size=(3,3),
                       padding=(1,1), layout="NCHW")
    g, ldict = correct_layout(y)
    assert(ldict["data"][0] == "NCHW")
    assert(ldict["pool"][0] == "NCHW")
    # if index of H and W remain the same,
    # pool2d does not convert the layout.
    g, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["data"][0] == "NCHW16c")
    assert(ldict["pool"][0] == "NCHW16c")
    # for other layout it requires a layout transform.
    g, ldict = correct_layout(g, "NHWC")
    assert(ldict["data"][0] == "NHWC")
    assert(ldict["data_NCHW"][0] == "NCHW")
    assert(ldict["pool"][0] == "NCHW")


def test_global_pool2d():
    x = sym.Variable("data", shape=(1, 32, 512, 512))
    y = sym.global_max_pool2d(x, name="pool", layout="NCHW")
    g, ldict = correct_layout(y)
    assert(ldict["data"][0] == "NCHW")
    assert(ldict["pool"][0] == "NCHW")
    # if index of H and W remain the same,
    # pool2d does not convert the layout.
    g, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["data"][0] == "NCHW16c")
    assert(ldict["pool"][0] == "NCHW16c")
    # for other layout it requires a layout transform.
    g, ldict = correct_layout(g, "NHWC")
    assert(ldict["data"][0] == "NHWC")
    assert(ldict["data_NCHW"][0] == "NCHW")
    assert(ldict["pool"][0] == "NCHW")


# Level 3
def test_reshape():
    x = sym.Variable("x", shape=(4,))
    y = sym.reshape(x, shape=(2,2), name="y")
    g, ldict = correct_layout(y, "C")
    assert(ldict["x"][0] == "C")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    g, ldict = correct_layout(g, "C16c")
    assert(ldict["x"][0] == "C16c")
    assert(ldict["x_C"][0] == "C")
    assert(ldict["y"][0] == "__undef__")


def test_transpose():
    x = sym.Variable("x", shape=(1, 32, 512, 512))
    y = sym.transpose(x, name="y", axes=(0, 2, 3, 1))
    g, ldict = correct_layout(y, "NCHW")
    assert(ldict["x"][0] == "NCHW")
    assert(ldict["y"][0] == "NHWC")
    # second pass will insert layout transform
    g, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["x"][0] == "NCHW16c")
    assert(ldict["x_NCHW"][0] == "NCHW")
    assert(ldict["y"][0] == "NHWC")


def test_broadcast_to():
    x = sym.Variable("x", shape=(4, 1))
    y = sym.broadcast_to(x, shape=(0, 4), name="y")
    g, ldict = correct_layout(y, "HW")
    assert(ldict["x"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    g, ldict = correct_layout(g, "HW16h")
    assert(ldict["x"][0] == "HW16h")
    assert(ldict["x_HW"][0] == "HW")
    assert(ldict["y"][0] == "__undef__")


def test_broadcast_binary():
    x = sym.Variable("x", shape=(1, 16, 512, 512))
    y = sym.Variable("y", shape=(16, 512, 512))
    z = sym.broadcast_add(x, y, name="z")
    g, ldict = correct_layout(z, {"x": "NCHW", "y": "CHW"})
    assert(ldict["x"][0] == "NCHW")
    assert(ldict["y"][0] == "CHW")
    assert(ldict["z"][0] == "NCHW")
    # prior to keep the left layout if they do not match.
    g, ldict = correct_layout(g, {"x": "NCHW16c", "y": "CHW"})
    assert(ldict["x"][0] == "NCHW16c")
    assert(ldict["y"][0] == "CHW")
    assert(ldict["y_CHW16c"][0] == "CHW16c")
    assert(ldict["z"][0] == "NCHW16c")
    # broadcast_add(HCW16c, N16nCH16cW)
    g, ldict = correct_layout(z, {"x": "HCW16c", "y": "N16nCH16cW"})
    assert(ldict["x"][0] == "HCW16c")
    assert(ldict["y"][0] == "N16nCH16cW")
    assert(ldict["x_CH16cW"][0] == "CH16cW")
    assert(ldict["z"][0] == "N16nCH16cW")


def test_reduce():
    x = sym.Variable("x", shape=(1, 16, 512, 512))
    y = sym.sum(x, name="y", axis=1)
    g, ldict = correct_layout(y, "NCHW")
    assert(ldict["x"][0] == "NCHW")
    assert(ldict["y"][0] == "__undef__")
    # second pass will insert layout transform
    g, ldict = correct_layout(g, "NCHW16c")
    assert(ldict["x"][0] == "NCHW16c")
    assert(ldict["x_NCHW"][0] == "NCHW")
    assert(ldict["y"][0] == "__undef__")


if __name__ == "__main__":
    test_dense()
    test_matmul()
    test_concatenate()
    test_expand_dims()
    test_split()
    test_batchnorm()
    test_flatten()
    test_softmax()
    test_conv2d()
    test_conv2d_transpose()
    test_max_pool2d()
    test_global_pool2d()
    test_reshape()
    test_transpose()
    test_broadcast_to()
    test_broadcast_binary()
    test_reduce()
