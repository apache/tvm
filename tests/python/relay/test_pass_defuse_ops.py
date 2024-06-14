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
import numpy
import pytest
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass


def test_defuse_simple():
    """Simple testcase."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        return relay.Function([x], w)

    x = before()
    x = run_opt_pass(x, transform.InferType())
    fused = run_opt_pass(x, transform.FuseOps())
    defused = run_opt_pass(fused, transform.DefuseOps())

    tvm.ir.assert_structural_equal(x, defused)


def test_inception_like():
    def conv(data):
        y = relay.nn.conv2d(data, relay.var("w"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        return relay.nn.relu(data=y)

    def inception_like(data):
        c0 = conv(data)
        c1 = conv(data)
        return relay.concatenate((c0, c1), axis=1)

    def before(dshape):
        x = relay.var("x", shape=dshape)
        in1 = inception_like(x)
        in2 = inception_like(in1)
        return relay.Function(relay.analysis.free_vars(in2), in2)

    dshape = (1, 16, 64, 64)
    x = before(dshape)
    x = run_opt_pass(x, transform.InferType())
    fused = run_opt_pass(x, transform.FuseOps())
    defused = run_opt_pass(fused, transform.DefuseOps())

    tvm.ir.assert_structural_equal(x, defused)


def test_defuse_complex():
    """Complex defuse testcase"""

    def fused_conv2d_batch_norm(w):
        data = relay.var("data", shape=(1, 224, 224, 3))
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((64,), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((64,), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((64,), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((64,), "float32"))
        c0 = relay.nn.conv2d(
            data,
            w,
            strides=(2, 2),
            padding=(3, 3, 3, 3),
            channels=64,
            kernel_size=(7, 7),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c2 = c1[0]
        return relay.Function(relay.analysis.free_vars(c2), c2)

    def fused_conv2d_batch_norm_relu(z):
        data2 = relay.var("data2", shape=(1, 56, 56, 64))
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((64,), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((64,), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((64,), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((64,), "float32"))
        c0 = relay.nn.conv2d(
            data2,
            z,
            padding=(1, 1, 1, 1),
            channels=64,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c2 = c1[0]
        c3 = relay.nn.relu(data=c2)
        return relay.Function(relay.analysis.free_vars(c3), c3)

    def fused_max_pool2d():
        data1 = relay.var("data1", shape=(1, 112, 112, 64))
        a1 = relay.nn.max_pool2d(
            data1,
            pool_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1, 1, 1),
            layout="NHWC",
            out_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(a1), a1)

    def fused_add_relu():
        data1 = relay.var("data1", shape=(1, 56, 56, 64))
        data2 = relay.var("data2", shape=(1, 56, 56, 64))
        a0 = relay.add(data1, data2)
        a1 = relay.nn.relu(a0)
        return relay.Function(relay.analysis.free_vars(a1), a1)

    def before_fused(conv_layer1_weight, conv_layer2_weight):
        data = relay.var("data", shape=(1, 3, 224, 224))
        data1 = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
        bn_gamma0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_beta0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_mmean0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_mvar0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        a0 = fused_conv2d_batch_norm(conv_layer1_weight)
        a1 = fused_max_pool2d()
        a2 = fused_conv2d_batch_norm_relu(conv_layer2_weight)
        a3 = fused_add_relu()
        y0 = relay.Call(a0, [data1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0])
        y1 = relay.Call(a1, [y0])
        y2 = relay.Call(a2, [y1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0])
        y3 = relay.Call(a3, [y1, y2])
        return relay.Function(relay.analysis.free_vars(y3), y3)

    def golden_defused(conv_layer1_weight, conv_layer2_weight):
        data = relay.var("data", shape=(1, 3, 224, 224))
        data1 = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
        bn_gamma0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_beta0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_mmean0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        bn_mvar0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
        c0 = relay.nn.conv2d(
            data1,
            conv_layer1_weight,
            strides=(2, 2),
            padding=(3, 3, 3, 3),
            channels=64,
            kernel_size=(7, 7),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c2 = c1[0]
        c3 = relay.nn.max_pool2d(
            c2,
            pool_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1, 1, 1),
            layout="NHWC",
            out_layout="NHWC",
        )
        c4 = relay.nn.conv2d(
            c3,
            conv_layer2_weight,
            padding=(1, 1, 1, 1),
            channels=64,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c5 = relay.nn.batch_norm(c4, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c6 = c5[0]
        c7 = relay.nn.relu(c6)
        c8 = relay.add(c3, c7)
        c9 = relay.nn.relu(c8)
        return relay.Function(relay.analysis.free_vars(c9), c9)

    # creating weight constants for the two convolution layers
    # in the input fused model and the golden defused model.
    conv_layer1_weight = relay.nn.Constant(
        tvm.nd.array(numpy.ndarray(shape=(64, 7, 7, 3), dtype="float32"))
    )
    conv_layer2_weight = relay.nn.Constant(
        tvm.nd.array(numpy.ndarray(shape=(64, 3, 3, 64), dtype="float32"))
    )
    x = before_fused(conv_layer1_weight, conv_layer2_weight)
    x = run_opt_pass(x, transform.InferType())
    defused = run_opt_pass(x, transform.DefuseOps())

    golden1 = golden_defused(conv_layer1_weight, conv_layer2_weight)
    golden1 = run_opt_pass(golden1, transform.InferType())

    tvm.ir.assert_structural_equal(defused, golden1)


if __name__ == "__main__":
    tvm.testing.main()
