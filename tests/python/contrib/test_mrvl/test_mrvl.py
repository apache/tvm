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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name

"""Test Marvell BYOC partitioning, code generation and runtime"""

import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
from tvm.testing.utils import requires_mrvl
from tvm.relay.op.contrib.mrvl import partition_for_mrvl
from .infrastructure import verify_codegen
from .infrastructure import run_and_verify_func
from tvm.testing import requires_mrvl


@requires_mrvl
def test_mrvl_fuse():
    def get_blocks(
        prefix,
        data,
        in_channel,
        out_channel,
        include_bias_add=True,
        include_bn=True,
        include_sigmoid=False,
    ):
        weight = relay.var(prefix + "weight")
        bias = relay.var(prefix + "bias")
        bn_gamma = relay.var(prefix + "bn_gamma")
        bn_beta = relay.var(prefix + "bn_beta")
        bn_mmean = relay.var(prefix + "bn_mean")
        bn_mvar = relay.var(prefix + "bn_var")

        layer = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=out_channel, padding=(1, 1)
        )
        if include_bias_add:
            layer = relay.nn.bias_add(layer, bias)
        if include_bn:
            bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta, bn_mmean, bn_mvar)
            layer = bn_output[0]
        if include_sigmoid:
            layer = relay.sigmoid(layer)
        layer = relay.nn.relu(layer)
        return layer

    def get_net(include_bias_add=True, include_bn=True, include_sigmoid=False):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        block1 = get_blocks("block1_", data, 3, 8, include_bias_add, include_bn, include_sigmoid)
        block2 = get_blocks("block2_", block1, 8, 8, False, False, include_sigmoid)
        return relay.Function(relay.analysis.free_vars(block2), block2)

    def test_detect_pattern(include_bias_add, include_bn, include_sigmoid, num_expected_partition):
        net = get_net(include_bias_add, include_bn, include_sigmoid)
        mod, params = tvm.relay.testing.create_workload(net)
        mod = partition_for_mrvl(mod, params)
        assert len(mod.functions) - 1 == num_expected_partition

    def test_sum_pattern(num_expected_partition):
        def get_conv2d_bn_sum_relu(
            x_shape=(1, 32, 8, 8),
            k_shape=(16, 32, 3, 3),
            sum_shape=(1, 16, 6, 6),
            dtype="float32",
        ):
            x = relay.var("x", shape=(x_shape), dtype=dtype)
            kernel = relay.const(np.random.randint(0, 1, k_shape).astype(dtype))
            bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
            beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
            gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
            moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
            moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
            sum_data = relay.var("data1", shape=sum_shape, dtype=dtype)

            dic = {"x": x_shape, "bias": (k_shape[0],), "sum_data": sum_shape}
            param_lst = ["bias", "sum_data"]

            conv = relay.nn.conv2d(
                x,
                kernel,
                channels=k_shape[0],
                kernel_size=k_shape[2:4],
            )
            conv_bias = relay.nn.bias_add(conv, bias)
            conv_bias_bn, _, _ = relay.nn.batch_norm(
                conv_bias,
                gamma=gamma,
                beta=beta,
                moving_mean=moving_mean,
                moving_var=moving_var,
                axis=1,
                center=True,
                scale=True,
                epsilon=1e-5,
            )
            conv_bias_bn_sum = relay.add(conv_bias_bn, sum_data)
            return relay.nn.relu(conv_bias_bn_sum), dic, param_lst

        net, dic, param_lst = get_conv2d_bn_sum_relu()
        net = tvm.IRModule.from_expr(net)
        params = {x: np.random.uniform(-1, 1, dic[x]).astype("float32") for x in param_lst}
        mod = partition_for_mrvl(net, params)
        assert len(mod.functions) - 1 == num_expected_partition

    def test_partition():
        test_detect_pattern(True, False, False, 1)
        test_detect_pattern(False, True, False, 1)
        test_detect_pattern(False, False, True, 2)
        test_detect_pattern(True, True, False, 1)
        test_detect_pattern(True, False, True, 2)
        test_detect_pattern(False, True, True, 2)
        test_detect_pattern(False, False, False, 1)
        test_detect_pattern(True, True, True, 2)
        test_sum_pattern(1)

    def test_partition_mobilenet(num_expected_partition):
        mod, params = relay.testing.mobilenet.get_workload()
        mod = partition_for_mrvl(mod, params)
        assert len(mod.functions) - 1 == num_expected_partition

    test_partition()
    test_partition_mobilenet(1)


@requires_mrvl
def test_conv2d():
    """Test conv2d operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        func = relay.Function([x], y)
        params = {}
        params["w"] = arr
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params=params, tvm_ops=1, contains="mrvl.conv2d_nhwc2nhwc")
        return func, {"x": (1, 3, 224, 224), "w": (16, 3, 3, 3)}, ["w"], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_dense():
    """Test dense operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 16))
        arr = np.random.rand(16, 16).astype("float32")
        w = relay.const(arr)
        y = relay.nn.dense(x, w)
        func = relay.Function([x], y)
        params = {}
        params["w"] = arr
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params=params, tvm_ops=0, contains="mrvl.fc_ni2no")
        return func, {"x": (1, 16), "w": (16, 16)}, ["w"], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_maxpool2d():
    """Test maxpool2d operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        y = relay.nn.max_pool2d(y)
        func = relay.Function([x], y)
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params={}, tvm_ops=1, contains="mrvl.maxpool2d_nhwc2nhwc")
        return func, {"x": (1, 3, 224, 224)}, [], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_avgpool2d():
    """Test avgpool2d operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        y = relay.nn.avg_pool2d(y)
        func = relay.Function([x], y)
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params={}, tvm_ops=1, contains="mrvl.avgpool2d_nhwc2nhwc")
        return func, {"x": (1, 3, 224, 224)}, [], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_globalavgpool2d():
    """Test globalavgpool2d operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        y = relay.nn.global_avg_pool2d(y)
        func = relay.Function([x], y)
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params={}, tvm_ops=1, contains="mrvl.globalavgpool2d_nhwc2nhwc")
        return func, {"x": (1, 3, 224, 224)}, [], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_globalmaxpool2d():
    """Test globalmaxpool2d operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        y = relay.nn.global_max_pool2d(y)
        func = relay.Function([x], y)
        params = {}
        params["w"] = arr
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params=params, tvm_ops=2, contains="mrvl.globalmaxpool2d_nhwc2nhwc")
        return func, {"x": (1, 3, 224, 224), "w": (16, 3, 3, 3)}, ["w"], option_dict

    run_and_verify_func(get_graph())


@requires_mrvl
def test_squeeze():
    """Test squeeze operator for "mrvl" targets"""

    def get_graph():
        x = relay.var("x", shape=(1, 3, 224, 224))
        arr = np.random.rand(16, 3, 3, 3).astype("float32")
        w = relay.const(arr)
        y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
        y = relay.reshape(y, newshape=(1, 1, 16, 112, 112))
        y = relay.squeeze(y, axis=[0, 1])
        func = relay.Function([x], y)
        mod = tvm.IRModule()
        mod["main"] = func
        option_dict = {"num_tiles": 1}
        verify_codegen(mod, params={}, tvm_ops=3, contains="mrvl.squeeze")
        return func, {"x": (1, 3, 224, 224)}, [], option_dict

    run_and_verify_func(get_graph())


if __name__ == "__main__":
    test_mrvl_fuse()
    test_conv2d()
    test_dense()
    test_maxpool2d()
    test_avgpool2d()
    test_globalavgpool2d()
    test_globalmaxpool2d()
    test_squeeze()
