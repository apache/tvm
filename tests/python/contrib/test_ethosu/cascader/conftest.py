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
import pytest

ethosu_enabled = True
try:
    import ethosu.vela
except ImportError:
    ethosu_enabled = False

import tvm.contrib.ethosu.cascader as cs


@pytest.fixture
def FLASH():
    return cs.MemoryRegion(
        name="FLASH",
        size=10**7,
        read_bandwidth=4,
        write_bandwidth=4,
        read_latency=0,
        write_latency=0,
        burst_length=1,
    )


@pytest.fixture
def DRAM():
    return cs.MemoryRegion(
        name="DRAM",
        size=10**9,
        read_bandwidth=8,
        write_bandwidth=8,
        read_latency=0,
        write_latency=0,
        burst_length=1,
    )


@pytest.fixture
def SRAM():
    return cs.MemoryRegion(
        name="SRAM",
        size=10**6,
        read_bandwidth=16,
        write_bandwidth=16,
        read_latency=0,
        write_latency=0,
        burst_length=1,
    )


if ethosu_enabled:
    import tvm
    from tvm import relay
    from tvm.relay.testing import run_opt_pass

    from .infra import create_te_graph
    from ..infra import (
        make_ethosu_conv2d,
        make_ethosu_depthwise_conv2d,
        make_ethosu_binary_elementwise,
    )

    def make_TwoConv2DTE():
        def _get_func():
            ifm = relay.var("ifm", shape=(1, 12, 12, 8), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm,
                ifm_channels=8,
                ofm_channels=32,
                kernel_shape=(1, 1),
                padding=(0, 0),
                strides=(1, 1),
                dilation=(1, 1),
                activation="NONE",
                ifm_layout="NHWC",
                ofm_layout="NHCWB16",
            )
            conv2 = make_ethosu_conv2d(
                ifm=conv1,
                ifm_channels=32,
                ofm_channels=16,
                kernel_shape=(3, 3),
                padding=(1, 1),
                strides=(1, 1),
                dilation=(1, 1),
                activation="NONE",
                ifm_layout="NHCWB16",
                ofm_layout="NHWC",
            )
            func = relay.Function(relay.analysis.free_vars(conv2), conv2)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def TwoConv2DTE():
        return make_TwoConv2DTE()

    @pytest.fixture
    def TwoConv2DGraph():
        _, te_graph, const_dict = make_TwoConv2DTE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)

    def make_TwoConv2DWithSliceTE():
        def _get_func():
            ifm = relay.var("ifm", shape=(1, 12, 12, 8), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm,
                ifm_channels=8,
                ofm_channels=64,
                kernel_shape=(1, 1),
                padding=(0, 0),
                strides=(1, 1),
                dilation=(1, 1),
                activation="NONE",
                ifm_layout="NHWC",
                ofm_layout="NHWC",
            )
            strided_slice = relay.strided_slice(conv1, [0, 0, 0, 0], [1, 6, 6, 128])
            conv2 = make_ethosu_conv2d(
                ifm=strided_slice,
                ifm_channels=64,
                ofm_channels=16,
                kernel_shape=(3, 3),
                padding=(1, 1),
                strides=(1, 1),
                dilation=(1, 1),
                activation="NONE",
                ifm_layout="NHWC",
                ofm_layout="NHCWB16",
            )
            func = relay.Function(relay.analysis.free_vars(conv2), conv2)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def TwoConv2DWithSliceTE():
        return make_TwoConv2DWithSliceTE()

    @pytest.fixture
    def TwoConv2DWithSliceGraph():
        _, te_graph, const_dict = make_TwoConv2DWithSliceTE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)

    def make_MobileNetv2DiamondTE():
        def _get_func():
            ifm = relay.var("ifm", shape=(1, 56, 56, 96), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm,
                ifm_channels=96,
                ofm_channels=24,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv2 = make_ethosu_conv2d(
                ifm=conv1,
                ifm_channels=24,
                ofm_channels=144,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth1 = make_ethosu_depthwise_conv2d(
                ifm=conv2,
                channels=144,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv3 = make_ethosu_conv2d(
                ifm=depth1,
                ifm_channels=144,
                ofm_channels=24,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            add1 = make_ethosu_binary_elementwise(
                ifm=conv1,
                ifm2=conv3,
                ifm_channels=24,
                ifm2_channels=24,
                operator_type="ADD",
                ofm_dtype="int8",
            )
            func = relay.Function(relay.analysis.free_vars(add1), add1)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def MobileNetv2DiamondTE():
        return make_MobileNetv2DiamondTE()

    @pytest.fixture
    def MobileNetv2DiamondGraph():
        _, te_graph, const_dict = make_MobileNetv2DiamondTE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)

    def make_BinaryTE():
        def _get_func():
            ifm_a = relay.var("ifm_a", shape=(1, 8, 8, 8), dtype="int8")
            ifm_b = relay.var("ifm_b", shape=(1, 8, 8, 8), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm_a,
                ifm_channels=8,
                ofm_channels=8,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv2 = make_ethosu_conv2d(
                ifm=ifm_b,
                ifm_channels=8,
                ofm_channels=8,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            add1 = make_ethosu_binary_elementwise(
                ifm=conv1,
                ifm2=conv2,
                ifm_channels=8,
                ifm2_channels=8,
                operator_type="ADD",
                ofm_dtype="int8",
            )
            func = relay.Function(relay.analysis.free_vars(add1), add1)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def BinaryTE():
        return make_BinaryTE()

    @pytest.fixture
    def BinaryGraph():
        _, te_graph, const_dict = make_BinaryTE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)

    def make_MobileNetv1StartTE():
        def _get_func():
            ifm = relay.var("ifm", shape=(1, 224, 224, 3), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm,
                ifm_channels=3,
                ofm_channels=32,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            depth1 = make_ethosu_depthwise_conv2d(
                ifm=conv1,
                channels=32,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv2 = make_ethosu_conv2d(
                ifm=depth1,
                ifm_channels=32,
                ofm_channels=64,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth2 = make_ethosu_depthwise_conv2d(
                ifm=conv2,
                channels=64,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            conv3 = make_ethosu_conv2d(
                ifm=depth2,
                ifm_channels=64,
                ofm_channels=128,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth3 = make_ethosu_depthwise_conv2d(
                ifm=conv3,
                channels=128,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv4 = make_ethosu_conv2d(
                ifm=depth3,
                ifm_channels=128,
                ofm_channels=128,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth4 = make_ethosu_depthwise_conv2d(
                ifm=conv4,
                channels=128,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            func = relay.Function(relay.analysis.free_vars(depth4), depth4)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def MobileNetv1StartTE():
        return make_MobileNetv1StartTE()

    @pytest.fixture
    def MobileNetv1StartGraph():
        _, te_graph, const_dict = make_MobileNetv1StartTE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)

    def make_MobileNetv1TE():
        def _get_func():
            ifm = relay.var("ifm", shape=(1, 224, 224, 3), dtype="int8")
            conv1 = make_ethosu_conv2d(
                ifm=ifm,
                ifm_channels=3,
                ofm_channels=32,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            depth1 = make_ethosu_depthwise_conv2d(
                ifm=conv1,
                channels=32,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv2 = make_ethosu_conv2d(
                ifm=depth1,
                ifm_channels=32,
                ofm_channels=64,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth2 = make_ethosu_depthwise_conv2d(
                ifm=conv2,
                channels=64,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            conv3 = make_ethosu_conv2d(
                ifm=depth2,
                ifm_channels=64,
                ofm_channels=128,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth3 = make_ethosu_depthwise_conv2d(
                ifm=conv3,
                channels=128,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv4 = make_ethosu_conv2d(
                ifm=depth3,
                ifm_channels=128,
                ofm_channels=128,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth4 = make_ethosu_depthwise_conv2d(
                ifm=conv4,
                channels=128,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            conv5 = make_ethosu_conv2d(
                ifm=depth4,
                ifm_channels=128,
                ofm_channels=256,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth5 = make_ethosu_depthwise_conv2d(
                ifm=conv5,
                channels=256,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv6 = make_ethosu_conv2d(
                ifm=depth5,
                ifm_channels=256,
                ofm_channels=256,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth6 = make_ethosu_depthwise_conv2d(
                ifm=conv6,
                channels=256,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            conv7 = make_ethosu_conv2d(
                ifm=depth6,
                ifm_channels=256,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth7 = make_ethosu_depthwise_conv2d(
                ifm=conv7,
                channels=512,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv8 = make_ethosu_conv2d(
                ifm=depth7,
                ifm_channels=512,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth8 = make_ethosu_depthwise_conv2d(
                ifm=conv8,
                channels=512,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv9 = make_ethosu_conv2d(
                ifm=depth8,
                ifm_channels=512,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth9 = make_ethosu_depthwise_conv2d(
                ifm=conv9,
                channels=512,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv10 = make_ethosu_conv2d(
                ifm=depth9,
                ifm_channels=512,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth10 = make_ethosu_depthwise_conv2d(
                ifm=conv10,
                channels=512,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv11 = make_ethosu_conv2d(
                ifm=depth10,
                ifm_channels=512,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth11 = make_ethosu_depthwise_conv2d(
                ifm=conv11,
                channels=512,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv12 = make_ethosu_conv2d(
                ifm=depth11,
                ifm_channels=512,
                ofm_channels=512,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth12 = make_ethosu_depthwise_conv2d(
                ifm=conv12,
                channels=512,
                kernel_shape=(3, 3),
                padding=(0, 0, 1, 1),
                strides=(2, 2),
                dilation=(1, 1),
            )
            conv13 = make_ethosu_conv2d(
                ifm=depth12,
                ifm_channels=512,
                ofm_channels=1024,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            depth13 = make_ethosu_depthwise_conv2d(
                ifm=conv13,
                channels=1024,
                kernel_shape=(3, 3),
                padding=(1, 1, 1, 1),
                strides=(1, 1),
                dilation=(1, 1),
            )
            conv14 = make_ethosu_conv2d(
                ifm=depth13,
                ifm_channels=1024,
                ofm_channels=1024,
                kernel_shape=(1, 1),
                padding=(0, 0, 0, 0),
                strides=(1, 1),
                dilation=(1, 1),
            )
            func = relay.Function(relay.analysis.free_vars(conv14), conv14)
            func = run_opt_pass(func, relay.transform.InferType())
            return func

        func = _get_func()
        te_graph, const_dict = create_te_graph(func)
        sch = tvm.te.create_schedule([t.op for t in te_graph.outputs])
        return sch, te_graph, const_dict

    @pytest.fixture
    def MobileNetv1TE():
        return make_MobileNetv1TE()

    @pytest.fixture
    def MobileNetv1Graph():
        _, te_graph, const_dict = make_MobileNetv1TE()
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        return cs.create_cascader_graph(te_graph, const_dict, device_config)
