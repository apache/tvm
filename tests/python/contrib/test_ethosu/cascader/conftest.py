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


if ethosu_enabled:
    import tvm
    from tvm import relay
    from tvm.relay.testing import run_opt_pass

    from .infra import create_te_graph
    from ..infra import make_ethosu_conv2d

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
