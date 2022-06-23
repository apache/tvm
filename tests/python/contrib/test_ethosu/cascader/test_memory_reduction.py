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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")

import numpy as np
import tensorflow as tf
import tflite.Model
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.micro import model_library_format as mlf
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
import tvm
from tvm import WorkspaceMemoryPools, WorkspacePoolInfo, PoolInfoProperties

from .. import infra


def _get_ethosu_workspace_size(
    mod, params, accel_type, pool_size, enable_cascader, enable_striping
):
    enable_usmp = True

    target = tvm.target.Target("c")
    ethosu_target = tvm.target.Target("ethos-u")
    runtime = Runtime("crt")

    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": 16,
            "interface-api": "c",
            "unpacked-api": True,
        },
    )
    pass_config = {
        "tir.disable_vectorize": True,
        "relay.ext.ethos-u.options": {
            "accelerator_config": accel_type,
            "enable_cascader": enable_cascader,
            "enable_striping": enable_striping,
        },
        "tir.usmp.enable": enable_usmp,
        "tir.usmp.algorithm": "hill_climb",
        "tir.disable_storage_rewrite": enable_usmp,
    }

    workspace_memory_pools = WorkspaceMemoryPools(
        [
            WorkspacePoolInfo(
                "SRAM",
                [target, ethosu_target],
                PoolInfoProperties(
                    size_hint_bytes=pool_size,
                    read_bandwidth_bytes_per_cycle=16,
                    write_bandwidth_bytes_per_cycle=16,
                    target_burst_bytes={ethosu_target: 1},
                ),
            ),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, config=pass_config):
        lib = tvm.relay.build(
            mod,
            target,
            executor=executor,
            runtime=runtime,
            workspace_memory_pools=workspace_memory_pools,
            params=params,
        )

    mlf_memory_map = mlf._build_function_memory_map(lib.function_metadata)
    return mlf_memory_map["main"][0]["workspace_size_bytes"]


@pytest.mark.parametrize(
    "accel_type, expected_ws_size_without_striping, expected_ws_size_with_striping",
    [
        ("ethos-u55-256", 1067520, 14208),
        ("ethos-u55-128", 1067520, 4080),
        ("ethos-u55-64", 1067520, 4080),
        ("ethos-u55-32", 1067504, 4064),
    ],
)
def test_double_conv2d(
    accel_type, expected_ws_size_without_striping, expected_ws_size_with_striping
):
    np.random.seed(1)
    ifm_shape = (1, 321, 212, 6)

    @tf.function
    def tf_graph(x):
        ofm_channels = 10
        conv2d = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[3, 2, ifm_shape[3], ofm_channels]),  # HWIO
                dtype=tf.float32,
            ),
            strides=(1, 1),
            padding="VALID",
            dilations=(2, 1),
        )
        conv2d = tf.nn.conv2d(
            conv2d,
            filters=tf.constant(
                np.random.uniform(size=(1, 1, ofm_channels, 3)),  # HWIO
                dtype=tf.float32,
            ),
            strides=(3, 2),
            padding="SAME",
            dilations=(1, 1),
        )

        return conv2d

    _, tflite_graph = infra.get_tflite_graph(tf_graph, [ifm_shape])
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(tflite_model)
    mod = partition_for_ethosu(relay_module, params)

    # Run the graph without the cascader, with lots of memory
    pool_size = 2000000
    workspace_size_cascader_disabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=False, enable_striping=False
    )
    workspace_size_cascader_enabled_striping_disabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=True, enable_striping=False
    )
    # if striping is not done, it should be same as cacader disabled
    assert workspace_size_cascader_disabled == workspace_size_cascader_enabled_striping_disabled

    # Run the same graph with the cascader, giving it less memory to persuade cascder to cascade
    pool_size = 600000
    workspace_size_cascader_enabled_striping_enabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=True, enable_striping=True
    )

    assert workspace_size_cascader_disabled == expected_ws_size_without_striping
    assert workspace_size_cascader_enabled_striping_enabled == expected_ws_size_with_striping


# TODO(ekalda): Fix a bug in the block config selection that selects block config that is too large
# for the smaller accelerators
@pytest.mark.parametrize(
    "accel_type, expected_ws_size_without_striping, expected_ws_size_with_striping",
    [
        ("ethos-u55-256", 180288, 15200),
        ("ethos-u55-128", 180288, 15200),
        ("ethos-u55-64", 180288, 14432),
        ("ethos-u55-32", 180272, 14416),
    ],
)
def test_depthwise2d_conv2d_pooling(
    accel_type, expected_ws_size_without_striping, expected_ws_size_with_striping
):
    np.random.seed(2)
    ifm_shape = (1, 80, 75, 3)

    @tf.function
    def tf_graph(x):
        # This graph will execute as one cascade
        ofm_channels = 7
        conv2d = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[3, 2, ifm_shape[3], ofm_channels]),  # HWIO
                dtype=tf.float32,
            ),
            strides=(1, 1),
            padding="VALID",
            dilations=(1, 1),
        )
        depthwise2d = tf.nn.depthwise_conv2d(
            conv2d,
            tf.constant(np.random.uniform(size=(3, 3, ofm_channels, 1)), dtype=tf.float32),  # HWC1
            strides=(1, 1, 1, 1),
            padding="VALID",
            dilations=(1, 1),
        )
        relu = tf.nn.relu(depthwise2d)
        conv2d = tf.nn.conv2d(
            relu,
            filters=tf.constant(
                np.random.uniform(size=[3, 2, ofm_channels, 2]),  # HWIO
                dtype=tf.float32,
            ),
            strides=(1, 1),
            padding="SAME",
            dilations=(1, 1),
        )
        max_pool = tf.nn.max_pool(conv2d, (3, 3), (1, 1), "SAME")

        return max_pool

    _, tflite_graph = infra.get_tflite_graph(tf_graph, [ifm_shape])
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(tflite_model)
    mod = partition_for_ethosu(relay_module, params)

    # Run the graph without the cascader, with lots of memory
    pool_size = 10**6
    workspace_size_cascader_disabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=False, enable_striping=False
    )
    workspace_size_cascader_enabled_striping_disabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=True, enable_striping=False
    )
    # if striping is not done, it should be same as cacader disabled
    assert workspace_size_cascader_disabled == workspace_size_cascader_enabled_striping_disabled

    # Run the same graph with the cascader, giving it less memory to persuade cascder to cascade
    pool_size = 50000
    workspace_size_cascader_enabled_striping_enabled = _get_ethosu_workspace_size(
        mod, params, accel_type, pool_size, enable_cascader=True, enable_striping=True
    )

    assert workspace_size_cascader_disabled == expected_ws_size_without_striping
    assert workspace_size_cascader_enabled_striping_enabled == expected_ws_size_with_striping
