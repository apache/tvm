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
"""Test task extraction for auto-scheduler"""
import tvm.relay.testing
import tvm.testing
from tvm import auto_scheduler, relay


def get_network(name, batch_size=1, layout="NHWC"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefer NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    elif layout == "NCDHW":
        image_shape = (3, 16, 224, 224)
    elif layout == "NDHWC":
        image_shape = (3, 224, 224, 16)
    else:
        raise ValueError("Invalid layout: " + layout)

    if name == "resnet-18":
        mod, params = relay.testing.resnet.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "resnet-50":
        mod, params = relay.testing.resnet.get_workload(
            num_layers=50, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "winograd-test":
        input_shape = [1, 7, 7, 64]
        output_shape = input_shape

        data = relay.var("data", shape=input_shape, dtype="float32")
        net = relay.testing.layers.conv2d(
            data=data,
            channels=64,
            kernel_size=3,
            strides=1,
            padding=1,
            data_layout="NHWC",
            kernel_layout="HWIO",
            name="",
        )
        bias = relay.var("conv1_bias")
        net = relay.nn.bias_add(net, bias, 3)
        net = relay.nn.relu(net)
        mod, params = relay.testing.create_workload(net)
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(
            num_layers=18, batch_size=batch_size, layout=layout, image_shape=image_shape
        )
    elif name == "dcgan":
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size, layout=layout)
    elif name == "mlp":
        data = relay.var("data", shape=(batch_size, 32))
        fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=32)
        fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
        act1 = relay.nn.relu(fc1)
        fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=32)
        fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
        act2 = relay.nn.relu(fc2)
        mlp = act2
        args = relay.analysis.free_vars(act2)
        mlp = relay.Function(args, mlp)
        mod, params = relay.testing.init.create_workload(mlp)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params


@tvm.testing.requires_cuda
def test_task_extraction_cuda():
    auto_scheduler.enable_relay_integration()
    target = tvm.target.Target("cuda")

    mod, params = get_network("mlp")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    assert len(tasks) == 1
    assert sum(task_weights) == 2

    for layout in ["NHWC", "NCHW"]:
        mod, params = get_network("resnet-18", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 21
        assert sum(task_weights) == 22

        mod, params = get_network("mobilenet", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 20
        assert sum(task_weights) == 28

    for layout in ["NCDHW", "NDHWC"]:
        mod, params = get_network("resnet3d-18", layout=layout)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        assert len(tasks) == 21
        assert sum(task_weights) == 22

    auto_scheduler.enable_relay_integration(False)


if __name__ == "__main__":
    test_task_extraction_cuda()
