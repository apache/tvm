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
"""OpenCL ML network tests."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from test_clml.infrastructure import build_and_run, build_and_run_vm
import pytest


def _build_and_run_network(remote, mod, params, input_data, target, executor_type, tvm_log=""):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        if executor_type == "ge":
            outputs.append(
                build_and_run(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
        else:
            outputs.append(
                build_and_run_vm(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
    return outputs


def get_network(name, batch_size, dtype="float32"):
    """Get the symbol definition and random weight of a network

    Parameters
    ----------
    name: str
        The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
    batch_size: int
        batch size
    dtype: str
        Data type

    Returns
    -------
    net: tvm.IRModule
        The relay function of network definition
    params: dict
        The random parameters for benchmark
    input_shape: tuple
        The shape of input tensor
    output_shape: tuple
        The shape of output tensor
    """
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if name == "mobilenet":
        net, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        net, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif "resnet" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "densenet" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.densenet.get_workload(
            densenet_size=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "squeezenet" in name:
        version = name.split("_v")[1]
        net, params = testing.squeezenet.get_workload(
            batch_size=batch_size, version=version, dtype=dtype
        )
    else:
        raise ValueError("Unsupported network: " + name)

    initializer = relay.testing.init.Xavier()
    for param_name in list(params.keys()):
        filter_data = np.zeros(params[param_name].shape).astype(params[param_name].dtype)
        if len(filter_data.shape) > 1:
            initializer("weight", filter_data)
        else:
            initializer("bias", filter_data)
        params[param_name] = tvm.nd.array(filter_data)

    return net, params, {"data": (input_shape, dtype)}, output_shape


executor_type = tvm.testing.parameter("ge", "vm")


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "name",
    [
        "resnet-18",
        "resnet-34",
        "resnet-50",
        "inception_v3",
        "mobilenet",
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_network(remote, name, dtype, target, executor_type):
    print("Network evaluating .. " + name + " " + dtype)
    np.random.seed(0)
    mod, params, inputs, _ = get_network(name, 1, dtype=dtype)
    input_data = {}

    for name, (shape, dtype) in inputs.items():
        input_data[name] = np.random.uniform(-1.0, 1.0, shape).astype(dtype)

    outputs = _build_and_run_network(remote, mod, params, input_data, target, executor_type)
    opencl_sort = np.argsort(outputs[1].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0].asnumpy()).flatten()
    tvm.testing.assert_allclose(opencl_sort[-5:], clml_sort[-5:], rtol=0, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
