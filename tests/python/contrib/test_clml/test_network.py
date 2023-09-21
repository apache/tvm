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
from test_clml.infrastructure import build_and_run, Device
import pytest


def _build_and_run_network(mod, params, inputs, data, device, atol, rtol, tvm_log=""):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        outputs.append(
            build_and_run(mod, data, 1, params, device, enable_clml=clml, tune_log=tvm_log)[0][0]
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
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = net["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        net = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, {"data": (input_shape, dtype)}, output_shape

@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("name", [
                                "resnet-18",
                                "resnet-34",
                                "resnet-50",
                                # "vgg-16",
                                # "vgg-19",
                                "densenet-121",
                                "inception_v3",
                                "mobilenet",
                                "squeezenet_v1.0",
                                "squeezenet_v1.1",
])
@tvm.testing.requires_openclml
def test_network(device, name, dtype):
    print("Network evaluating .. " + name + " " + dtype)
    if device == None:
        device = Device()
    mod, params, inputs, _ = get_network(name, 1, dtype=dtype)
    input_data = {}
    np.random.seed(0)
    for name, (shape, dtype) in inputs.items():
        if dtype == "uint8":
            low, high = 0, 1
        else:
            low, high = -2, 1
        input_data[name] = np.random.uniform(low, high, shape).astype(dtype)
    outputs = _build_and_run_network(
        mod, params, inputs, input_data, device=device, atol=1e-5, rtol=1e-5
    )

    opencl_sort = np.argsort(outputs[1].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0].asnumpy()).flatten()
    tvm.testing.assert_allclose(opencl_sort[:10], clml_sort[:10], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    #networks = [
    #        "resnet-18",
    #        "resnet-34",
    #        "resnet-50",
    #        # "vgg-16",
    #        # "vgg-19",
    #        "densenet-121",
    #        "inception_v3",
    #        "mobilenet",
    #        "squeezenet_v1.0",
    #        "squeezenet_v1.1",
    #    ]
    #device = Device()
    #for name in networks:
    #    test_network(device, name, "float32")
    #    test_network(device, name, "float16")
    tvm.testing.main()
