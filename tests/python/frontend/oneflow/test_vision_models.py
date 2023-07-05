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
# pylint: disable=import-self, invalid-name
# pylint: disable=arguments-differ, unused-argument
"""Unit tests for various models and operators"""
import os

import numpy as np
import oneflow as flow
from flowvision.models.alexnet import alexnet
from flowvision.models.squeezenet import squeezenet1_0
from flowvision.models.shufflenet_v2 import shufflenet_v2_x0_5
from flowvision.models.mobilenet import mobilenet_v2
from flowvision.models.ghostnet import ghostnet
from flowvision.models.vision_transformer import vit_base_patch16_224
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay

MODEL_HOME = "test_model"


def mkdir(path):
    # init
    path = path.strip()
    path = path.rstrip("\\")

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"{path} is already here")


def rmdir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.removedirs(path)


def assert_shape(out1, out2):
    if out1.shape != out2.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(out1.shape, out2.shape))


class OneFlowGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


def get_oneflow_output(model, inputs):
    flow_output = model(inputs)
    return flow_output.numpy()


def get_tvm_output(graph, model_path, inputs: flow.tensor, target="llvm", dtype="float32"):
    """Generic function to execute and get tvm output"""
    inputs_numpy = inputs.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(tvm.nd.array(inputs_numpy.astype(dtype)), **params).numpy()
    return tvm_output


def verify_model(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """Generic function to generate and compare oneflow and TVM output"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_vision_models():
    """Vision models test"""

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    vision_alexnet = alexnet().eval()
    vision_squeezenet = squeezenet1_0().eval()
    vision_shufflenet = shufflenet_v2_x0_5().eval()
    vision_mobilenetv2 = mobilenet_v2().eval()
    vision_ghostnet = ghostnet().eval()
    vision_vit = vit_base_patch16_224().eval()

    for device in ["llvm"]:
        verify_model(vision_alexnet, device=device)
        verify_model(vision_squeezenet, device=device)
        verify_model(vision_shufflenet, device=device)
        verify_model(vision_mobilenetv2, device=device)
        verify_model(vision_ghostnet, device=device)
        verify_model(vision_vit, device=device)


if __name__ == "__main__":
    test_vision_models()
    rmdir("log")
