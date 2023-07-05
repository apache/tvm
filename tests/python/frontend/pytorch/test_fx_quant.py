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
""" Tests on fx-quantized torch model conversion """
import torch
import torchvision
import pytest
import numpy as np
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision.models.efficientnet import efficientnet_b4
from torchvision.models.resnet import resnet50
from tvm import relay
import tvm.testing


def quantize(model, example_inputs):
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    return convert_fx(prepare_fx(model, qconfig_dict, example_inputs))


def quantize_and_build(model, in_size):
    inp = torch.rand(1, 3, in_size, in_size)
    input_name = "inp"
    qmodel = quantize(model, inp)

    with torch.no_grad():
        script_module = torch.jit.trace(qmodel, inp)
        with tvm.testing.disable_span_filling():
            mod, _ = relay.frontend.from_pytorch(script_module, [(input_name, inp.shape)])
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_pytorch(script_module, [(input_name, inp.shape)])
        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)
        mod = relay.transform.InferType()(mod)

        # Make sure that the model is quantized
        assert "qnn.conv2d" in mod.astext(show_meta_data=False)

        # Skip building since it is slow on CI
        # relay.build(mod, params=params, target="llvm")


@pytest.mark.skip(reason="unsupported op aten::linalg_vector_norm")
def test_ssd_vgg():
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            features = self.model.backbone(inp)
            features = list(features.values())
            out = self.model.head(features)
            return out["bbox_regression"], out["cls_logits"]

    model_func = torchvision.models.detection.ssd300_vgg16
    model = TraceWrapper(model_func(num_classes=50, pretrained_backbone=True)).eval()
    quantize_and_build(model, 300)


def test_deeplab_v3():
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            return out["out"]

    deeplabv3 = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model = TraceWrapper(deeplabv3.eval()).eval()
    quantize_and_build(model, 300)


def test_imagenet():
    for model_func in [resnet50, efficientnet_b4]:
        quantize_and_build(model_func(pretrained=True).eval(), 224)
