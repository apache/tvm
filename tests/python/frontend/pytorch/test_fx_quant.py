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
import numpy as np
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from tvm import relay


def do_trace(model, in_size=500):
    model_trace = torch.jit.trace(model, torch.rand(1, 3, in_size, in_size))
    model_trace.eval()
    return model_trace


def quantize(model_fp):
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    return convert_fx(prepare_fx(model_fp, qconfig_dict))


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

    model = quantize(model)

    in_size = 500
    inp = torch.rand(1, 3, in_size, in_size)
    input_name = "inp"

    with torch.no_grad():
        script_module = do_trace(model, in_size)
        mod, params = relay.frontend.from_pytorch(script_module, [(input_name, inp.shape)])

    print(relay.transform.InferType()(mod))


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
    inp = torch.rand(8, 3, 512, 512)

    qmodel = quantize(model)

    with torch.no_grad():
        trace = torch.jit.trace(qmodel, inp)

    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])
    print(relay.transform.InferType()(mod))


def test_imagenet():
    from torchvision.models.efficientnet import efficientnet_b4
    from torchvision.models.resnet import resnet50

    for model_func in [resnet50, efficientnet_b4]:
        model = efficientnet_b4(pretrained=True).eval()
        model = quantize(model)

        x = torch.rand((1, 3, 224, 224))
        model_traced = torch.jit.trace(model, x).eval()

        mod, _ = relay.frontend.from_pytorch(model_traced, [("x", x.shape)])
        print(relay.transform.InferType()(mod))
