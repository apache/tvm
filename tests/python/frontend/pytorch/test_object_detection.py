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
# pylint: disable=import-self, invalid-name, unused-argument
"""Test torch vision fasterrcnn and maskrcnn models"""
import numpy as np
import torch
import torchvision
import cv2

import tvm

from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download


in_size = 300


def process_image(img):
    img = cv2.imread(img).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    img = torch.unsqueeze(img, axis=0)

    return img


def do_trace(model, inp, in_size=in_size):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def generate_jit_model(index):
    model_funcs = [
        torchvision.models.detection.fasterrcnn_resnet50_fpn,
        torchvision.models.detection.maskrcnn_resnet50_fpn,
    ]

    model_func = model_funcs[index]
    model = TraceWrapper(model_func(pretrained=True))

    model.eval()
    inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

    with torch.no_grad():
        out = model(inp)

        script_module = do_trace(model, inp)
        script_out = script_module(inp)

        assert len(out[0]) > 0 and len(script_out[0]) > 0
        return script_module


def test_detection_models():
    img = "test_street_small.jpg"
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/"
        "master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img)

    input_shape = (1, 3, in_size, in_size)
    target = "llvm"
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    score_threshold = 0.9

    scripted_model = generate_jit_model(1)
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        vm_exec = relay.vm.compile(mod, target=target, params=params)

    ctx = tvm.cpu()
    vm = VirtualMachine(vm_exec, ctx)
    data = process_image(img)
    pt_res = scripted_model(data)
    data = data.detach().numpy()
    vm.set_input("main", **{input_name: data})
    tvm_res = vm.run()

    # Note: due to accumulated numerical error, we can't directly compare results
    # with pytorch output. Some boxes might have a quite tiny difference in score
    # and the order can become different. We just measure how many valid boxes
    # there are for input image.
    pt_scores = pt_res[1].detach().numpy().tolist()
    tvm_scores = tvm_res[1].asnumpy().tolist()
    num_pt_valid_scores = num_tvm_valid_scores = 0

    for score in pt_scores:
        if score >= score_threshold:
            num_pt_valid_scores += 1
        else:
            break

    for score in tvm_scores:
        if score >= score_threshold:
            num_tvm_valid_scores += 1
        else:
            break

    assert num_pt_valid_scores == num_tvm_valid_scores, (
        "Output mismatch: Under score threshold {}, Pytorch has {} valid "
        "boxes while TVM has {}.".format(score_threshold, num_pt_valid_scores, num_tvm_valid_scores)
    )
