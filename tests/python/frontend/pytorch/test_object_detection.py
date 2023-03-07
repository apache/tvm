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
import cv2

import torch
import torchvision

import tvm

import tvm.testing
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.relay.frontend.pytorch_utils import (
    rewrite_nms_to_batched_nms,
    rewrite_batched_nms_with_max_out_size,
    rewrite_scatter_to_gather,
)
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
    model = TraceWrapper(model_func(pretrained=True, rpn_pre_nms_top_n_test=1000))

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
        "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img)

    input_shape = (1, 3, in_size, in_size)

    input_name = "input0"
    shape_list = [(input_name, input_shape)]

    scripted_model = generate_jit_model(1)
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    with tvm.testing.enable_span_filling():
        mod_with_span, _ = relay.frontend.from_pytorch(scripted_model, shape_list)
    assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

    data = process_image(img)
    data_np = data.detach().numpy()

    with torch.no_grad():
        pt_res = scripted_model(data)

    def compile_and_run_vm(mod, params, data_np, target):
        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

        dev = tvm.device(target, 0)
        vm = VirtualMachine(vm_exec, dev)
        vm.set_input("main", **{input_name: data_np})
        return vm.run()

    for target in ["llvm"]:
        tvm_res = compile_and_run_vm(mod, params, data_np, target)

        # Bounding boxes
        tvm.testing.assert_allclose(
            pt_res[0].cpu().numpy(), tvm_res[0].numpy(), rtol=1e-5, atol=1e-5
        )
        # Scores
        tvm.testing.assert_allclose(
            pt_res[1].cpu().numpy(), tvm_res[1].numpy(), rtol=1e-5, atol=1e-5
        )
        # Class ids
        np.testing.assert_equal(pt_res[2].cpu().numpy(), tvm_res[2].numpy())

        score_threshold = 0.9
        print("Num boxes:", pt_res[0].cpu().numpy().shape[0])
        print("Num valid boxes:", np.sum(pt_res[1].cpu().numpy() >= score_threshold))

    before = mod["main"]
    mod = rewrite_nms_to_batched_nms(mod)
    after = mod["main"]
    assert not tvm.ir.structural_equal(after, before)

    # TODO(masahi): It seems this rewrite causes flaky segfaults on CI
    # See https://github.com/apache/tvm/issues/7363
    # before = mod["main"]
    # mod = rewrite_batched_nms_with_max_out_size(mod)
    # after = mod["main"]
    # assert not tvm.ir.structural_equal(after, before)

    before = mod["main"]
    mod = rewrite_scatter_to_gather(mod, 4)  # num_scales is 4 for maskrcnn_resnet50_fpn
    after = mod["main"]
    assert not tvm.ir.structural_equal(after, before)

    tvm_res_after_rewrite = compile_and_run_vm(mod, params, data_np, "llvm")

    # Results should be equivalent after rewriting
    for res1, res2 in zip(tvm_res, tvm_res_after_rewrite):
        tvm.testing.assert_allclose(res1.numpy(), res2.numpy())
