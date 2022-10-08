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
import torch
from torchvision.models import resnet
from torchvision.models.quantization import resnet as qresnet

import tvm
from tvm import relay


def export_resnet50_fp16():
    model = resnet.resnet50(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)

    script_module = torch.jit.trace(model, pt_inp).eval()

    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    with open("resnet50_fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("resnet50_fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def export_resnet50_int8():
    def quantize_model(model, inp):
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        model(inp)
        torch.quantization.convert(model, inplace=True)

    model = qresnet.resnet50(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)
    quantize_model(model, pt_inp)

    script_module = torch.jit.trace(model, pt_inp).eval()

    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(
        script_module, input_shapes, keep_quantized_weight=True
    )

    with open("resnet50_int8.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("resnet50_int8.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


if __name__ == "__main__":
    export_resnet50_fp16()
    export_resnet50_int8()
