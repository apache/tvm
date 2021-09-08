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
import pytest
import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import distance

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

# PyTorch imports
import torch
import torchvision
from torchvision import transforms


def cosine_distance(a, b):
    res = distance.cosine(a, b)
    return res

def test_trt_int8():
    """
    This Function is used to use tensorrt int8 to compile a resnet34 model,
    and compare cosine distance between the output of the original model and trt int8 tvm ouput

    """
    os.environ["TVM_TENSORRT_USE_INT8"] = "1"
    os.environ["TENSORRT_NUM_CALI_INT8"] = "10"
    model_name = "resnet34"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)

    input_name = "input0"
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # compile the model
    target = "cuda"
    dev = tvm.cuda(1)
    mod, config = partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
        lib = relay.build(mod, target=target, params=params)


    dtype = "float32"
    gen_module = tvm.contrib.graph_executor.GraphModule(lib['default'](dev))

    num_cali_int8 = int(os.environ["TENSORRT_NUM_CALI_INT8"])
    if num_cali_int8 != 0:
        print("start calibrating data ... ")
        for i in range(num_cali_int8):
            tvm_data = tvm.nd.array(img)
            gen_module.set_input(input_name, tvm_data)
            gen_module.run(data=tvm_data)
        print("finished calibrating data ... ")


    # get output of tvm model 
    print("rebuild engine and test to run ... ")
    tvm_data = tvm.nd.array(img)
    gen_module.set_input(input_name, tvm_data)
    gen_module.run(data=tvm_data)
    out = gen_module.get_output(0)


    # check output of tvm and output of pytorch model are equal
    torch_data = torch.from_numpy(img)
    model = scripted_model.eval()
    torch_output = model(torch_data)

    cosine_distance_res = cosine_distance(out.numpy(), torch_output.detach().cpu().numpy())
    assert cosine_distance_res <= 0.01

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = gen_module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    message = "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
    print(message)

if __name__ == "__main__":
    test_trt_int8()
    # pytest.main([__file__])
