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

# FP16 TRT command to run : TVM_TENSORRT_USE_FP16=1 python test_trt.py
# INT8 TRT command to run : TVM_TENSORRT_USE_INT8=1 TENSORRT_NUM_CALI_INT8=10 python test_trt.py
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

# PyTorch imports
import torch
import torchvision
from torchvision import transforms

# additonal imports
import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import distance

def cosine_distance(a, b):
    res = distance.cosine(a, b)
    return res

# you can change model name into resnet18, mobilenet_v2 ...  
model_name = "resnet34"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save("mobilenet_v2.pt")

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

print("python script started building --------------")
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
    lib = relay.build(mod, target=target, params=params)
print("python script finsihed building -------------------")


dtype = "float32"
lib.export_library('compiled.so')
loaded_lib = tvm.runtime.load_module('compiled.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

num_cali_int8 = 0
try:
    num_cali_int8 = os.environ["TENSORRT_NUM_CALI_INT8"]
    print("we are going to set {} times calibration in this case".format(num_cali_int8))
except:
    print("no TENSORRT_NUM_CALI_INT8 found in this case ... ")

num_cali_int8 = int(num_cali_int8)
if num_cali_int8 != 0:
    print("calibration steps ... ")
    for i in range(num_cali_int8):
        tvm_data = tvm.nd.array(img)
        gen_module.set_input(input_name, tvm_data)
        gen_module.run(data=tvm_data)
    print("finished calibration step")


# get output of tvm model 
print("test run ... ")
tvm_data = tvm.nd.array(img)
gen_module.set_input(input_name, tvm_data)
gen_module.run(data=tvm_data)
out = gen_module.get_output(0)


# check output of tvm and output of pytorch model are equal
data_np = img
device = "cuda:1"
torch_data = torch.from_numpy(data_np)
torch_data = torch_data.to(device)
model = scripted_model
model = model.eval()
model = model.to(device)
torch_output = model(torch_data)

print("the cosine distance between torch output and trt int8 output of tvm : ")
cosine_distance_res = cosine_distance(out.numpy(), torch_output.detach().cpu().numpy())
print(cosine_distance_res)

# Evaluate
print("Evaluate inference time cost...")
ftimer = gen_module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
message = "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
print(message)