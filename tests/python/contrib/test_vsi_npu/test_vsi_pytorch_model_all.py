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
import os
from PIL import Image
import numpy as np

import tvm
from tvm import relay, transform
from tvm import rpc
from tvm.contrib import utils as util
from tvm.relay.op.contrib import vsi_npu

# TODO(Sven) : this is workaround for new version of TVM
from tvm.contrib import graph_executor as graph_runtime
from tvm.contrib import graph_executor as runtime

import torch
import torch.nn as nn
import torchvision

np.set_printoptions(threshold=np.inf)
RPC_HOST = os.environ["RPC_HOST"]
RPC_PORT = int(os.environ["RPC_PORT"])
CROSS_CC = os.environ["CROSS_CC"]
ROOTFS = os.environ["ROOTFS"]
lib_name = os.environ["MOD_NAME"]
lib_path = os.environ["MOD_PATH"]

def get_ref_result(shape, model_path, image_data, input_tensor_name, DTYPE):
    inputs = input_tensor_name
    model = eval(model_path)
    scripted_model = torch.jit.trace(model, torch.from_numpy(image_data))
    scripted_model = scripted_model.eval()
    mod, params = relay.frontend.from_pytorch(
        scripted_model, [(inputs, shape)]
    )
    print(mod.astext())
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3,
                                   disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target, params=params)
    ctx = tvm.cpu()
    cpu_mod = graph_runtime.GraphModule(lib["default"](ctx))
    cpu_mod.set_input(inputs, tvm.nd.array(image_data))

    # if True:
    #     print("Evaluate graph runtime inference cost on CPU")
    #     ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=1)
    #     # Measure in millisecond.
    #     prof_res = np.array(ftimer().results) * 1000
    #     print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
    #           % (np.mean(prof_res), np.std(prof_res)))

    cpu_mod.run()
    ref_out = cpu_mod.get_output(0)
    return ref_out.asnumpy()

def compile_pytorch_model(shape, model_path, input_data, input_tensor_name, DTYPE):
    vsi_input_data = {
    input_tensor_name: tvm.nd.array(input_data),
    }
    model = eval(model_path)
    scripted_model = torch.jit.trace(model, torch.from_numpy(input_data))
    scripted_model = scripted_model.eval()
    mod, params = relay.frontend.from_pytorch(
        scripted_model, [(input_tensor_name, shape)]
    )
    print(mod.astext())
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    tmp_path = util.tempdir()
    lib_name = "model.so"
    lib_path = tmp_path.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = CROSS_CC
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    kwargs["options"] = ["-L"+ROOTFS+"/lib ", "-L" + ROOTFS+"/usr/lib ",
                         "-L" + ROOTFS+"/usr/lib/aarch64-poky-linux/9.2.0 ", "--sysroot=" + ROOTFS]
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    # remote = rpc.connect(RPC_HOST, RPC_PORT)
    # remote.upload(lib_path + lib_name)
    # lib = remote.load_module(lib_name)

    remote.upload(lib_path)
    lib = remote.load_module(lib_name)
    ctx = remote.cpu()

    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))

    # ctx = remote.cpu()
    # rt_mod = graph_runtime.GraphModule(lib["default"](ctx))
    rt_mod.set_input(**vsi_input_data)
    rt_mod.run()
    rf_output = rt_mod.get_output(0)
    return rf_output.asnumpy()

def print_top5(input):
    k = 5
    n = input.flatten()
    n_arg0 = np.argpartition(n, -k)[-k:]
    n = n[n_arg0]
    n_arg1 = np.argsort(n)[::-1]
    n_arg0 = n_arg0[n_arg1]
    for i in range(k):
        print("{} : {}".format(n_arg0[i], n[n_arg1[i]]))

def get_model(model_list, model_name):
    for model in model_list:
        if (model['name'] == model_name):
            return model

model_list = [
    {'name': "quantization_mobilenet_v2",
     'model': 'torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)',
     'shape': (1, 3, 224, 224),
     'input_tensor_name': 'input',
     'dtype': "float32"},
]

model_name = os.environ["PYTORCH_MODEL"]

model = get_model(model_list, model_name)
print (model)
shape = model['shape']
input_tensor_name = model['input_tensor_name']
DTYPE = model['dtype']
wait=input("press any key and continue...")

path = "./"
img = Image.open(path + "space_shuttle_224x224.jpg")
shape_nhwc = [shape[0], shape[2], shape[3], shape[1]]
img = img.resize((shape_nhwc[1], shape_nhwc[2]))
n1 = np.array(img)
#n1 = n1[:, :, 0] # pick one channel
#n1 = np.broadcast_to(n1, (4, 224, 224, 3)) # batch the image
n1 = n1.reshape(shape_nhwc)
n1 = np.transpose(n1, (0, 3, 1, 2))
n1 = (n1 - 114 ) / 255.0
input_data = n1.astype(DTYPE)

# input_data = np.ones(shape, DTYPE)


vsi_output = compile_pytorch_model(shape, model["model"], input_data, input_tensor_name, DTYPE)
ref_output = get_ref_result(shape, model["model"], input_data, input_tensor_name, DTYPE)
# vsi_output = compile_pytorch_model(shape, model["model"], vsi_input_data, input_tensor_name, DTYPE)

#print("ref_output:",ref_output)
#print("vsi_output",vsi_output)

if DTYPE == "uint8":
    tolerance = 5
else:
    tolerance = 1e-3

result = abs(vsi_output.astype("float32") - ref_output.astype("float32"))
result0 = result < tolerance

print("number of false number:", np.sum(result0 != True), np.max(result), np.argmax(result))
print("ratio of false number:", np.sum(result0 != True) / result0.size)

print("top5 of ref:")
print_top5(ref_output)

print("top5 of vsi:")
print_top5(vsi_output)

np.savetxt(path + "ref_output.txt", ref_output.flatten(), fmt='%.3f')
np.savetxt(path + "vsi_output.txt", vsi_output.flatten(), fmt='%.3f')
np.savetxt(path + "diff.txt", result.flatten(), fmt='%.3f')
