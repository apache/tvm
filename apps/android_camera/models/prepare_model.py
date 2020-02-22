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

import logging

import mxnet as mx
import tvm
import nnvm.frontend
import nnvm.compiler
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from tvm import relay
from tvm.contrib import ndk
import os


target_host =  'llvm -target=arm64-linux-android'

def get_model(model_name, batch_size=1):
    gluon_model = vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    with relay.build_config(opt_level=3):
        func = relay.optimize(net, target=None, params=params)
    return func

def get_model_nnvm(model_name, batch_size=1):
    gluon_model = vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    sym, params = nnvm.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return sym, params

def main_nnvm(model_str):
    print(model_str)
    print("getting model...")
    sym, params = get_model_nnvm(model_str)
    try:
        os.mkdir(model_str)
    except FileExistsError:
        pass
    target = tvm.target.arm_cpu(model='pixel2')
    print("building model...")
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(sym, target, {"data": (1, 3,
                                                 224, 224)}, params=params, target_host=None)
    print("dumping lib...")
    lib.export_library(model_str + '/' + 'deploy_lib_cpu.so', ndk.create_shared)
    print("dumping graph...")
    with open(model_str + '/' + 'deploy_graph.json', 'w') as f:
        f.write(graph.json())
    print("dumping params...")
    with open (model_str + '/' + 'deploy_param.params', 'wb') as f:
        f.write(nnvm.compiler.save_param_dict(params))

def main(model_str):
    print(model_str)
    print("getting model...")
    func = get_model(model_str)
    try:
        os.mkdir(model_str)
    except FileExistsError:
        pass
    print("building...")
    target = tvm.target.arm_cpu(model='pixel2')
    print("(relay)")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, target_host=target_host)
    print("dumping lib...")
    lib.export_library(model_str + '/' + 'deploy_lib_cpu.so', ndk.create_shared)
    print("dumping graph...")
    with open(model_str + '/' + 'deploy_graph.json', 'w') as f:
        f.write(graph)
    print("dumping params...")
    with open(model_str + '/' + 'deploy_param.params', 'wb') as f:
        f.write(relay.save_param_dict(params))

if __name__ == '__main__':
    models = ['resnet18_v1']
    for model in models:
        main(model)
