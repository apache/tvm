#!/usr/bin/env python3
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

import argparse
import logging
import shutil
from os import path as osp

import numpy as np
import torch
import torchvision
import tvm
from PIL import Image
from tvm import relay, runtime
from tvm.contrib import cc, graph_executor
from tvm.contrib.download import download_testdata

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Resnet build example")
aa = parser.add_argument
aa("--build-dir", type=str, required=True, help="directory to put the build artifacts")
aa("--batch-size", type=int, default=1, help="input image batch size")
aa(
    "--opt-level",
    type=int,
    default=3,
    help="level of optimization. 0 is unoptimized and 3 is the highest level",
)
aa("--target", type=str, default="llvm", help="target for compilation")
aa("--image-shape", type=str, default="3,224,224", help="input image dimensions")
aa("--image-name", type=str, default="cat.png", help="name of input image to download")
args = parser.parse_args()

build_dir = args.build_dir
batch_size = args.batch_size
opt_level = args.opt_level
target = tvm.target.create(args.target)
image_shape = tuple(map(int, args.image_shape.split(",")))
data_shape = (batch_size,) + image_shape


def build(target_dir):
    """Compiles resnet18 with TVM"""
    # Download the pretrained model from Torchvision.
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    torch_model = torchvision.models.resnet18(weights=weights).eval()

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch_model, input_data)
    input_infos = [("data", input_data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, input_infos)

    # Add softmax to do classification in last layer.
    func = mod["main"]
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    # save the model artifacts
    deploy_lib = osp.join(target_dir, "deploy_lib.o")
    lib.save(deploy_lib)
    cc.create_shared(osp.join(target_dir, "deploy_lib.so"), [osp.join(target_dir, "deploy_lib.o")])

    with open(osp.join(target_dir, "deploy_graph.json"), "w") as fo:
        fo.write(graph)

    with open(osp.join(target_dir, "deploy_param.params"), "wb") as fo:
        fo.write(runtime.save_param_dict(params))


def download_img_labels():
    """Download an image and imagenet1k class labels for test"""

    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "synset.txt"
    synset_path = download_testdata(synset_url, synset_name + ".raw", module="data", overwrite=True)

    with open(synset_path) as fin:
        data = fin.read()
        synset = eval(data)

    with open(synset_name, "w") as f:
        for key in synset:
            f.write(synset[key])
            f.write("\n")

    print(synset_path)
    print(synset_name)

    return synset


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def get_cat_image():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    shutil.copyfile(img_path, "cat.png")
    img = Image.open(img_path).resize((224, 224))
    return transform_image(img)


def test_build(build_dir):
    """Sanity check with the cat image we download."""
    graph = open(osp.join(build_dir, "deploy_graph.json")).read()
    lib = tvm.runtime.load_module(osp.join(build_dir, "deploy_lib.so"))
    params = bytearray(open(osp.join(build_dir, "deploy_param.params"), "rb").read())
    input_data = get_cat_image()
    dev = tvm.cpu()
    module = graph_executor.create(graph, lib, dev)
    module.load_params(params)
    module.run(data=input_data)
    out = module.get_output(0).numpy()
    top1 = np.argmax(out[0])
    synset = download_img_labels()
    print("TVM prediction top-1:", top1, synset[top1])


if __name__ == "__main__":
    logger.info("Compiling the model to graph executor.")
    build(build_dir)
    logger.info("Testing the model's predication on test data.")
    test_build(build_dir)
