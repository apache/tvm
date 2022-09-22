#!/usr/bin/env bash

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

import urllib.request
import logging

LOGGER = None


# To update this list, run the workflow <HERE> with the URL to download and the SHA512 of the file
BASE = "https://tvm-ci-resources.s3.us-west-2.amazonaws.com"
URL_MAP = {
    "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip": f"{BASE}/oneflow/resnet18.zip",
    "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_cora.torch": f"{BASE}/gcn_cora.torch",
    "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg": f"{BASE}/vta_cat.jpg",
    "https://people.linaro.org/~tom.gall/sine_model.tflite": f"{BASE}/sine_model.tflite",
    "https://pjreddie.com/media/files/yolov3-tiny.weights?raw=true": f"{BASE}/yolov3-tiny.weights",
    "https://pjreddie.com/media/files/yolov3.weights": f"{BASE}/yolov3.weights",
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec": f"{BASE}/mxnet-val_256_q90.rec",
    "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz": f"{BASE}/tf-mobilenet_v1_1.0_224.tgz",
    "http://images.cocodataset.org/zips/val2017.zip": f"{BASE}/cocodataset-val2017.zip",
    "https://bj.bcebos.com/x2paddle/models/paddle_resnet50.tar": f"{BASE}/bcebos-paddle_resnet50.tar",
    "https://data.deepai.org/stanfordcars.zip": f"{BASE}/deepai-stanfordcars.zip",
    "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel": f"{BASE}/bvlc_alexnet.caffemodel",
    "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel": f"{BASE}/bvlc_googlenet.caffemodel",
    "https://github.com/dmlc/web-data/blob/main/darknet/data/dog.jpg": f"{BASE}/dog.jpg",
    "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/mnist/model/mnist-1.onnx": f"{BASE}/onnx/mnist-1.onnx",
}


class TvmRequestHook(urllib.request.Request):
    def __init__(self, url, *args, **kwargs):
        LOGGER.info(f"Caught access to {url}")
        if url in URL_MAP:
            new_url = URL_MAP[url]
            LOGGER.info(f"Mapped URL {url} to {new_url}")
        else:
            new_url = url
        super().__init__(new_url, *args, **kwargs)


def init():
    global LOGGER
    urllib.request.Request = TvmRequestHook
    LOGGER = logging.getLogger("tvm_request_hook")
    LOGGER.setLevel(logging.DEBUG)
    fh = logging.FileHandler("redirected_urls.log")
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)
