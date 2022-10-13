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
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec": f"{BASE}/mxnet-val_256_q90.rec",
    "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel": f"{BASE}/bvlc_alexnet.caffemodel",
    "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel": f"{BASE}/bvlc_googlenet.caffemodel",
    "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz": f"{BASE}/tf-mobilenet_v1_1.0_224.tgz",
    "http://images.cocodataset.org/zips/val2017.zip": f"{BASE}/cocodataset-val2017.zip",
    "https://bj.bcebos.com/x2paddle/models/paddle_resnet50.tar": f"{BASE}/bcebos-paddle_resnet50.tar",
    "https://data.deepai.org/stanfordcars.zip": f"{BASE}/deepai-stanfordcars.zip",
    "https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel": f"{BASE}/2022-10-05/MobileNet.mlmodel",
    "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth": f"{BASE}/2022-10-05/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
    "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth": f"{BASE}/2022-10-05/mobilenet_v2-b0353104.pth",
    "https://download.pytorch.org/models/resnet18-f37072fd.pth": f"{BASE}/2022-10-05/resnet18-f37072fd.pth",
    "https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx": f"{BASE}/2022-10-05/super_resolution_0.2.onnx",
    "https://gist.githubusercontent.com/zhreshold/4d0b62f3d01426887599d4f7ede23ee5/raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt": f"{BASE}/2022-10-05/imagenet1000_clsid_to_human.txt",
    "https://github.com/dmlc/web-data/blob/main/darknet/data/dog.jpg": f"{BASE}/dog.jpg",
    "https://github.com/dmlc/web-data/blob/main/gluoncv/detection/street_small.jpg?raw=true": f"{BASE}/2022-10-05/small_street_raw.jpg",
    "https://github.com/dmlc/web-data/raw/main/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/gluon-small-stree.jpg",
    "https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5": f"{BASE}/2022-10-05/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/mnist/model/mnist-1.onnx": f"{BASE}/onnx/mnist-1.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx": f"{BASE}/2022-10-05/resnet50-v2-7.onnx",
    "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true": f"{BASE}/2022-10-05/yolov3-tiny-raw.cfg",
    "https://github.com/uwsampl/web-data/raw/main/vta/models/synset.txt": f"{BASE}/2022-10-05/synset.txt",
    "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_cora.torch": f"{BASE}/gcn_cora.torch",
    "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg": f"{BASE}/vta_cat.jpg",
    "https://objects.githubusercontent.com/github-production-release-asset-2e65be/130932608/4b196a8a-4e2d-11e8-9a11-be3c41846711?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20221004%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221004T170456Z&X-Amz-Expires=300&X-Amz-Signature=0602b68e8864b9b01c9142eee22aed3543fe98a5482686eec33d98e2617a2295&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=130932608&response-content-disposition=attachment%3B%20filename%3Dmobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5&response-content-type=application%2Foctet-stream": f"{BASE}/2022-10-05/aws-mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip": f"{BASE}/oneflow/resnet18.zip",
    "https://people.linaro.org/~tom.gall/sine_model.tflite": f"{BASE}/sine_model.tflite",
    "https://pjreddie.com/media/files/yolov3-tiny.weights?raw=true": f"{BASE}/yolov3-tiny.weights",
    "https://pjreddie.com/media/files/yolov3.weights": f"{BASE}/yolov3.weights",
    "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_classes.txt": f"{BASE}/2022-10-05/imagenet_classes.txt",
    "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_synsets.txt": f"{BASE}/2022-10-05/imagenet_synsets.txt",
    "https://raw.githubusercontent.com/dmlc/web-data/main/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/small_street.jpg",
    "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/street_small.jpg",
    "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt": f"{BASE}/2022-10-05/labels_mobilenet_quant_v1_224.txt",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/mali_v0.06.log": f"{BASE}/2022-10-05/mali_v0.06.log",
    "https://s3.amazonaws.com/model-server/inputs/kitten.jpg": f"{BASE}/2022-10-05/kitten.jpg",
    "https://s3.amazonaws.com/onnx-model-zoo/synset.txt": f"{BASE}/2022-10-05/synset-s3.txt",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz": f"{BASE}/2022-10-05/mobilenet_v2_1.0_224_quant.tgz",
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_128_tf.h5": f"{BASE}/2022-10-05/mobilenet_2_5_128_tf.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5": f"{BASE}/2022-10-05/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
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
