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

from urllib.parse import quote

LOGGER = None


# To update this list, run https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml
# with the URL to download and the SHA-256 hash of the file.
BASE = "https://tvm-ci-resources.s3.us-west-2.amazonaws.com"
URL_MAP = {
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec": f"{BASE}/mxnet-val_256_q90.rec",
    "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel": f"{BASE}/bvlc_alexnet.caffemodel",
    "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel": f"{BASE}/bvlc_googlenet.caffemodel",
    "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz": f"{BASE}/tf-mobilenet_v1_1.0_224.tgz",
    "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz": f"{BASE}/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz",
    "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz": f"{BASE}/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
    "http://images.cocodataset.org/zips/val2017.zip": f"{BASE}/cocodataset-val2017.zip",
    "http://pjreddie.com/media/files/alexnet.weights?raw=true": f"{BASE}/media/files/alexnet.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/alexnet.weights?raw=true": f"{BASE}/media/files/alexnet.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/extraction.weights?raw=true": f"{BASE}/media/files/extraction.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/extraction.weights?raw=true": f"{BASE}/media/files/extraction.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/resnet50.weights?raw=true": f"{BASE}/media/files/resnet50.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/resnext50.weights?raw=true": f"{BASE}/media/files/resnext50.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/yolov2.weights?raw=true": f"{BASE}/media/files/yolov2.weights"
    + quote("?raw=true"),
    "http://pjreddie.com/media/files/yolov3.weights?raw=true": f"{BASE}/media/files/yolov3.weights"
    + quote("?raw=true"),
    "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz": f"{BASE}/imikolov/rnnlm/simple-examples.tgz",
    "https://bj.bcebos.com/x2paddle/models/paddle_resnet50.tar": f"{BASE}/bcebos-paddle_resnet50.tar",
    "https://data.deepai.org/stanfordcars.zip": f"{BASE}/deepai-stanfordcars.zip",
    "https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth": f"{BASE}/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
    "https://github.com/ARM-software/ML-zoo/blob/48f458af1e9065d9aad2ad94d24b58d6e7c00817/models/keyword_spotting/ds_cnn_small/tflite_int16/ds_cnn_quantized.tflite?raw=true": f"{BASE}/ARM-software/ML-zoo/blob/48f458af1e9065d9aad2ad94d24b58d6e7c00817/models/keyword_spotting/ds_cnn_small/tflite_int16/ds_cnn_quantized.tflite"
    + quote("?raw=true"),
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/adreno_v0.01.log": f"{BASE}/tlc-pack/tophub/main/tophub/adreno_v0.01.log",
    "https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel": f"{BASE}/2022-10-05/MobileNet.mlmodel",
    "https://docs-assets.developer.apple.com/coreml/models/Resnet50.mlmodel": f"{BASE}/coreml/models/Resnet50.mlmodel",
    "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth": f"{BASE}/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth": f"{BASE}/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "https://download.pytorch.org/models/densenet121-a639ec97.pth": f"{BASE}/models/densenet121-a639ec97.pth",
    "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth": f"{BASE}/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth": f"{BASE}/models/fcn_resnet101_coco-7ecb50ca.pth",
    "https://download.pytorch.org/models/googlenet-1378be20.pth": f"{BASE}/models/googlenet-1378be20.pth",
    "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth": f"{BASE}/models/inception_v3_google-0cc3c7bd.pth",
    "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth": f"{BASE}/2022-10-05/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
    "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth": f"{BASE}/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth": f"{BASE}/2022-10-05/mobilenet_v2-b0353104.pth",
    "https://download.pytorch.org/models/r3d_18-b3b3357e.pth": f"{BASE}/models/r3d_18-b3b3357e.pth",
    "https://download.pytorch.org/models/resnet18-f37072fd.pth": f"{BASE}/2022-10-05/resnet18-f37072fd.pth",
    "https://download.pytorch.org/models/resnet50-0676ba61.pth": f"{BASE}/models/resnet50-0676ba61.pth",
    "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth": f"{BASE}/models/squeezenet1_0-b66bff10.pth",
    "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth": f"{BASE}/models/squeezenet1_1-b8a52dc0.pth",
    "https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth": f"{BASE}/models/vgg16_features-amdegroot-88682ab5.pth",
    "https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx": f"{BASE}/2022-10-05/super_resolution_0.2.onnx",
    "https://gist.githubusercontent.com/zhreshold/4d0b62f3d01426887599d4f7ede23ee5/raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt": f"{BASE}/2022-10-05/imagenet1000_clsid_to_human.txt",
    "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png": f"{BASE}/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png",
    "https://github.com/ARM-software/ML-zoo/raw/48a22ee22325d15d2371a6df24eb7d67e21dcc97/models/keyword_spotting/cnn_small/tflite_int8/cnn_s_quantized.tflite": f"{BASE}/ARM-software/ML-zoo/raw/48a22ee22325d15d2371a6df24eb7d67e21dcc97/models/keyword_spotting/cnn_small/tflite_int8/cnn_s_quantized.tflite",
    "https://github.com/czh978/models_for_tvm_test/raw/main/tflite_graph_with_postprocess.pb": f"{BASE}/czh978/models_for_tvm_test/raw/main/tflite_graph_with_postprocess.pb",
    "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true": f"{BASE}/dmlc/mxnet.js/blob/main/data/cat.png"
    + quote("?raw=true"),
    "https://github.com/dmlc/mxnet.js/raw/main/data/cat.png": f"{BASE}/dmlc/mxnet.js/raw/main/data/cat.png",
    "https://github.com/dmlc/web-data/blob/main/darknet/cfg/yolov3.cfg?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/cfg/yolov3.cfg"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/darknet/data/arial.ttf?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/data/arial.ttf"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/darknet/data/coco.names?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/data/coco.names"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/darknet/data/dog.jpg?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/data/dog.jpg"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/darknet/data/dog.jpg": f"{BASE}/dog.jpg",
    "https://github.com/dmlc/web-data/blob/main/darknet/data/person.jpg?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/data/person.jpg"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/darknet/lib/libdarknet2.0.so?raw=true": f"{BASE}/dmlc/web-data/blob/main/darknet/lib/libdarknet2.0.so"
    + quote("?raw=true"),
    "https://github.com/dmlc/web-data/blob/main/gluoncv/detection/street_small.jpg?raw=true": f"{BASE}/2022-10-05/small_street_raw.jpg",
    "https://github.com/dmlc/web-data/raw/main/darknet/cfg/yolov3.cfg": f"{BASE}/dmlc/web-data/raw/main/darknet/cfg/yolov3.cfg",
    "https://github.com/dmlc/web-data/raw/main/darknet/data/arial.ttf": f"{BASE}/dmlc/web-data/raw/main/darknet/data/arial.ttf",
    "https://github.com/dmlc/web-data/raw/main/darknet/data/coco.names": f"{BASE}/dmlc/web-data/raw/main/darknet/data/coco.names",
    "https://github.com/dmlc/web-data/raw/main/darknet/data/dog.jpg": f"{BASE}/dmlc/web-data/raw/main/darknet/data/dog.jpg",
    "https://github.com/dmlc/web-data/raw/main/darknet/data/person.jpg": f"{BASE}/dmlc/web-data/raw/main/darknet/data/person.jpg",
    "https://github.com/dmlc/web-data/raw/main/darknet/lib/libdarknet2.0.so": f"{BASE}/dmlc/web-data/raw/main/darknet/lib/libdarknet2.0.so",
    "https://github.com/dmlc/web-data/raw/main/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/gluon-small-stree.jpg",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/Custom/placeholder.pb": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/Custom/placeholder.pb",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/classify_image_graph_def-with_shapes.pb": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/classify_image_graph_def-with_shapes.pb",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/ResnetV2/resnet-20180601_resnet_v2_imagenet-shapes.pb": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/ResnetV2/resnet-20180601_resnet_v2_imagenet-shapes.pb",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/elephant-299.jpg": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/elephant-299.jpg",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/imagenet_2012_challenge_label_map_proto.pbtxt": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/imagenet_2012_challenge_label_map_proto.pbtxt",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/imagenet_synset_to_human_label_map.txt": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/imagenet_synset_to_human_label_map.txt",
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/RNN/ptb/ptb_model_with_lstmblockcell.pb": f"{BASE}/dmlc/web-data/raw/main/tensorflow/models/RNN/ptb/ptb_model_with_lstmblockcell.pb",
    "https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/elephant-299.jpg": f"{BASE}/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/elephant-299.jpg",
    "https://github.com/fernchen/CaffeModels/raw/master/resnet/ResNet-50-deploy.prototxt": f"{BASE}/fernchen/CaffeModels/raw/master/resnet/ResNet-50-deploy.prototxt",
    "https://github.com/fernchen/CaffeModels/raw/master/resnet/ResNet-50-deploy.prototxt": f"{BASE}/fernchen/CaffeModels/raw/master/resnet/ResNet-50-deploy.prototxt",
    "https://github.com/fernchen/CaffeModels/raw/master/resnet/ResNet-50-model.caffemodel": f"{BASE}/fernchen/CaffeModels/raw/master/resnet/ResNet-50-model.caffemodel",
    "https://github.com/google/mediapipe/raw/v0.7.4/mediapipe/models/hand_landmark.tflite": f"{BASE}/google/mediapipe/raw/v0.7.4/mediapipe/models/hand_landmark.tflite",
    "https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5": f"{BASE}/2022-10-05/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/mnist/model/mnist-1.onnx": f"{BASE}/onnx/mnist-1.onnx",
    "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/resnet/model/resnet50-v2-7.onnx": f"{BASE}/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/vision/classification/resnet/model/resnet50-v2-7.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx": f"{BASE}/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx": f"{BASE}/2022-10-05/resnet50-v2-7.onnx",
    "https://github.com/pjreddie/darknet/blob/master/cfg/alexnet.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/alexnet.cfg"
    + quote("?raw=true"),
    "https://github.com/pjreddie/darknet/blob/master/cfg/extraction.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/extraction.cfg"
    + quote("?raw=true"),
    "https://github.com/pjreddie/darknet/blob/master/cfg/resnet50.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/resnet50.cfg"
    + quote("?raw=true"),
    "https://github.com/pjreddie/darknet/blob/master/cfg/resnext50.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/resnext50.cfg"
    + quote("?raw=true"),
    "https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/yolov2.cfg"
    + quote("?raw=true"),
    "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true": f"{BASE}/2022-10-05/yolov3-tiny-raw.cfg",
    "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true": f"{BASE}/pjreddie/darknet/blob/master/cfg/yolov3.cfg"
    + quote("?raw=true"),
    "https://github.com/SebastianBoblestETAS/nn_models/blob/ce49c5de64889493161ca4194a20e0fd5eb707e6/lstm_1_in_3_out_2_ts_4.tflite?raw=true": f"{BASE}/SebastianBoblestETAS/nn_models/blob/ce49c5de64889493161ca4194a20e0fd5eb707e6/lstm_1_in_3_out_2_ts_4.tflite"
    + quote("?raw=true"),
    "https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel?raw=true": f"{BASE}/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel"
    + quote("?raw=true"),
    "https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt": f"{BASE}/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt",
    "https://github.com/tensorflow/tflite-micro/raw/a56087ffa2703b4d5632f024a8a4c899815c31bb/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite": f"{BASE}/tensorflow/tflite-micro/raw/a56087ffa2703b4d5632f024a8a4c899815c31bb/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite",
    "https://github.com/mlcommons/tiny/raw/v0.7/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite": f"{BASE}/mlcommons/tiny/raw/v0.7/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite",
    "https://github.com/uwsampl/web-data/raw/main/vta/models/synset.txt": f"{BASE}/2022-10-05/synset.txt",
    "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_cora.torch": f"{BASE}/gcn_cora.torch",
    "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg": f"{BASE}/vta_cat.jpg",
    "https://objects.githubusercontent.com/github-production-release-asset-2e65be/130932608/4b196a8a-4e2d-11e8-9a11-be3c41846711?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20221004%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221004T170456Z&X-Amz-Expires=300&X-Amz-Signature=0602b68e8864b9b01c9142eee22aed3543fe98a5482686eec33d98e2617a2295&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=130932608&response-content-disposition=attachment%3B%20filename%3Dmobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5&response-content-type=application%2Foctet-stream": f"{BASE}/2022-10-05/aws-mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip": f"{BASE}/oneflow/resnet18.zip",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/sine_model.tflite": f"{BASE}/tlc-pack/web-data/testdata/microTVM/model/sine_model.tflite",
    "https://pjreddie.com/media/files/yolov3-tiny.weights?raw=true": f"{BASE}/yolov3-tiny.weights",
    "https://pjreddie.com/media/files/yolov3.weights": f"{BASE}/yolov3.weights",
    "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_classes.txt": f"{BASE}/2022-10-05/imagenet_classes.txt",
    "https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/imagenet_synsets.txt": f"{BASE}/2022-10-05/imagenet_synsets.txt",
    "https://raw.githubusercontent.com/dmlc/mxnet.js/main/data/cat.png": f"{BASE}/dmlc/mxnet.js/main/data/cat.png",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/cfg/yolov3.cfg": f"{BASE}/dmlc/web-data/main/darknet/cfg/yolov3.cfg",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/data/arial.ttf": f"{BASE}/dmlc/web-data/main/darknet/data/arial.ttf",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/data/coco.names": f"{BASE}/dmlc/web-data/main/darknet/data/coco.names",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/data/dog.jpg": f"{BASE}/dmlc/web-data/main/darknet/data/dog.jpg",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/data/person.jpg": f"{BASE}/dmlc/web-data/main/darknet/data/person.jpg",
    "https://raw.githubusercontent.com/dmlc/web-data/main/darknet/lib/libdarknet2.0.so": f"{BASE}/dmlc/web-data/main/darknet/lib/libdarknet2.0.so",
    "https://raw.githubusercontent.com/dmlc/web-data/main/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/small_street.jpg",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/InceptionV1/classify_image_graph_def-with_shapes.pb": f"{BASE}/dmlc/web-data/main/tensorflow/models/InceptionV1/classify_image_graph_def-with_shapes.pb",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/InceptionV1/elephant-299.jpg": f"{BASE}/dmlc/web-data/main/tensorflow/models/InceptionV1/elephant-299.jpg",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/InceptionV1/imagenet_2012_challenge_label_map_proto.pbtxt": f"{BASE}/dmlc/web-data/main/tensorflow/models/InceptionV1/imagenet_2012_challenge_label_map_proto.pbtxt",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/InceptionV1/imagenet_synset_to_human_label_map.txt": f"{BASE}/dmlc/web-data/main/tensorflow/models/InceptionV1/imagenet_synset_to_human_label_map.txt",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tgz": f"{BASE}/dmlc/web-data/main/tensorflow/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tgz",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/inception_v1_quantized.tflite": f"{BASE}/dmlc/web-data/main/tensorflow/models/Quantized/inception_v1_quantized.tflite",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/mobilenet_v2_quantized.tflite": f"{BASE}/dmlc/web-data/main/tensorflow/models/Quantized/mobilenet_v2_quantized.tflite",
    "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/resnet_50_quantized.tflite": f"{BASE}/dmlc/web-data/main/tensorflow/models/Quantized/resnet_50_quantized.tflite",
    "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg": f"{BASE}/2022-10-05/street_small.jpg",
    "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt": f"{BASE}/2022-10-05/labels_mobilenet_quant_v1_224.txt",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/arm_cpu_v0.08.log": f"{BASE}/tlc-pack/tophub/main/tophub/arm_cpu_v0.08.log",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/cuda_v0.10.log": f"{BASE}/tlc-pack/tophub/main/tophub/cuda_v0.10.log",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/llvm_v0.04.log": f"{BASE}/tlc-pack/tophub/main/tophub/llvm_v0.04.log",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/mali_v0.06.log": f"{BASE}/2022-10-05/mali_v0.06.log",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/opencl_v0.04.log": f"{BASE}/tlc-pack/tophub/main/tophub/opencl_v0.04.log",
    "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub/vta_v0.10.log": f"{BASE}/tlc-pack/tophub/main/tophub/vta_v0.10.log",
    "https://s3.amazonaws.com/model-server/inputs/kitten.jpg": f"{BASE}/2022-10-05/kitten.jpg",
    "https://s3.amazonaws.com/onnx-model-zoo/synset.txt": f"{BASE}/2022-10-05/synset-s3.txt",
    "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz": f"{BASE}/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz": f"{BASE}/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz": f"{BASE}/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz": f"{BASE}/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz": f"{BASE}/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz": f"{BASE}/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz": f"{BASE}/2022-10-05/mobilenet_v2_1.0_224_quant.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip": f"{BASE}/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/digit_classifier/mnist.tflite": f"{BASE}/download.tensorflow.org/models/tflite/digit_classifier/mnist.tflite",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz": f"{BASE}/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz": f"{BASE}/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz": f"{BASE}/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz",
    "https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_140_90_12b4_720.tflite": f"{BASE}/fast-convnets/tflite-models/mbv1_140_90_12b4_720.tflite",
    "https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_200_85_11-16b2_744.tflite": f"{BASE}/fast-convnets/tflite-models/mbv2_200_85_11-16b2_744.tflite",
    "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz": f"{BASE}/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
    "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz": f"{BASE}/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz",
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5": f"{BASE}/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf.h5": f"{BASE}/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_128_tf.h5": f"{BASE}/2022-10-05/mobilenet_2_5_128_tf.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5": f"{BASE}/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5": f"{BASE}/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
    "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5": f"{BASE}/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz": f"{BASE}/tensorflow/tf-keras-datasets/mnist.npz",
    "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite": f"{BASE}/mlcommons/tiny/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite",
    "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite": f"{BASE}/mlcommons/tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite",
    "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite": f"{BASE}/mlcommons/tiny/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite",
    "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite": f"{BASE}/mlcommons/tiny/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy": f"{BASE}/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/visual_wake_word_int8_1.npy": f"{BASE}/tlc-pack/web-data/raw/main/testdata/microTVM/data/visual_wake_word_int8_1.npy",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/anomaly_detection_normal_id_01.npy": f"{BASE}/tlc-pack/web-data/raw/main/testdata/microTVM/data/anomaly_detection_normal_id_01.npy",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/image_classification_int8_0.npy": f"{BASE}/tlc-pack/web-data/raw/main/testdata/microTVM/data/image_classification_int8_0.npy",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_person.jpg": f"{BASE}/tlc-pack/web-data/testdata/microTVM/data/vww_sample_person.jpg",
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_not_person.jpg": f"{BASE}/tlc-pack/web-data/testdata/microTVM/data/vww_sample_not_person.jpg",
    "https://github.com/tensorflow/tflite-micro/raw/de8f61a074460e1fa5227d875c95aa303be01240/tensorflow/lite/micro/models/keyword_scrambled.tflite": f"{BASE}/models/tflite/keyword_scrambled_8bit.tflite",
    "https://github.com/Grovety/ModelZoo/raw/52fb82156ae8c8e3f62c7d7caf6867b25261dda4/models/object_detection/ssd_mobilenet_v1/tflite_int8/tflite_graph_with_regular_nms.pb": f"{BASE}/ssd_mobilenet_v1/tflite_int8/tflite_graph_with_regular_nms.pb",
}


class TvmRequestHook(urllib.request.Request):
    def __init__(self, url, *args, **kwargs):
        LOGGER.info(f"Caught access to {url}")
        url = url.strip()
        if url not in URL_MAP and not url.startswith(BASE):
            # Dis-allow any accesses that aren't going through S3
            msg = (
                f"Uncaught URL found in CI: {url}. "
                "A committer must upload the relevant file to S3 via "
                "https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml "
                "and add it to the mapping in tests/scripts/request_hook/request_hook.py"
            )
            raise RuntimeError(msg)

        new_url = URL_MAP[url]
        LOGGER.info(f"Mapped URL {url} to {new_url}")
        super().__init__(new_url, *args, **kwargs)


def init():
    global LOGGER
    urllib.request.Request = TvmRequestHook
    LOGGER = logging.getLogger("tvm_request_hook")
    LOGGER.setLevel(logging.DEBUG)
    fh = logging.FileHandler("redirected_urls.log")
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)
