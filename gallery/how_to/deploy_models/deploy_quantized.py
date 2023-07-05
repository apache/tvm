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
"""
Deploy a Quantized Model on Cuda
================================
**Author**: `Wuwei Lin <https://github.com/vinx13>`_

This article is an introductory tutorial of automatic quantization with TVM.
Automatic quantization is one of the quantization modes in TVM. More details on
the quantization story in TVM can be found
`here <https://discuss.tvm.apache.org/t/quantization-story/3920>`_.
In this tutorial, we will import a GluonCV pre-trained model on ImageNet to
Relay, quantize the Relay model and then perform the inference.
"""


import tvm
from tvm import te
from tvm import relay
import mxnet as mx
from tvm.contrib.download import download_testdata
from mxnet import gluon
import logging
import os

batch_size = 1
model_name = "resnet18_v1"
target = "cuda"
dev = tvm.device(target)

###############################################################################
# Prepare the Dataset
# -------------------
# We will demonstrate how to prepare the calibration dataset for quantization.
# We first download the validation set of ImageNet and pre-process the dataset.
calibration_rec = download_testdata(
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec",
    "val_256_q90.rec",
)


def get_val_data(num_workers=4):
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch):
        return batch.data[0].asnumpy(), batch.label[0].asnumpy()

    img_size = 299 if model_name == "inceptionv3" else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec=calibration_rec,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        data_shape=(3, img_size, img_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return val_data, batch_fn


###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.

calibration_samples = 10


def calibrate_dataset():
    val_data, batch_fn = get_val_data()
    val_data.reset()
    for i, batch in enumerate(val_data):
        if i * batch_size >= calibration_samples:
            break
        data, _ = batch_fn(batch)
        yield {"data": data}


###############################################################################
# Import the model
# ----------------
# We use the Relay MxNet frontend to import a model from the Gluon model zoo.
def get_model():
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == "inceptionv3" else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params


###############################################################################
# Quantize the Model
# ------------------
# In quantization, we need to find the scale for each weight and intermediate
# feature map tensor of each layer.
#
# For weights, the scales are directly calculated based on the value of the
# weights. Two modes are supported: `power2` and `max`. Both modes find the
# maximum value within the weight tensor first. In `power2` mode, the maximum
# is rounded down to power of two. If the scales of both weights and
# intermediate feature maps are power of two, we can leverage bit shifting for
# multiplications. This make it computationally more efficient. In `max` mode,
# the maximum is used as the scale. Without rounding, `max` mode might have
# better accuracy in some cases. When the scales are not powers of two, fixed
# point multiplications will be used.
#
# For intermediate feature maps, we can find the scales with data-aware
# quantization. Data-aware quantization takes a calibration dataset as the
# input argument. Scales are calculated by minimizing the KL divergence between
# distribution of activation before and after quantization.
# Alternatively, we can also use pre-defined global scales. This saves the time
# for calibration. But the accuracy might be impacted.


def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


###############################################################################
# Run Inference
# -------------
# We create a Relay VM to build and execute the model.
def run_inference(mod):
    model = relay.create_executor("vm", mod, dev, target).evaluate()
    val_data, batch_fn = get_val_data()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch)
        prediction = model(data)
        if i > 10:  # only run inference on a few samples in this tutorial
            break


def main():
    mod, params = get_model()
    mod = quantize(mod, params, data_aware=True)
    run_inference(mod)


if __name__ == "__main__":
    main()
