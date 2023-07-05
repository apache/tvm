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
Deploy a Framework-prequantized Model with TVM - Part 3 (TFLite)
================================================================
**Author**: `Siju Samuel <https://github.com/siju-samuel>`_

Welcome to part 3 of the Deploy Framework-Prequantized Model with TVM tutorial.
In this part, we will start with a Quantized TFLite graph and then compile and execute it via TVM.


For more details on quantizing the model using TFLite, readers are encouraged to
go through `Converting Quantized Models
<https://www.tensorflow.org/lite/convert/quantization>`_.

The TFLite models can be downloaded from this `link
<https://www.tensorflow.org/lite/guide/hosted_models>`_.

To get started, Tensorflow and TFLite package needs to be installed as prerequisite.

.. code-block:: bash

    # install tensorflow and tflite
    pip install tensorflow==2.1.0
    pip install tflite==2.1.0

Now please check if TFLite package is installed successfully, ``python -c "import tflite"``

"""


###############################################################################
# Necessary imports
# -----------------
import os

import numpy as np
import tflite

import tvm
from tvm import relay


######################################################################
# Download pretrained Quantized TFLite model
# ------------------------------------------

# Download mobilenet V2 TFLite model provided by Google
from tvm.contrib.download import download_testdata

model_url = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
)

# Download model tar file and extract it to get mobilenet_v2_1.0_224.tflite
model_path = download_testdata(
    model_url, "mobilenet_v2_1.0_224_quant.tgz", module=["tf", "official"]
)
model_dir = os.path.dirname(model_path)


######################################################################
# Utils for downloading and extracting zip files
# ----------------------------------------------
def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


extract(model_path)


######################################################################
# Load a test image
# -----------------

#######################################################################
# Get a real image for e2e testing
# --------------------------------
def get_real_image(im_height, im_width):
    from PIL import Image

    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


data = get_real_image(224, 224)

######################################################################
# Load a tflite model
# -------------------

######################################################################
# Now we can open mobilenet_v2_1.0_224.tflite
tflite_model_file = os.path.join(model_dir, "mobilenet_v2_1.0_224_quant.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

###############################################################################
# Lets run TFLite pre-quantized model inference and get the TFLite prediction.
def run_tflite_model(tflite_model_buf, input_data):
    """Generic function to execute TFLite"""
    try:
        from tensorflow import lite as interpreter_wrapper
    except ImportError:
        from tensorflow.contrib import lite as interpreter_wrapper

    input_data = input_data if isinstance(input_data, list) else [input_data]

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]["index"], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]["index"]))

    return tflite_output


###############################################################################
# Lets run TVM compiled pre-quantized model inference and get the TVM prediction.
def run_tvm(lib):
    from tvm.contrib import graph_executor

    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    rt_mod.set_input("input", data)
    rt_mod.run()
    tvm_res = rt_mod.get_output(0).numpy()
    tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
    return tvm_pred, rt_mod


###############################################################################
# TFLite inference
# ----------------

###############################################################################
# Run TFLite inference on the quantized model.
tflite_res = run_tflite_model(tflite_model_buf, data)
tflite_pred = np.squeeze(tflite_res).argsort()[-5:][::-1]

###############################################################################
# TVM compilation and inference
# -----------------------------

###############################################################################
# We use the TFLite-Relay parser to convert the TFLite pre-quantized graph into Relay IR. Note that
# frontend parser call for a pre-quantized model is exactly same as frontend parser call for a FP32
# model. We encourage you to remove the comment from print(mod) and inspect the Relay module. You
# will see many QNN operators, like, Requantize, Quantize and QNN Conv2D.
dtype_dict = {"input": data.dtype.name}
shape_dict = {"input": data.shape}

mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
# print(mod)

###############################################################################
# Lets now the compile the Relay module. We use the "llvm" target here. Please replace it with the
# target platform that you are interested in.
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_module.build(mod, target=target, params=params)

###############################################################################
# Finally, lets call inference on the TVM compiled module.
tvm_pred, rt_mod = run_tvm(lib)

###############################################################################
# Accuracy comparison
# -------------------

###############################################################################
# Print the top-5 labels for MXNet and TVM inference.
# Checking the labels because the requantize implementation is different between
# TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via labels.

print("TVM Top-5 labels:", tvm_pred)
print("TFLite Top-5 labels:", tflite_pred)


##########################################################################
# Measure performance
# -------------------
# Here we give an example of how to measure performance of TVM compiled models.
n_repeat = 100  # should be bigger to make the measurement more accurate
dev = tvm.cpu(0)
print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))

######################################################################
# .. note::
#
#   Unless the hardware has special support for fast 8 bit instructions, quantized models are
#   not expected to be any faster than FP32 models. Without fast 8 bit instructions, TVM does
#   quantized convolution in 16 bit, even if the model itself is 8 bit.
#
#   For x86, the best performance can be achieved on CPUs with AVX512 instructions set.
#   In this case, TVM utilizes the fastest available 8 bit instructions for the given target.
#   This includes support for the VNNI 8 bit dot product instruction (CascadeLake or newer).
#   For EC2 C5.12x large instance, TVM latency for this tutorial is ~2 ms.
#
#   Intel conv2d NCHWc schedule on ARM gives better end-to-end latency compared to ARM NCHW
#   conv2d spatial pack schedule for many TFLite networks. ARM winograd performance is higher but
#   it has a high memory footprint.
#
#   Moreover, the following general tips for CPU performance equally applies:
#
#    * Set the environment variable TVM_NUM_THREADS to the number of physical cores
#    * Choose the best target for your hardware, such as "llvm -mcpu=skylake-avx512" or
#      "llvm -mcpu=cascadelake" (more CPUs with AVX512 would come in the future)
#    * Perform autotuning - :ref:`Auto-tuning a convolution network for x86 CPU
#      <tune_relay_x86>`.
#    * To get best inference performance on ARM CPU, change target argument
#      according to your device and follow :ref:`Auto-tuning a convolution
#      network for ARM CPU <tune_relay_arm>`.
