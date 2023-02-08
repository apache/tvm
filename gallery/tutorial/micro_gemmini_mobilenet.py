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
Running TVM on the Gemmini accelerator - A complete MobileNet example
======================================================================================
**Author**:
`Federico Peccia <https://fPecc.github.io/>`_

This tutorials shows how a quantized MobileNet network can be compiled to be executed on the Gemmini accelerator. The generated baremetal C code is then tested on the Spike RISC-V ISA simulator. Before starting this tutorial, you should have downloaded the Chipyard repository and installed the Spike simulator with the Gemmini extension.
"""

import numpy as np
import tensorflow as tf
import os
import tvm.contrib.gemmini as gemmini
from tvm import relay
import tvm
from mobilenet_utils import generate_mobilenet_tflite_model, get_real_image, run_tflite_model
from tvm.contrib.download import download_testdata

##################################
# Pre-requisites
# --------------------------------
#
# After the installation of the Chipyard development tools, you should have an env.sh file in your Chipyard home directory. This file needs to be sourced before running this tutorial:
#
# .. code-block:: bash
#
#   source <your chipyard home path>/env.sh
#
# WARNING: if you have installed TVM in a virtual environment, FIRST activate the Chipyard environment, and THEN activate the tvm entironment.

##################################
# Helper functions
# --------------------------------
#
# This functions will help us generate the MobileNet model


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


def download_model():
    model_url = (
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "tflite_11_05_08/mobilenet_v2_1.0_224.tgz"
    )

    # Download model tar file and extract it to get mobilenet_v2_1.0_224.tflite
    model_path = download_testdata(
        model_url, "mobilenet_v2_1.0_224.tgz", module=["tf", "official", "mobilenet_v2"]
    )
    model_dir = os.path.dirname(model_path)

    return model_dir, model_path


def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


def create_tflite_model(model_dir: str):
    # tflite_model_name = [f for f in os.listdir(model_dir) if f.endswith(".tflite")][0]
    # return f"{model_dir}/{tflite_model_name}"
    def representative_data_gen():
        dataset = [
            np.array(np.random.randint(0, 255, size=(1, 224, 224, 3)), dtype=np.float32)
            for s in range(100)
        ]
        for input_value in dataset:
            # Model has only one input so each data point has one element.s
            yield [input_value]

    pb_file = [f for f in os.listdir(model_dir) if f.endswith(".pb")][0]
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        f"{model_dir}/{pb_file}",
        input_arrays=["input"],
        input_shapes={"input": [1, 224, 224, 3]},
        output_arrays=["MobilenetV2/Predictions/Reshape"],
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_data_gen
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()
    tflite_model_name = pb_file.replace(".pb", ".tflite")
    with open(f"{model_dir}/{tflite_model_name}", "wb") as f:
        f.write(tflite_model)

    return f"{model_dir}/{tflite_model_name}"


def generate_mobilenet_tflite_model():
    model_dir, model_path = download_model()
    extract(model_path)
    return create_tflite_model(model_dir)


##################################
# Baseline generation
# --------------------------------
#
# In this section, we will generate the baseline input and expected output, which we are going to use to compare with the actual obtained output after running on the Gemmini accelerator.

# We clean and prepare the workspace
os.system("rm -rf model.tar dev/ include/ generated-project/")
os.system("mkdir -p include")

# We will generate a prequantized TFLite model, because for now the Gemmini integration only supports models that were quantized with specific flags as input.
tflite_model_dir = generate_mobilenet_tflite_model()

input_image = get_real_image(224, 224)

tflite_model_file = os.path.join(tflite_model_dir)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Now that we have created the model, we import the model and run it. We store the output, in order to compare it with the output that will be later obtained from the Gemmini accelerator.
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

tflite_res = run_tflite_model(tflite_model_buf, input_image)
tflite_pred = np.squeeze(tflite_res).argsort()[-5:][::-1]
print("Expected argmax = %i" % (tflite_pred[0],))
print("Expected max labels = %s" % (tflite_pred,))

# Here, we create C files and headers with the inputs and expected output, so that we can then execute the same operation on the Gemmini accelerator, and compare the expected output with the actual predicted one.
gemmini.create_header_file("inputs", "data", "input", input_image, "./include")
gemmini.create_header_file("outputs", "data", "output", tflite_pred.astype(np.uint32), "./include")

##################################
# Compiling the model with TVM
# --------------------------------
#
# In this section, we will compile the model using TVM and the Gemmini integration.

# The Gemmini environment class needs to be initialized with the parameters of the Gemmini accelerator where we want to execute our operation. We use here the default parameters.
gemmini.Environment.init_overwrite(dim=16, acc_rows=1024, bank_rows=4096)

# The TFLite model generated in the previous steps is now imported into TVM.
dtype_dict = {"input": input_image.dtype.name}
shape_dict = {"input": input_image.shape}

mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
mod = relay.transform.InferType()(mod)
mod["main"]

# In order to be able to build a model for the Gemmini accelerator, we need to replace all supported layers by the Gemmini specific operators. This is done using the gemmini.preprocess pass. Notice the changes in the "main" function after running the preprocess pass.
mod = gemmini.preprocess_pass(mod)
mod["main"]

# Now, we build the Relay Graph. Notice that we are using the CRT runtime, the target is C because we want to generate C code (but the device is Gemmini), and we use the AOT executor and the USMP feature in order to get a complete bare metal C code, without calls to memory allocator APIs.
# The gemmini.build_config function returns a PassContext object containing the specific parameters needed to correctly build the model for the Gemmini accelerator.
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": False})
TARGET = tvm.target.target.Target({"kind": "c", "device": "gemmini"})
EXECUTOR = tvm.relay.backend.Executor("aot", options={"interface-api": "c", "unpacked-api": 1})

with gemmini.build_config(usmp_alg="hill_climb", opt_level=3, disabled_pass=["AlterOpLayout"]):
    module = relay.build(mod, executor=EXECUTOR, runtime=RUNTIME, target=TARGET, params=params)

#################################################
# Exporting and testing the model using microTVM
# -----------------------------------------------
#
# In this section, we will export the model using one of the provided example microTVM projects, we will compile it using the Chipyard tool, and then test the generated baremetal code on the Spike simulator.

# The builded model is exported to the model library format. This will be used in the next steps to generate the baremetal project.
import pathlib

os.system("mkdir dev")
model_library_format_tar_path = pathlib.Path(pathlib.Path.cwd(), "dev/model.tar")
tvm.micro.export_model_library_format(module, model_library_format_tar_path)

import tarfile

with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# Here, we create the test project, using the example project provided for this tutorial in the Gemmini microTVM template projects.
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("gemmini"))
project_options = {"project_type": "mobilenet_example"}

generated_project_dir = pathlib.Path(pathlib.Path.cwd(), "generated-project")
generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# We build the project. This will generate an executable we can run on the Spike simulator.
generated_project.build()

# Finally, we execute the compiled baremetal project on the Spike simulator.
generated_project.flash()
