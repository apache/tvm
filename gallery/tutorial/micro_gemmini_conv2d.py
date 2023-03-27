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
Running TVM on the Gemmini accelerator - A single 2d convolutional layer example
======================================================================================
**Author**:
`Federico Peccia <https://fPecc.github.io/>`_

This tutorials shows how a quantized 2d convolution layer can be compiled to be executed on the Gemmini accelerator. The generated baremetal C code is then tested on the Spike RISC-V ISA simulator. Before starting this tutorial, you should have downloaded the Chipyard repository and installed the Spike simulator with the Gemmini extension.

"""

import tensorflow as tf
from tensorflow import keras
import tarfile
import tempfile
import pathlib
from tensorflow.keras import layers
import numpy as np
import os
import tvm.contrib.gemmini as gemmini
from tvm import relay
import tvm
from tvm.micro.testing.utils import create_header_file

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
# Baseline generation
# --------------------------------
#
# In this section, we will generate the baseline input and expected output, which we are going to use to compare with the actual obtained output after running on the Gemmini accelerator.

# Then we define the parameters of the layer we want to test. In this case:
input_height = 16
input_width = 16
input_channels = 16
output_channels = 16
kernel_size = 3
stride = 1
padding = "valid"
activation = None
bias = True

# We can add a max pooling layer after the convolution. This can be merged by the integration and can be executed together with the convolution on the Gemmini accelerator.
pool_size = 1
pool_stride = 1
pool_padding = "valid"
use_pool = False

# We will generate a prequantized TFLite model, because for now the Gemmini integration only supports models that were quantized with specific flags as input.

layer_sequence = [
    layers.Conv2D(
        output_channels,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        use_bias=True,
        bias_initializer="ones",
        input_shape=(input_height, input_width, input_channels),
        strides=stride,
    )
]
if use_pool:
    layer_sequence.append(
        layers.MaxPool2D(pool_size=pool_size, strides=pool_stride, padding=pool_padding)
    )

model = keras.Sequential(layer_sequence)

# Convert the concrete functions using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)


def representative_data_gen():
    dataset = [
        np.array(
            np.random.randint(0, 10, size=(100, input_height, input_width, input_channels)),
            dtype=np.float32,
        )
        for s in range(10)
    ]
    for input_value in dataset:
        # Model has only one input so each data point has one element.s
        yield [input_value]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_data_gen
converter._experimental_disable_per_channel = True

tflite_model = converter.convert()

# Save the model.
tmpdir = tvm.contrib.utils.tempdir()
tflite_file = tmpdir / "conv.tflite"
with open(tflite_file, "wb") as f:
    f.write(tflite_model)

# Now that we have created the model, we import the model and run it. We store the output, in order to compare it with the output that will be later obtained from the Gemmini accelerator.

os.system("rm -rf generated-project/")

tflite_model_buf = open(tflite_file, "rb").read()
input_tensor = "layer1_input"
input_dtype = "uint8"

# os.system("mkdir -p include")

try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_matrix = np.random.randint(
    0, 127, (1, input_height, input_width, input_channels), dtype=np.uint8
)
interpreter.set_tensor(input_details[0]["index"], input_matrix)
interpreter.invoke()
expected_output = interpreter.get_tensor(output_details[0]["index"])

##################################
# Compiling the model with TVM
# --------------------------------
#
# In this section, we will compile the model using TVM and the Gemmini integration.

# The Gemmini environment class needs to be initialized with the parameters of the Gemmini accelerator where we want to execute our operation. We use here the default parameters.
gemmini.Environment.init_overwrite(dim=16, acc_rows=1024, bank_rows=4096)

# The TFLite model generated in the previous steps is now imported into TVM.
mod, params = relay.frontend.from_tflite(
    tflite_model,
    shape_dict={input_tensor: (input_height, input_width, input_channels)},
    dtype_dict={input_tensor: input_dtype},
)
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

tmpdir = tvm.contrib.utils.tempdir()
model_library_format_tar_path = tvm.micro.export_model_library_format(module, tmpdir / "model.tar")
with tempfile.NamedTemporaryFile() as tar_temp_file:
    with tarfile.open(tar_temp_file.name, "w:gz") as tar_file:
        # Here, we create headers with the inputs and expected output, so that we can then execute the same operation on the Gemmini accelerator, and compare the expected output with the actual predicted one.
        create_header_file("input", input_matrix, "include/tvm", tar_file)
        create_header_file("output", expected_output, "include/tvm", tar_file)

    # Here, we create the test project, using the example project provided for this tutorial in the Gemmini microTVM template projects.
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("gemmini"))
    project_options = {"project_type": "conv2d_example", "extra_files_tar": tar_temp_file.name}

    generated_project_dir = pathlib.Path(pathlib.Path.cwd(), "generated-project")
    generated_project = tvm.micro.generate_project(
        template_project_path, module, generated_project_dir, project_options
    )

# We build the project. This will generate an executable we can run on the Spike simulator.
generated_project.build()

# Finally, we execute the compiled baremetal project on the Spike simulator.
# Note: if there are errors, these can be related to rounding errors.
generated_project.flash()
