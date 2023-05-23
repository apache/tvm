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
.. _tutorial-micro-ide:

9. Bring microTVM to your own development environment
======================================================
**Author**:
`Mohamad Katanbaf <https://github.com/mkatanbaf>`_

This tutorial describes the steps required to integrate a model compiled with microTVM into a custom development environment.
We use `STM32CubeIDE <https://www.st.com/en/development-tools/stm32cubeide.html>`_, as the target IDE in this tutorial, but we do not rely on any specific feature of this IDE and integrating microTVM in other IDEs would be similar.
We also use the Visual Wake Word (VWW) model from MLPerf Tiny and the nucleo_l4r5zi board here, but the same steps can be used for any other model or target MCU.
If you want to use another target MCU with the vww model, we recommend a cortex-M4 or cortex-M7 device with ~512 KB and ~256 KB of Flash and RAM respectively.

Here is a brief overview of the steps that we would take in this tutorial.

1. We start by importing the model, compiling it using TVM and generating the `Model Library Format <https://tvm.apache.org/docs/arch/model_library_format.html>`_ (MLF) tar-file that includes the generated code for the model as well as all the required TVM dependencies.
2. We also add two sample images in binary format (one person and one not-person sample) to the .tar file for evaluating the model.
3. Next we use the stmCubeMX to generate the initialization code for the project in stmCube IDE.
4. After that, we include our MLF file and the required CMSIS libraries in the project and build it.
5. Finally, we flash the device and evaluate the model performance on our sample images.

Let's Begin.
"""

######################################################################
# Install microTVM Python dependencies
# ------------------------------------
#
# TVM does not include a package for Python serial communication, so
# we must install one before using microTVM. We will also need TFLite
# to load models, and Pillow to prepare the sample images.
#
#   .. code-block:: bash
#
#     %%shell
#     pip install pyserial==3.5 tflite==2.1 Pillow==9.0 typing_extensions
#


######################################################################
# Import Python dependencies
# ---------------------------
#
# If you want to run this script locally, check out `TVM Online Documentation <https://tvm.apache.org/docs/install/index.html>`_ for instructions to install TVM.
#

import os
import numpy as np
import pathlib
import json
from PIL import Image
import tarfile

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
from tvm.relay.op.contrib import cmsisnn
from tvm.micro.testing.utils import create_header_file

######################################################################
# Import the TFLite model
# ------------------------
#
# To begin with, download and import a Visual Wake Word TFLite model. This model takes in a 96x96x3 RGB image and determines whether a person is present in the image or not.
# This model is originally from `MLPerf Tiny repository <https://github.com/mlcommons/tiny>`_.
# To test this model, we use two samples from `COCO 2014 Train images <https://cocodataset.org/>`_.
#
MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
MODEL_NAME = "vww_96_int8.tflite"
MODEL_PATH = download_testdata(MODEL_URL, MODEL_NAME, module="model")

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 96, 96, 3)
INPUT_NAME = "input_1_int8"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)

######################################################################
# Generate the Model Library Format file
# -----------------------------------------
#
# First we define the target, runtime and executor. Then we compile the model for the target device and
# finally we export the generated code and all the required dependencies in a single file.
#

# We can use TVM native schedules or rely on the CMSIS-NN kernels using TVM Bring-Your-Own-Code (BYOC) capability.
USE_CMSIS_NN = True

# USMP (Unified Static Memory Planning) performs memory planning of all tensors holistically to achieve best memory utilization
DISABLE_USMP = False

# Use the C runtime (crt)
RUNTIME = Runtime("crt")

# We define the target by passing the board name to `tvm.target.target.micro`.
# If your board is not included in the supported models, you can define the target such as:
# TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m4")
TARGET = tvm.target.target.micro("stm32l4r5zi")

# Use the AOT executor rather than graph or vm executors. Use unpacked API and C calling style.
EXECUTOR = tvm.relay.backend.Executor(
    "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
)

# Now, we set the compilation configurations and compile the model for the target:
config = {"tir.disable_vectorize": True}
if USE_CMSIS_NN:
    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
if DISABLE_USMP:
    config["tir.usmp.enable"] = False

with tvm.transform.PassContext(opt_level=3, config=config):
    if USE_CMSIS_NN:
        # When we are using CMSIS-NN, TVM searches for patterns in the
        # relay graph that it can offload to the CMSIS-NN kernels.
        relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    lowered = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
print(f"Model parameter size: {parameter_size}")

# We need to pick a directory where our file will be saved.
# If running on Google Colab, we'll save everything in ``/root/tutorial`` (aka ``~/tutorial``)
# but you'll probably want to store it elsewhere if running locally.

BUILD_DIR = pathlib.Path("/root/tutorial")
# sphinx_gallery_start_ignore
BUILD_DIR = pathlib.Path(os.getcwd()) / "tutorial"
# sphinx_gallery_end_ignore

BUILD_DIR.mkdir(exist_ok=True)

# Now, we export the model into a tar file:
TAR_PATH = pathlib.Path(BUILD_DIR) / "model.tar"
export_model_library_format(lowered, TAR_PATH)

######################################################################
# Add sample images to the MLF files
# -----------------------------------
# Finally, we downlaod two sample images (one person and one not-person), convert them to binary format and store them in two header files.
#

with tarfile.open(TAR_PATH, mode="a") as tar_file:
    SAMPLES_DIR = "samples"
    SAMPLE_PERSON_URL = (
        "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_person.jpg"
    )
    SAMPLE_NOT_PERSON_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_not_person.jpg"

    SAMPLE_PERSON_PATH = download_testdata(SAMPLE_PERSON_URL, "person.jpg", module=SAMPLES_DIR)
    img = Image.open(SAMPLE_PERSON_PATH)
    create_header_file("sample_person", np.asarray(img), SAMPLES_DIR, tar_file)

    SAMPLE_NOT_PERSON_PATH = download_testdata(
        SAMPLE_NOT_PERSON_URL, "not_person.jpg", module=SAMPLES_DIR
    )
    img = Image.open(SAMPLE_NOT_PERSON_PATH)
    create_header_file("sample_not_person", np.asarray(img), SAMPLES_DIR, tar_file)

######################################################################
# At this point you have all you need to take the compiled model to your IDE and evaluate it. Inside the MLF file (model.tar), you should find the following file hierearchy:
#
#     .. code-block::
#
#      /root
#      ├── codegen
#      ├── parameters
#      ├── runtime
#      ├── samples
#      ├── src
#      ├── templates
#      ├── metadata.json
#
# * The codegen folder includes the C code TVM generated for your model.
# * The runtime folder includes all the TVM dependencies that the target needs to compile the generated C code.
# * The samples folder includes the two generated sample files for evaluating the model.
# * The src folder includes the relay module describing the model.
# * The templates folder includes two template files that you might need to edit based on your platform.
# * The metadata.json file includes information about the model, its layers and memory requirement.
#


######################################################################
# Generate the project in your IDE
# -----------------------------------
#
# The next step is to create a project for our target device. We use STM32CubeIDE, you can download it `here <https://www.st.com/en/development-tools/stm32cubeide.html>`_.
# We are using version 1.11.0 in this tutorial. Once you install STM32CubeIDE follow these steps to create a project:
#
# #. select File -> New -> STM32Project. The target selection Window appears.
#
# #. Navigate to the "Board Selector" tab, type in the board name "nucleo-l4r5zi" in the "Commercial Part Number" text box. Select the board from the list of boards that appear on the right side of the screen and click "Next".
#
# #. Type in your project name (for example microtvm_vww_demo). We are using the default options. (Target Language: C, Binary Type: Executable, Project Type: STM32Cube). Click "Finish".
#
# #. A text box will appear asking if you want to "Initialize all the peripherals with their default mode?". click "Yes". This will generate the project and open the device configuration tool where you can use the GUI to setup the peripherals. By default the USB, USART3 and LPUART1 are enabled, as well as a few GPIOs.
#
# #. We will use LPUART1 to send data to the host pc. From the connectivity section, select the LPUART1 and set the "Baud Rate" to 115200 and the "Word Length" to 8. Save the changes and click "Yes" to regenerate the initialization code. This should regenerate the code and open your main.c file. You can also find main.c from the Project Explorer panel on the left, under microtvm_vww_demo -> Core -> Src.
#
# #. For sanity check, copy the code below and paste it in the "Infinite loop (aka. While (1) ) section of the main function.
#
#    * Note: Make sure to write your code inside the sections marked by USER CODE BEGIN <...> and USER CODE END <...>. The code outside these sections get erased if you regenerate the initialization code.
#
#        .. code-block:: c
#
#         HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
#         HAL_UART_Transmit(&hlpuart1, "Hello World.\r\n", 14, 100);
#         HAL_Delay(1000);
#
# #. From the menu bar, select Project -> Build (or right click on project name and select Build). This should build the project and generate the .elf file. Select Run -> Run to download the binary on your MCU. If the "Edit Configuration" window opens, just click "OK".
#
# #. Open the terminal console on your host machine. On Mac you can simply use the "screen <usb_device> 115200" command, e.g. "screen tty.usbmodemXXXX 115200". An LED should blink on the board and the string "Hello World." should print out on your terminal console every second. Press "Control-a k" to exit screen.
#

######################################################################
# Import the model to the generated project
# ------------------------------------------
#
# To integrate the compiled model into the generated project, follow these steps:
#
# #. Extract the tar file and include it in the project
#
#    * Open the project Properties. (by right clicking on the project name and selecting "Properties" or by selecting Project -> Properties from the menu bar).
#    * Select C/C++ General -> Paths and Symbols. Select the Source Location tab.
#    * If you extracted the model inside the project folder, click "Add Folder" and select the "model" folder. (You might need to right click on the project name and select "Refresh" before it appears.)
#    * If you extracted the model file somewhere else, click on the "Link Folder" button, check the box for "Link to folder in the file system" in the window that appears, click "Browse" and select the model folder.
#
# #. If you used CMSIS-NN in compiling the model, you need to include the CMSIS-NN source files in your project too.
#
#    * Download or clone the files from the `CMSIS-NN repository <https://github.com/ARM-software/CMSIS-NN>`_, and follow the above steps to include the CMSIS-NN folder in the project.
#
# #. Open the project properties. In C/C++ Build -> Settings: add the following folders to the list of Include Paths for MCU GCC Compiler (and MCU G++ Compiler if you have a C++ project) by clicking on the "+" button, selecting "Workspace" and navigating to each of the following folders:
#
#    * model/runtime/include
#    * model/codegen/host/include
#    * model/samples
#    * CMSIS-NN/Include
#
# #. Copy crt_config.h.template from model/templates to the Core/Inc folder, and rename it to crt_config.h.
#
# #. Copy platform.c.template from model/templates to the Core/Src folder, and rename it to platform.c.
#    * This file includes functions for managing the memory that you might need to edit based on your platform.
#    * define "TVM_WORKSPACE_SIZE_BYTES" in platform.c. if you are using USMP, a small value (for example 1024 Bytes) is enough.
#    * if you are not using usmp, checkout "workspace_size_bytes" field in metadata.json for an estimate of the required memory.
#
# #. Exclude the following folders from build (right click on the folder name, select Resource Configuration → Exclude from build). Check Debug and Release configurations.
#
#    * CMSIS_NN/Tests
#
# #. Download the CMSIS drivers from `CMSIS Version 5 repository <https://github.com/ARM-software/CMSIS_5>`_.
#
#    * In your Project directory, delete the Drivers/CMSIS/Include folder (which is an older version of the CMSIS drivers) and copy the CMSIS/Core/Include from the one you downloaded in its place.
#
# #. Edit the main.c file:
#
#    * Include following header files:
#
#        .. code-block:: c
#
#         #include <stdio.h>
#         #include <string.h>
#         #include <stdarg.h>
#         #include "tvmgen_default.h"
#         #include "sample_person.h"
#         #include "sample_not_person.h"
#
#    * Copy the following code into the main function right before the infinite loop. It sets the input and output to the model.
#
#        .. code-block:: c
#
#         TVMPlatformInitialize();
#         signed char output[2];
#         struct tvmgen_default_inputs inputs = {
#         .input_1_int8 = (void*)&sample_person,
#         };
#         struct tvmgen_default_outputs outputs = {
#         .Identity_int8 = (void*)&output,
#         };
#         char msg[] = "Evaluating VWW model using microTVM:\r\n";
#         HAL_UART_Transmit(&hlpuart1, msg, strlen(msg), 100);
#         uint8_t sample = 0;
#         uint32_t timer_val;
#         char buf[50];
#         uint16_t buf_len;
#
#    * Copy the following code inside the infinite loop to run inference on both images and print the result on the console:
#
#        .. code-block:: c
#
#         if (sample == 0)
#             inputs.input_1_int8 = (void*)&sample_person;
#         else
#             inputs.input_1_int8 = (void*)&sample_not_person;
#
#         timer_val = HAL_GetTick();
#         tvmgen_default_run(&inputs, &outputs);
#         timer_val = HAL_GetTick() - timer_val;
#         if (output[0] > output[1])
#             buf_len = sprintf(buf, "Person not detected, inference time = %lu ms\r\n", timer_val);
#         else
#             buf_len = sprintf(buf, "Person detected, inference time = %lu ms\r\n", timer_val);
#         HAL_UART_Transmit(&hlpuart1, buf, buf_len, 100);
#
#         sample++;
#         if (sample == 2)
#             sample = 0;
#
#
#    * Define the TVMLogf function in main, to receive TVM runtime errors on serial console.
#
#        .. code-block:: c
#
#         void TVMLogf(const char* msg, ...) {
#           char buffer[128];
#           int size;
#           va_list args;
#           va_start(args, msg);
#           size = TVMPlatformFormatMessage(buffer, 128, msg, args);
#           va_end(args);
#           HAL_UART_Transmit(&hlpuart1, buffer, size, 100);
#         }
#
# #. In project properties, C/C++ Build -> Settings, MCU GCC Compiler -> Optimization, set the Optimization level to "Optimize more (-O2)"


######################################################################
# Evaluate the model
# -------------------
#
# Now, select Run -> Run from the menu bar to flash the MCU and run the project.
# You should see the LED blinking and the inference result printing on the console.
#
