<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


Running TVM on bare metal Arm(R) Cortex(R)-M55 CPU and CMSIS-NN
===============================================================

This folder contains an example of how to use TVM to run a model
on bare metal Cortex(R)-M55 CPU and CMSIS-NN.

Prerequisites
-------------
If the demo is run in the ci_cpu Docker container provided with TVM, then the following
software will already be installed.

If the demo is not run in the ci_cpu Docker container, then you will need the following:
- Software required to build and run the demo (These can all be installed by running
  tvm/docker/install/ubuntu_install_ethosu_driver_stack.sh.)
  - [Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)
  - [cmake 3.19.5](https://github.com/Kitware/CMake/releases/)
  - [GCC toolchain from Arm(R)](https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2)
  - [Arm(R) Ethos(TM)-U NPU driver stack](https://review.mlplatform.org)
  - [CMSIS](https://github.com/ARM-software/CMSIS_5)
  - [CMSIS NN](https://github.com/ARM-software/CMSIS-NN)
- The python libraries listed in the requirements.txt of this directory
  - These can be installed by running the following from the current directory:
    ```bash
    pip install -r ./requirements.txt
    ```

You will also need TVM which can either be:
  - Built from source (see [Install from Source](https://tvm.apache.org/docs/install/from_source.html))
    - When building from source, the following need to be set in config.cmake:
      - set(USE_CMSISNN ON)
      - set(USE_MICRO ON)
      - set(USE_LLVM ON)
  - Installed from TLCPack(see [TLCPack](https://tlcpack.ai/))

You will need to update your PATH environment variable to include the path to cmake 3.19.5 and the FVP.
For example if you've installed these in ```/opt/arm``` , then you would do the following:
```bash
export PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4:/opt/arm/cmake/bin:$PATH
```

Running the demo application
----------------------------
Type the following command to run the bare metal demo application ([src/demo_bare_metal.c](./src/demo_bare_metal.c)):

```bash
./run_demo.sh
```

If the Ethos(TM)-U platform and/or CMSIS have not been installed in /opt/arm/ethosu then
the locations for these can be specified as arguments to run_demo.sh, for example:

```bash
./run_demo.sh --cmsis_path /home/tvm-user/cmsis \
--ethosu_platform_path /home/tvm-user/ethosu/core_platform
```

This will:
- Download a quantized (int8) person detection model
- Use tvmc to compile the model for Cortex(R)-M55 CPU and CMSIS-NN
- Download an image to run the model on
- Create a C header file inputs.c containing the image data as a C array
- Create a C header file outputs.c containing a C array where the output of inference will be stored
- Build the demo application
- Run the demo application on a Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software
- The application will report whether a person was detected e.g. "Person detected." or "No person detected."

Using your own image
--------------------
The create_image.py script takes a single argument on the command line which is the path of the
image to be converted into an array of bytes for consumption by the model.

The demo can be modified to use an image of your choice by changing the following line in run_demo.sh

```bash
curl -sS https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp -o input_image.bmp
```
