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


Running TVM on bare metal Arm(R) Cortex(R)-M55 CPU, Ethos(TM)-U55 NPU
=====================================================================

This folder contains an example of how to use TVM to run a model
on bare metal Cortex(R)-M55 CPU, Ethos(TM)-U55 NPU.

Prerequisites
-------------
If the demo is run in the ci_cortexm Docker container provided with TVM, then the following software will already be installed.

If the demo is not run in the ci_cortexm Docker container, then you will need the following:
- Software required to build the Ethos(TM)-U driver stack and run the demo (These can all be
  installed by running tvm/docker/install/ubuntu_install_ethosu_driver_stack.sh.)
  - [Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)
  - [cmake 3.19.5](https://github.com/Kitware/CMake/releases/)
  - [GCC toolchain from Arm(R)](https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2)
  - [Arm(R) Ethos(TM)-U NPU driver stack](https://review.mlplatform.org)
  - [CMSIS](https://github.com/ARM-software/CMSIS_5)
- The python libraries listed in the requirements.txt of this directory
  - These can be installed by running the following from the current directory:
    ```bash
    pip install -r ./requirements.txt
    ```

You will also need TVM which can either be:
  - Built from source (see [Install from Source](https://tvm.apache.org/docs/install/from_source.html))
    - When building from source, the following need to be set in config.cmake:
      - set(USE_ETHOSU ON)
      - set(USE_CMSISNN ON)
      - set(USE_MICRO ON)
      - set(USE_LLVM ON)
  - Installed from TLCPack(see [TLCPack](https://tlcpack.ai/))

You will need to update your PATH environment variable to include the path to cmake 3.19.5 and the FVP.
For example if you've installed these in ```/opt/arm``` , then you would do the following:
```bash
export PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4:/opt/arm/cmake/bin:$PATH
```
If you are going to compile and run the demo on [Alif DevKit](https://alifsemi.com/support/kits/ensemble-devkit/) you will also need [Alif's SETools](https://alifsemi.com/support/software-tools/ensemble/) installed and configured.

Understanding the demo application
----------------------------------
This demo will:
- Download a quantized (int8) mobilenet v2 model
- Use tvmc to compile the model for Cortex(R)-M55 CPU, Ethos(TM)-U55 NPU and CMSIS-NN
- Download an image of a penguin to run the model on
- Create a C header file inputs.c containing the image data as a C array
- Create a C header file outputs.c containing a C array where the output of inference will be stored
- Build the Ethos(TM)-U55 core driver
- Build the demo application
- Run the demo application on a Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software or on the Alif DevKit
- The application will display what the image has been classified as e.g. "The image has been classified as 'king penguin'"

Running the demo application on FVP
-----------------------------------
Type the following command to run the bare metal demo application on FVP ([src/demo_bare_metal.c](./src/demo_bare_metal.c)):

```bash
./run_demo.sh
```

To run the demo on FVP using FreeRTOS task scheduling and queues ([src/demo_freertos.c](./src/demo_freertos.c)), specify the path to FreeRTOS using `--freertos_path`, for example:
```bash
./run_demo.sh --freertos_path /opt/freertos/FreeRTOSv202112.00/
```

Running the demo application on Alif DevKit
-------------------------------------------

To run the demo ([src/demo_bare_metal_alif.c](./src/demo_bare_metal_alif.c)) on Alif DevKit, specify board revision using `--alif_target_board`, for example:
```bash
./run_demo.sh --alif_target_board BOARD_AppKit_Alpha1
```
Than use `flash.sh` to upload the firmware to the board. Make sure that you have SETools installed and toolkit is configured according to AUGD0005 and console port connected to UART4.
```bash
./flash --alif_toolkit_path path_to_SETools --alif_console_port tty_device
```

If the Ethos(TM)-U driver and/or CMSIS have not been installed in /opt/arm/ethosu then the locations for these can be specified as arguments to run_demo.sh, for example:

```bash
./run_demo.sh --ethosu_driver_path /home/tvm-user/ethosu/core_driver --cmsis_path /home/tvm-user/cmsis \
--ethosu_platform_path /home/tvm-user/ethosu/core_platform
```


Using your own input
--------------------
The create_image.py script takes a single argument on the command line which is the path of the image to be converted into an array of bytes for consumption by the model.

The demo can be modified to use an image of your choice by changing the following lines in run_demo.sh

```bash
curl -sS https://upload.wikimedia.org/wikipedia/commons/1/18/Falkland_Islands_Penguins_29.jpg -o penguin.jpg
python3 ./convert_image.py ./build/penguin.jpg
```
