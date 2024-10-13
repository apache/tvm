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


Running TVM on ESP32-C3 (RISC-V) board for KWS task
===============================================================

This folder includes an example demonstrating the utilization of TVM to execute a model on an ESP32-C3 board using ESP-IDF FreeRTOS. The provided demo specifically runs on a designated physical board equipped with a microphone (refer to Prerequisites for details). In addition to executing the detection model, the demo encompasses components related to the microphone, preprocessing logic (MFCC), and a basic voice detection algorithm to enhance the accuracy of the detection process.

KWS model used is [ARM(R) DS-CNN model](https://github.com/ARM-software/ML-examples/tree/9da709d96e5448520521e17165637c056c9bfae7/tflu-kws-cortex-m) trained on [Speech Commands](https://arxiv.org/abs/1804.03209) dataset on the following words' subset: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go".

Prerequisites
-------------
This demo is intended to run on [Seeed Studio XIAO ESP32-C3](https://wiki.seeedstudio.com/XIAO_ESP32C3_Getting_Started) board with connected INMP441 I2S Microphone. GPIO pins configuration is descrided in [main/include/def.h](./main/include/def.h).

To run demo you will need the Espressif IoT Development Framework. It can be [installed](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/linux-macos-setup.html) in current environment or can be used through [docker image](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/tools/idf-docker-image.html).

You will also need TVM which can either be:
- Built from source (see [Install from Source](https://tvm.apache.org/docs/install/from_source.html))
  - When building from source, the following need to be set in config.cmake:
    - set(USE_MICRO ON)
    - set(USE_LLVM ON)
- Installed from TLCPack(see [TLCPack](https://tlcpack.ai/))

And the python libraries listed in the requirements.txt of this directory.
- These can be installed by running the following from the current directory:
  ```bash
  pip install -r ./requirements.txt
  ```

Running the demo application
----------------------------
Type the following command to run the demo application [main/main.cc](./main/main.cc):

```bash
./run.sh
```

This will:
- Download a model for KWS (key word spotting) from ARM(R) ML-examples repository
- Download [NMSIS](https://github.com/Nuclei-Software/NMSIS) library for preprocessing routines
- Use tvmc to compile the model for RISC-V
- Create a C header files inputs.h and outputs.h containing placeholders for input and output data
- Build the demo application as with ESP-IDF
- Flash and run the demo application on a connected board, showing a [monitor](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/tools/idf-monitor.html)
- After initialization the application will wait with `ready. waiting...` prompt for microphone input and will report `word is '<word>'` on detection
