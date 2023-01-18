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

# Zephyr RTOS Demo with CMSIS-NN

This project was used for the [TVMCon 2021 talk on Cortex-M improvements to TVM](https://www.youtube.com/watch?v=6a7o8U-8Op4). It runs a keyword spotting model with the Zephyr RTOS using CMSIS-NN with the Ahead-of-Time (AOT) executor and the stack allocation strategy.

The application starts from [the Zephyr base project](https://docs.zephyrproject.org/latest/application/index.html#application) and makes minimal changes to integrate TVM. To try it out, first refer to the [Zephyr Getting Started](https://docs.zephyrproject.org/latest/getting_started/index.html) page to setup your tooling such as `west` (you can also use the `tlcpack/ci_cortexm` image). Then download the [Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps) and set the path for Zephyr to find it:

```
export ARMFVP_BIN_PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/
```

Download the keyword spotting model to the `model` directory:
```
wget \
    https://github.com/ARM-software/ML-zoo/blob/ee35139af86bdace5e502b09fe8b9da9cb1f06bb/models/keyword_spotting/cnn_small/tflite_int8/cnn_s_quantized.tflite \
    -O model/cnn_s_quantized.tflite
```

Checkout [CMSIS_5](https://github.com/ARM-software/CMSIS_5.git) (default is `/opt/arm/ethosu/cmsis` to reflect `tlcpack/ci_cortexm`):
```
git clone "https://github.com/ARM-software/CMSIS_5.git" cmsis
```

Checkout [CMSIS NN](https://github.com/ARM-software/CMSIS-NN.git) (default is `/opt/arm/ethosu/cmsis/CMSIS-NN` to reflect `tlcpack/ci_cortexm`):
```
git clone "https://github.com/ARM-software/CMSIS-NN.git" cmsis/CMSIS-NN
```

And run the demo using `west`, with the path to CMSIS:
```
west build -t run -- -DCMSIS_PATH=/opt/arm/ethosu/cmsis
```
