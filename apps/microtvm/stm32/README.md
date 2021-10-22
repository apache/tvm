
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

# STM32 Application for Deep Learning Framework with TVM

- [STM32 Application for Deep Learning Framework with TVM](#stm32-application-for-deep-learning-framework-with-tvm)
    - [Motivation](#motivation)
    - [Prerequisites](#prerequisites)
    - [STM32 Application Development Flow](#stm32-application-development-flow)
        - [STM32 Application Stack](#stm32-application-stack)
        - [Compiling a Neural Model](#building)
        - [Integrating the compiled model with STM32 AI project](#integration)
        - [Flashing and running the application on the board](#running)
    - [STM32 Runtime API](#stm32-runtime-api)
    - [Appendix](#appendix)


## Motivation

Machine learning on embedded systems (often called TinyML) has the potential 
to allow the creation of small devices that can make smart decisions without 
needing to send data to the Cloud – great from efficiency and privacy 
perspective. This project shows how to build and run deep learning models (based on artificial neural networks) on the STM32 Microcontrollers (MCUs). 
The first supported development board is the Discovery H747 board.

In order to embed a pre-trained neural networks into an MCU, we leverage on the TVM compiler technology.
As demonstrated in TVM [apps](https://tvm.apache.org/docs/), TVM already 
supports C code generation. Thus, we use the TVM compiler for converting 
pre-trained deep learning models into C code that can run on STM32 MCUs, while 
optimizing code, minimizing complexity and memory requirements.
The generated C code can be integrated with the standard STM32 application
stack.

We are also working on integrating the TVM with the widely used STM32CubeMX 
tool.

## Prerequisites

The project has been tested with the Ubuntu development environment.
It requires installing following software packages:

 - ARM GCC Toolchain.
   Only GCC tollchain is supported at this time.
   [Download](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)

 - The STM32 board support package (BSP) for the H7 board and the STM32 hardwarwe abstraction layer (HAL) library for the H7 board:
   - [ZIP file](https://www.st.com/en/embedded-software/stm32cubeh7.html#get-software)
   - [GitHub](https://github.com/STMicroelectronics/STM32CubeH7)(tested with the version v1.9.0):

 - The STM32 Programmer software
   [Download](https://www.st.com/en/development-tools/stm32cubeprog.html)

The provided examples use the following python packages:

Note that the --user flag is not necessary if you’re installing to a managed 
local environment, like virtualenv.
```
pip3 install --user pyserial sklearn tqdm
```

## STM32 Application Development Flow

This project shows how to compile and integrate the compiled neural network 
model with a STM32 application. We focus on the
integration of a single model. This README also briefly describes
the possibility to integrate multiple models into a single STM32 application.

### STM32 Application Stack

        --------------  ----------  ---------        ---------
       | Application | | Model 0 | | Model 1 | .... | Model N |
        --------------  ----------  ---------        ---------

        ------------------------------------------------------
       |                    STM32 Runtime                     |
        ------------------------------------------------------

        -----------------------   ----------------------------
       | Board Support Package | | Hardware Abstraction layer |
       |         (BSP)         | |           (HAL)            |
       ------------------------   ----------------------------

        ------------------------------------------------------
       |       Development Board (eg. Discovery H747)         |
        ------------------------------------------------------


The TVM STM32 code generation is based on a lightweight [runtime API](#stm32-runtime-api) explained later in this README.

### Compiling a Neural Model

The scripts/compile.py script shows how to use the TVM compiler in order to 
compile a TFLite network model running on the STM32 microcontrollers. 
The tvm.contrib.stm32.CodeEmitter is used to generate the C implementation 
from the TVM C module model.
The script generates following files inside the '<target_dir>' directory:

 - <model_name>_graph.json : The JSON graph from TVM module
 - <model_name>_params.dat : The binary params from TVM module
 - <model_name>.[h,c] : C implementation of the JSON graph
 - <model_name>_data[h,c] : C implementation of the params data
 - <model_name>_lib.c : C implementation of the kernels from TVM module


### Integrating the compiled model with STM32 AI project

#### Setup the build environment

Setup your build environment by filling the following paths in the project 
Makefile or set them up in your shell environment:
```
 export ARM_PATH = <path to your GCC ARM toolchain>
 Ex: .../arm/gcc-arm-none-eabi/9-2019-q4-major/x86_64/bin

 export X_CUBE_PATH = <path to your STM32 BSP and HAL installation>
 Ex: .../STM32Cube_FW_H7_V1.9.0

 export X_CUBE_TOOL_PATH = <path to your ST CubeProgrammer installation>
 Ex: .../STM32CubeProgrammer/bin

 export TVM_PATH = <path to your TVM compiler>
 Ex: .../tvm
```

Setup the path to the compiled TFLite model:
```
 export MODEL_PATH = <path to your target_dir with TVM generated C implementation of the model>
```

#### Data Placement in STM32 Memory

The project includes two linker script examples. Using the linker
script it is possible to select placement of different application data
inside the STM32 board's memories. For example, following data sections 
correspond to application data managed by the stm32.CodeEmitter:
```
*.nn_weights:  model weights. Often placed into the flash memory.
*.nn_data_act: model activations.
```

The STM32H747XIHx_CM7_config1.ld script gives an example of placing all
application data in H747 internal memories, eg. the text section and the
weights are placed in the internal flash, the data, the stack and the 
heap sections are placed in the internal DTCM RAM, and the activations
are placed in the internal AXI RAM memory.

The STM32H747XIHx_CM7_config4.ld script places application weights in
external QSPI FLASH memory, while the heap section and activations are
allocated in the external SDRAM memory.

Similar linker scripts may be used for managing data placement depending
on user application requirements and the board configuration.

#### Build

The project includes two versions of the application:
 1. SystemPerformance: allows performance measurement on the board based on randomly generated input data.
 2. Validation: allows functional validation with a set of random generated or user generated inputs.

In order to build the the SystemPerformance application do the:
```
$ make perf
```

In order to build the the Validation application do:
```
$ make valid
```

### Using Multiple Neural Models

The STM32 API allows integration of multiple neural network models into a
single embedded application, or even multiple instances of the same model.

TODO: complete description.

### Flashing and running the SystemPerformance application on the board

Connect your STM32 DiscoveryH747 development board to the host computer where 
you build the application.

Flash your application:
```
$ make flash-perf
```

Flashing the application starts its execution in the board. The
SystemPerformance is running independently and its progress can be
followed in the monitor window. In order to monitor your application 
execution, setup the dedicated monitor window. 

Following example is using the Linux
'screen' utility:
```
$ screen /dev/ttyACM0 115200
```

Checking whether a screen is already running:
```
$ screen -ls
```

If there is one, the monitor window can be reattached to the existing
'screen' instance with:
```
$ screen -x <pid from screen -ls>
```

Killing a screen instance from the monitor window:
```
$ Ctrl+a-d
$ screen -ls   (gives you the PID, ex. 10265.pts-5.xd0103)
$ screen -XS 10265.pts-5.xd0103 quit
```


### Flashing and running the Validation application on the board

The Validation application allows interacting with the board by giving your
model inputs (or generating random inputs) and reading the inference
outputs.

Notice that a monitor window need to be detached from the board in order
for the Validation application to work.

Flash your Validation application:
```
$ make flash-valid
```

Run with random inputs and check agains the original model:
```
$ python3 scripts/tflite_test.py -m <model>.tflite -d serial:/dev/ttyACM0:115200 -b 10
```

Instead of using random inputs, a Numpy npz file with user generated model inputs can be provided:
```
$ python3 scripts/tflite_test.py -m <model>.tflite -d serial:/dev/ttyACM0:115200 --npz <user_data>.npz
```

The AI Runner (`tvm.contrin.stm32`  Python module) can be used to write
a validation script with the user data and specific metrics (see `scripts/mnist_test.py` example).

```python
from tvm.contrib.stm32 import AiRunner

runner = AiRunner()
runner.connect('serial:/dev/ttyACM0:115200')

tvm_outputs, _ = runner.invoke(tvm_inputs)
```


## STM32 Runtime API

For each neural network model, the compiler generates a 'ai_model_info' descriptor and a small number of API interface functions:
```
ai_status ai_create (ai_model_info * nn, ai_ptr activations, ai_handle * handle);
```
  The function takes the 'ai_model_info' descriptor and a pointer to the
  memory pool for allocating the activations, initializes the runtime
  internal structures for the model, and returns a new 'ai_handle' for this
  model.
```
ai_status ai_destroy (ai_handle handle);
```
  Releases the internal structures for the model referenced via 'handle'.
```
const char * ai_get_error (ai_handle handle);
int32_t ai_get_input_size (ai_handle handle);
int32_t ai_get_output_size (ai_handle handle);
```
  Ditto for the model referenced via 'handle'.
```
ai_tensor * ai_get_input (ai_handle handle, int32_t index);
ai_tensor * ai_get_output (ai_handle handle, int32_t index);
```
  Return the 'ai_tensor' associated with the 'index' input/output of the
  model referenced via 'handle'.
```
ai_status ai_run (ai_handle handle);
```
  Execute the model referenced via 'handle'.

The STM32 runtime API implements a number of additional methods that allow
retrieving, at runtime, a number of informations associated with a given
neural network model:
```
const char * ai_get_name (ai_handle handle);
const char * ai_get_datetime (ai_handle handle);
const char * ai_get_revision (ai_handle handle);
const char * ai_get_tool_version (ai_handle handle);
const char * ai_get_api_version (ai_handle handle);
uint32_t ai_get_node_size (ai_handle handle);
uint32_t ai_get_activations_size (ai_handle handle);
uint32_t ai_get_params_size (ai_handle handle);
ai_ptr ai_get_activations (ai_handle handle);
const ai_ptr ai_get_params (ai_handle handle);
```

## Appendix


