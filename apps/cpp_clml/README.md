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

# OpenCLML Debug Tool

Tool to generate OpenCLML source file given a model from any framework and compile it as a native application that runs on Android target.
This tool helps to debug or triage OpenCLML offloaded sub graphs as a standalone application.

### Codegen

Models can be downloaded from well known frameworks like Tensorflow, PyTorch, TFLite, Onnx ..etc.
Assuming  ```resnet50.h5``` is a Keras ResNet50 model file, use the below command to generate a OpenCLML source for the model.

```bash
python3 scripts/clml_codegen.py resnet50.h5
```

Above command generates ```clml_models.cc``` and ```clml_params.npz```.
```clml_models.cc``` contains cpp representation of all OpenCLML subgraphs offloaded by TVM compilation. This file will be used to build tool ```clml_run```.
```clml_params.npz``` is a numpy dump of all params involved in all sub graphs of TVM module. This file to be copied to target.

### Build Tool

Copy the generated models source ```clml_models.cc``` under ```cpp_clml```.

Below commands will compile the tool ```clml_run``` from generated source and other static dependents.

```bash
cmake -S . -B build_64 -D ANDROID_ABI=arm64-v8a -D CLML_SDK=<CLML SDK PATH> -D CMAKE_TOOLCHAIN_FILE=<ANDROID NDK PATH>/build/cmake/android.toolchain.cmake -D ANDROID_PLATFORM=latest
cmake --build build_64
```

### Run the tool

Copy ```clml_params.npz``` and ```clml_run``` to the target Android device

```bash
Android:/data/local/tmp $ ./clml_run --dump-meta
Input         =
Output        =
Params        =
DumpMeta      = 1
.....
Subgraph Name: tvmgen_default_clml_main_1
    Input Count  : 1
    Output Count : 1
    Input MetaInfo
        Input: tvmgen_default_clml_main_1_input_0
            Dtype : float32
            Shape : [1, 1, 1, 2048]
    Output MetaInfo
        Output: tvmgen_default_clml_main_1_layer_out_5
            Dtype : float32
            Shape : [1, 1000]

Subgraph Name: tvmgen_default_clml_main_0
    Input Count  : 1
    Output Count : 1
    Input MetaInfo
        Input: tvmgen_default_clml_main_0_input_0
            Dtype : float32
            Shape : [1, 3, 230, 230]
    Output MetaInfo
        Output: tvmgen_default_clml_main_0_layer_out_406
            Dtype : float32
            Shape : [1, 2048, 1, 1]
.....
```

The meta information above indicates that the ResNet50 model is partitioned such a way that there exists two OpenCLML subgraphs.

Below command runs the models by setting the parameters from ```clml_params.npz```.

```bash
Android:/data/local/tmp $ ./clml_run --params=./clml_params.npz
Input         =
Output        =
Params        = ./clml_params.npz
DumpMeta      = 1
......
CLMLRunner Loading Params:./clml_params.npz
CLMLRunner Loading Params:./clml_params.npz
CLMLRunner::Run :tvmgen_default_clml_main_1
CLMLRunner::Run :tvmgen_default_clml_main_0
......
```

Below command can set the model inputs from ```input.npz```  and can output sub graph outputs to ```output.npz```.
```input.npz``` should have numpy arrays for ```tvmgen_default_clml_main_1_input_0``` from sub graph ```tvmgen_default_clml_main_1``` and ```tvmgen_default_clml_main_0_input_0``` from sub graph ```tvmgen_default_clml_main_0```.

```bash
Android:/data/local/tmp $ ./clml_run --params=./clml_params.npz --input=./input.npz --output=./output.npz                                                                       <
Input         = ./input.npz
Output        = ./output.npz
Params        = ./clml_params.npz
DumpMeta      = 0
Call Build Modules
CLMLRunner Constructor: Input:./input.npz Output:./output.npz Params:./clml_params.npz
CLML Target version:3
CLMLRunner Loading Params:./clml_params.npz
CLMLRunner Loading Inputs:./input.npz
Set Input For:tvmgen_default_clml_main_1_input_0

CLMLRunner Constructor: Input:./input.npz Output:./output.npz Params:./clml_params.npz
CLML Target version:3
CLMLRunner Loading Params:./clml_params.npz
CLMLRunner Loading Inputs:./input.npz
Set Input For:tvmgen_default_clml_main_0_input_0

Loop Through the Modules
CLMLRunner::Run :tvmgen_default_clml_main_1
Saving Output:tvmgen_default_clml_main_1_layer_out_5
CLMLRunner::Run :tvmgen_default_clml_main_0
Saving Output:tvmgen_default_clml_main_0_layer_out_406
......
```

The generated output file ```output.npz``` contains all the output from all sub modules.
In this case it contains ```tvmgen_default_clml_main_1_layer_out_5``` for sub graph ```tvmgen_default_clml_main_1``` and ```tvmgen_default_clml_main_0_layer_out_406``` for sub graph ```tvmgen_default_clml_main_0``` as shown below.


```bash
Android:/data/local/tmp $ unzip -l output.npz
Archive:  output.npz
  Length      Date    Time    Name
---------  ---------- -----   ----
     4080  1980-00-00 00:00   tvmgen_default_clml_main_1_layer_out_5.npy
     8272  1980-00-00 00:00   tvmgen_default_clml_main_0_layer_out_406.npy
---------                     -------
    12352                     2 files
```
