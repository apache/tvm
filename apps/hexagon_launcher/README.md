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
# Hexagon Graph Launcher

## Compilation

The launcher consists of two parts: part running on Hexagon, and part running
on Android. Each component must be compiled separately. 

The supported Snapdragon architectures are 855, 865, and 888.

### Prerequisites

1. Android NDK version r19c or later.
2. Hexagon SDK version 4.0.0 or later.

Android NDK can be downloaded from https://developer.android.com/ndk.
Hexagon SDK is available at //developer.qualcomm.com/software/hexagon-dsp-sdk.

### Compilation with TVM

Building the Hexagon launcher application as a component of the main TVM build
used for Hexagon codegen can be achieved by setting `USE_HEXAGON_LAUNCHER=ON`.
This option will compile core tvm, the android launcher binary and its corresponding
tvm_runtime, as well as the Hexagon launcher shared library and its corresponding
tvm_runtime. As described in the [Manual compilation](#Manual compilation) section 
each component requires Hexagon and android dependencies. When building the launcher 
along with TVM these configurations must be providing when invoking cmake. A minimal 
example invocation for compiling TVM along with the Hexagon launcher is included below:

```
cmake -DCMAKE_C_COMPILER=/path/to/clang \
      -DCMAKE_CXX_COMPILER=/path/to/clang++ \
      -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
      -DCMAKE_CXX_STANDARD=14 \
      -DUSE_LLVM=/path/to/llvm/bin/llvm-config \
      -DUSE_HEXAGON_ARCH=v65|v66|v68 \
      -DUSE_HEXAGON_LAUNCHER=ON \
      -DUSE_HEXAGON_SDK=/path/to/hexagon/SDK \
      -DUSE_HEXAGON_TOOLCHAIN=/path/to/hexagon/toolchain/ ..
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DUSE_ANDROID_TOOLCHAIN=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
      ..
```

where `v65|v66|v68` means "one of" these architecture versions.
The Hexagon launcher application is an android binary and thus requires the use 
of an android toolchain for compilation. Similarly, the Hexagon tvm runtime 
requires the use of the Hexagon toolchain and depends on the Hexagon SDK. The 
resulting hexagon launcher binaries can be found in the `launcher` subdirectory 
of the cmake build directory. Please note that the above command will not build
the support for Hexagon codegen into the TVM, for that please additionally define
the `USE_HEXAGON_DEVICE` variable.

### Manual compilation

Since some source files are shared between the Hexagon and android builds, 
make sure to delete all object files between compilations. Compile the Hexagon
code first.

#### Compilation of the Hexagon part

Create a subdirectory for the build files, and run `cmake` with the
following variables set:

```
cmake -DCMAKE_C_COMPILER=/path/to/hexagon-clang \
      -DCMAKE_CXX_COMPILER=/path/to/hexagon-clang++ \
      -DUSE_HEXAGON_ARCH=v65|v66|v68 \
      -DUSE_HEXAGON_SDK=/path/to/hexagon/SDK \
      /path/to/apps/hexagon_launcher/cmake/hexagon
```

Run `make`. This will create `liblauncher_rpc_skel.so`. The static version of
the TVM runtime for Hexagon will be built as a part of the process.

#### Compilation of the Android part

2. Create a subdirectory for the build files (different from the one used for
   Hexagon files), and run `cmake` with the following variables set:

```
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DUSE_HEXAGON_SDK=/p/Hexagon_SDK/4.3.0.0
      -DUSE_HEXAGON_ARCH=v65|v66|v68
      /path/to/apps/hexagon_launcher/cmake/android
```

Run `make`. This will create `launcher_android`. The TVM runtime for Android will
be built as a part of the process. Depending on the version of cmake that you are
using, you may see the following warnings---they can be ignored.

```
An old version of CMake is being used that cannot automatically detect
compiler attributes.  Compiler identification is being bypassed.  Some
values may be wrong or missing.  Update to CMake 3.19 or newer to use
CMake's built-in compiler identification.
```

## Execution

From the Android shell, do
```
./launcher_android --in_config input.json --out_config output.json
```

You may need to add the location of `libtvm_runtime.so` to `LD_LIBRARY_PATH`.
See below for more information about the setup and launcher's inputs.

### Preparation steps

Copy the following binaries to the device:
- `liblauncher_rpc_skel.so`: created by the compilation step for Hexagon,
- `libgcc.so`: take this one from the Hexagon toolchain,
- `launcher_android`: created by the compilation step for Android,
- `libtvm_runtime.so`: built for Android.

These are only the binaries related to the launcher itself. To run a model
copy the shared object with the model and the model JSON file over to the
device (both are obtained from relay).  Also, copy all input files for the
model as well.

The following snippet illustrates how to obtain the shared object and the
JSON file from a TFLite model (using Inception V3 as an example):

```
# Skipped imports, etc.

with open("inception_v3.tflite", "rb") as f:
    tflite_model_buf = f.read()
tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

shape_dict = { "input": [1,299,299,3] }
dtype_dict = { "input": "float32" }

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
)

target = tvm.target.hexagon('v68', link_params=True)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, target_host=target, params=params, mod_name="default")

# Save model.so and model.json:
with open('model.json', 'w') as f:
    f.write(lib.get_graph_json())
lib.get_lib().save('model.so')
```

The final thing is to prepare a JSON configuration file for the launcher.
The JSON has two attributes describing the model: `model-library` and
`model-json`, and an attribute `inputs`, which is a list of records, one
for each input file.
An input file record has three attributes: `file`, `shape`, and `dtype`.

Below is an example of the input config file for Inception V3:
```
{
  "model-library": "inceptionv3-float32.so",
  "model-json": "inceptionv3-float32.json",
  "inputs" : [
    {
      "file": "panda_299x299_fp.dat",
      "shape": [1,299,299,3],
      "dtype": "float32"
    }
  ]
}
```

The launcher will then create the output JSON file (with the name given via
`--out_config`) containing information about the execution time and the model
outputs. The output JSON file has three attributes: "pcycles", "usecs" that
contain the execution duration in terms of processor cycles and microseconds
respectivaly, and an attribute `outputs`, which is a list of output file records
whose syntax is identical to the input file records in the input file.
A sample output JSON from running the Inception V3 model may look like
```
{
  "pcycles": 112965680178,
  "usecs": 79532302,
  "outputs": [
    {
      "file": "output0.dat",
      "shape": [1, 1001],
      "dtype": "float32"
    }
  ]
}
```

# Disclaimer

The launcher does not perform any correctness verification. In order to verify
correctness, the user needs to copy the output files from the device and
verify their contents.

This launcher is intended for use with prototyping and does not utilize any
performance acceleration, as such the measured performance may be very poor.
