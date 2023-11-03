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


# Native Inference application for CPP Native

Native inference tool ```rtvm``` helps in deploying TVM compiled models from a standalone cpp environment.
Overall process starts from getting a model from a framework all the way up to running on target device using `rtvm` tool.

### Models

Models can be downloaded from well known frameworks like Tensorflow, PyTorch, TFLite, Onnx ..etc.
scripts/download_models.py has a reference to prepare sample network ```resnet50``` from keras framework.

```bash
python3  scripts/download_models.py
```

### Auto Tuning
Auto tuning process tunes various operatrors the given model for respective target. Auto tuning for remote devices use ```tvm_rpc``` and we need to setup the rpc environment before we invoke tuning.
Please refer below section [RPC setup](#rpc-setup) for the same.

Auto tunng is necessary to obtain best performaning kernels. We can skip this step if we have tuning log already or the tuning cache is available from tophub (implicite by TVM compilation process).
Below message indicate that there exists some kernels not optimized for the selected target. In this case we can proceed with tuning to best performance.
```One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.```

with below environment from [RPC setup](#rpc-setup)
``` bash
tvm tracker running on ```TVM_TRACKER_HOST```
tracker port being ```TVM_TRACKER_PORT```
rpc device access key being ```TVM_RPC_KEY```
the model to be tuned being ```./model_data/keras-resnet50/resnet50.h5```
```

the below command we can generate the tuning cache to file ```./model_data/keras-resnet50/keras-resnet50.log```

```bash
python3 -m tvm.driver.tvmc tune --target="opencl" --target-host="llvm -mtriple=aarch64-linux-gnu" \
./model_data/keras-resnet50/resnet50.h5 -o ./model_data/keras-resnet50/keras-resnet50.log \
--early-stopping 0 --repeat 30 --rpc-key ${TVM_RPC_KEY} --rpc-tracker ${TVM_TRACKER_HOST}:${TVM_TRACKER_PORT} --trials 1024 \
--tuning-records ./model_data/keras-resnet50/keras-resnet50-records.log --tuner xgb
```

where
```bash
--target="opencl" refers to opencl device on Android device
--target-host="llvm -mtriple=aarch64-linux-gnu" refers to target_host being an ARM64 CPU
Options --early-stopping, --repeat, --trials, --tuner are Auto TVM specific options.
```
Please refer to AutoTVM documentation for more details [here](https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html?highlight=autotvm).

### Compile the model

Compilation step generates TVM compiler output artifacts which need to be taken to target device for deployment.
These artifacts is a compressed archive with kernel shared lib, json with graph description and params binary.

Below command will generate the same


```bash
python3 -m tvm.driver.tvmc compile --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
--target="opencl, llvm" --target-llvm-mtriple aarch64-linux-gnu -o keras-resnet50.tar ./model_data/keras-resnet50/resnet50.h5
```

where
```
--cross-compiler : Indicates the cross compiler path for kernel library generation
--target="opencl, llvm" indicates target and host devices
```

### Test Run via RPC

At this stage we can verify the generated compiler output for execution correctness over the RPC setup interface.
Below command can run the compiled output on remote target device.

with

``` bash
tvm tracker running on ```TVM_TRACKER_HOST```
tracker port being ```TVM_TRACKER_PORT```
rpc device access key being ```TVM_RPC_KEY```
compilation out being keras-resnet50.tar
```

```bash
python3 -m tvm.driver.tvmc run --device="cl" keras-resnet50.tar --rpc-key ${TVM_RPC_KEY} --rpc-tracker ${TVM_TRACKER_HOST}:${TVM_TRACKER_PORT} --print-time
```

This inputs random inputs and validates the execution correctness of the compiled model.

```tvmc``` tool has various options to input custom data, profile the model and benchmark the execution.


### Deployment Run

Now we will verify the deployment run of the compiled model using ```rtvm``` tool on target device without any RPC or host based execution.

We need to extract the tar achive on target device. We can copy the extracted contents of ```keras-resnet50.tar``` under Android temp folder at ```/data/local/tmp/keras-resnet50/```

Also copy the cross compiled tool ```rtvm``` and ```libtvm_runtime.so``` to ```data/local/tmp/```

```rtvm``` usage can be quired as below
```bash
Android:/data/local/tmp $ LD_LIBRARY_PATH=./ ./rtvm
Command line usage
--model        - The folder containing tvm artifacts(mod.so, mod.param, mod.json)
--device       - The target device to use {llvm, opencl, cpu, cuda, metal, rocm, vpi, oneapi}
--input        - Numpy file for the model input (optional and we use random of not given)
--output       - Numpy file name to dump the model output as numpy
--dump-meta    - Dump model meta information
--pre-compiled - The file name of a file where pre-compiled programs should be stored
--profile      - Profile over all execution
--dry-run      - Profile after given dry runs, default 10
--run-count    - Profile for given runs, default 50
--zero-copy    - Profile with zero copy api

  Example
  ./rtvm --model=keras-resnet50 --device="opencl" --dump-meta
  ./rtvm --model=keras-resnet50 --device="opencl" --input input.npz --output=output.npz
```

```rtvm``` can run the model using no inputs (just a dry run without any valid inputs) and also with specific input supplied as a numpy npz format file.

We can create npz dump for all inputs by saving the dict object as shown below.

With ```keras-resnet50``` having one  input ```input_1``` with shape ```[1, 224, 224, 3]``` and dtype ```float32```

```
# Random initilization
input1 = np.random.uniform(low=-1, high=1, size=(1, 224, 224, 3)).astype("float32")
dataset = {"input_1": input1}
np.savez("input.npz", **dataset)
```

Copy ```input.npz``` also to the target device as ```/data/local/tmp/input.npz```


Now, on Android shell we can do a dry run as well as with specific input as shown below.
```bash
# Query meta data information
Android:/data/local/tmp/ $ LD_LIBRARY_PATH=./ ./rtvm --model=keras-resnet50 --device=opencl --dump-meta
. . . . . .
Meta Information:keras-resnet50
    Number of Inputs:183
    Number of Outputs:1
    Input MetaInfo:
        Input:input_1
            DType:float32
            Shape:[1, 224, 224, 3]
    Output MetaInfo:
        Output:tvmgen_default_fused_nn_softmax
            DType:float32
            Shape:[1, 1000]
. . . . . .

# Dry run with out any inputs
Android:/data/local/tmp/ $ LD_LIBRARY_PATH=./ ./rtvm --model=keras-resnet50 --device=opencl
Model         = keras-resnet50
Device        = opencl
Input         =
Output        =
Dump Metadata = False
TVMRunner Constructor:keras-resnet50 Devices:opencl
TVMRunner Load:keras-resnet50
TVMRunner::GetMetaInfo
Executing dry run ...
Set Random Input for :input_1
TVMRunner::GetInputMemSize:input_1
Random Input Size:602112  bytes
TVMRunner::SetInput (Raw)
TVMRunner::Run
Get Output for :tvmgen_default_fused_nn_softmax
TVMRunner::GetOutputMemSize:tvmgen_default_fused_nn_softmax
TVMRunner::GetOutput (Raw)
Output Size:4000  bytes


# Run with input and dump output as npz file
Android:/data/local/tmp/ $ LD_LIBRARY_PATH=./ ./rtvm --model=keras-resnet50 --device=opencl --input=input.npz --output=output.npz
Model         = keras-resnet50
Device        = opencl
Input         = input.npz
Output        = output.npz
Dump Metadata = False
TVMRunner Constructor:keras-resnet50 Devices:opencl
TVMRunner Load:keras-resnet50
TVMRunner::GetMetaInfo
Executing with Input:input.npz Output:output.npz
TVMRunner::SetInput (Numpy):input.npz
Set Numpy Input for :input_1
TVMRunner::Run
TVMRunner::GetOutput (Numpy):output.npz
Get Output for :tvmgen_default_fused_nn_softmax
Output Size:4000  bytes
```

output.npz contains the modle outputs. Below is a quick look of its contents.
```bash
tvm-host:~$ unzip -l output.npz
Archive:  output.npz
  Length      Date    Time    Name
---------  ---------- -----   ----
     4080  1980-00-00 00:00   tvmgen_default_fused_nn_softmax.npy
---------                     -------
     4080                     1 file

```

Building ```cpp_rtvm``` produces ```libtvm_runner.so```, a simplified interface that rtvm use internally for loading and executing tvm compiled models from C/C++ environments.
```tvm_runner.h``` describes the interface definition here. Alternatively pro users can use TVM's [c_native_api](https://github.com/apache/tvm/blob/main/include/tvm/runtime/c_runtime_api.h) interface for more access to TVM features.


# RPC Setup

For Android devices we require cross compilation of tvm_rpc (also libtvm_runtime.so which is a dependency) for remote device.
RPC setup involves running tracker on host device and running tvm_rpc on target device.

### Tracker

Below command runs the tracker on host over port ```9100```

```bash
python3 -m tvm.exec.rpc_tracker --host 127.0.0.1 --port 9100"
```
### RPC on Target

With ```abcd1234ef``` being adb device id and tvm_rpc (and libtvm_runtime.so) is pushed to target device at ```/data/local/tmp/tvm_rpc/```

```bash
export ANDROID_SERIAL=abcd1234ef
# Below settings will reroute networking tcm connections on devices to host device via adb interface
adb reverse tcp:9100 tcp:9100
adb forward tcp:5000 tcp:5000
# Run the tvm_rpc on device
env adb shell "cd /data/local/tmp/tvm_rpc; killall -9 tvm_rpc; \
LD_LIBRARY_PATH=/data/local/tmp/tvm_rpc/ ./tvm_rpc server --host=0.0.0.0 --port=5000 --port-end=5010 --tracker=127.0.0.1:9100 --key=android
```

Now we have the rpc setup with ```TVM_TRACKER_HOST=127.0.0.1```, ```TVM_TRACKER_PORT=9100``` and ```TVM_RPC_KEY=android```.

We can also check connected and available devices on tracker as shown below.

```bash
python3 -m tvm.exec.query_rpc_tracker --port ${TVM_TRACKER_PORT}
Tracker address 127.0.0.1:9100

Server List
------------------------------
server-address           key
------------------------------
       127.0.0.1:5000    server:android
------------------------------

Queue Status
-------------------------------
key       total  free  pending
-------------------------------
android   1      1     0
-------------------------------
```


# Target Specific Configuration

Below sections describe device/target specific settings to be used with ```tvmc``` tool.

### Adreno GPU

Adreno GPU has a docker definition that helps to ease the development environment.

We can build the docker image by using below command from TVM repo.

```bash
./docker/build.sh ci_adreno
docker tag tvm.ci_adreno ci_adreno
```

Below command builds host and target rpc components for Adreno and drops into an interactive shell.

```bash
./tests/scripts/ci.py adreno -i
```

Also, one can build with Adreno OpenCLML SDK support

```bash
export ADRENO_OPENCL=<Path to OpenCLML SDK>
./tests/scripts/ci.py adreno -i
```

Above command produces
```build-adreno``` which is host build
```build-adreno-target``` which contains cross compiled tvm_rpc and libtvm_runtime.so


Below options to be used for Adreno GPU while working with tvmc

* Tuning

  ```
  --target="opencl -device=adreno"
  --target-host="llvm -mtriple=aarch64-linux-gnu"
  ```

* Compilation

  ```
  --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang
  --target="opencl, llvm"
  --target-opencl-device adreno
  --target-llvm-mtriple aarch64-linux-gnu
  ```

  While enabling CLML just need to specify below target option for compilation.
  ```--target="opencl, clml, llvm"```


* Running

  ```--device="cl"```


For example with a model from keras ```./model_data/keras-resnet50/resnet50.h5```


```bash
# Tuning
python3 -m tvm.driver.tvmc tune --desired-layout NCHW --target="opencl -device=adreno" --target-host="llvm -mtriple=aarch64-linux-gnu" \
./model_data/keras-resnet50/resnet50.h5 -o ./model_data/keras-resnet50/keras-resnet50.log --early-stopping 0 --repeat 30 \
--rpc-key ${TVM_RPC_KEY} --rpc-tracker {TVM_TRACKER_HOST}:{TVM_TRACKER_PORT} --trials 1024 --tuning-records ./model_data/keras-resnet50/keras-resnet50-records.log --tuner xgb

# Tuning produces tuning log ./model_data/keras-resnet50/keras-resnet50.log


# Compilation
python3 -m tvm.driver.tvmc compile --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
--desired-layout NCHW --target="opencl, llvm" --target-opencl-device adreno --target-llvm-mtriple aarch64-linux-gnu \
./model_data/keras-resnet50/resnet50.h5 -o keras-resnet50.tar

# Compilation produces target artifacts keras-resnet50.tar

# Run on adreno device via RPC
python3 -m tvm.driver.tvmc run --device="cl" keras-resnet50.tar --rpc-key ${TVM_RPC_KEY} --rpc-tracker {TVM_TRACKER_HOST}:{TVM_TRACKER_PORT} --print-time

```

# Use pre-compiled OpenCL kernels
Using pre-compiled programs might significantly improve inference time of the
first run. E.g. for topology with ~300 kernels compilation time on Adreno was
about 26 seconds. But after dumping compiled programs to binary files and reuse
them on the next runs, the compilation time was significantly decreased (more
than 1000 times) and starts to be around 25 ms.

To use such functionality, the developer have to pass parameter `--pre-compiled`
to the `rtvm` and specify the file name where pre-compiled programs will be
stored. If the pre-compiled file name was passed to the `rtvm` then After method
`Load`, method `UsePreCompiledProgram` is called. This method loads pre-compiled
programs if the file exists. In opposite case the file will be created and
pre-compiled programs will be saved to this file.

# Performnace Profiling Options
The tool has added few options to measure wall clock performance of the given model on Target natively.
--profile : Can turn on the profiling
--dry-run : The number of times dry run the model before mearuring the performance. Default value os 10
--run-count : The number times to run the model and take an average. Default value is 50.
--zero-copy: This option enables graph runtime zero copy to be used for input and output than byte copy to DLTensor.

Performance profile options dumps information summary as given below.
     Module Load              :27 ms
     Graph Runtime Create     :11 ms
     Params Read              :15 ms
     Params Set               :41 ms
     Pre Compiled Progs Load  :24 ms
Total Load Time     :118 ms
Average ExecTime    :27 ms
Unload Time         :35.9236 ms
