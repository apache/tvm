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
Overall process starts from getting a model from a framework all the way up to running on target device using ``rtvm` tool.

### Models

Models can be downloaded from well known frameworks like Tensorflow, PyTorch, TFLite, Onnx ..etc.
scripts/download_models.py can be used to download varius well known models from different frameworks.
It will dump various models under model_data in current directory.

```bash
python3  scripts/download_models.py
```

### Auto Tuning
Auto tuning process tunes various operatrors the given model for respective target. Auto tuning for remote devices use ```tvm_rpc``` and we need to setup the rpc environment before we invoke tuning.
Please refer below section RPC setup for the same.

Auto tunng is necessary to obtain best performaning kernels. We can skip this step if we have tuning log already or the tuning cashe is available from tophub (inplicite by TVM compilation process).
Below message indicate that there exists some kernels not optimized for the selected target. In this case we can proceed with tuning to best performance.
```One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.```

with

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
--early-stopping 0 --repeat 30 --rpc-key android --rpc-tracker 127.0.0.1:9120 --trials 1024 \
--tuning-records ./model_data/keras-resnet50/keras-resnet50-records.log --tuner xgb
```

where
```bash
--target="opencl -device=adreno" refers to opencl device on Android device
--target-host="llvm -mtriple=aarch64-linux-gnu" refers to target_host being an ARM64 CPU
Options --early-stopping, --repeat, --trials, --tuner are Auto TVM specific options. Please refer to AutoTVM documentation for more details here.
```

### Compile the model

Compilation step generates TVM compiler output artifacts which need to be taken to target device for deployment.
These artifacts is a compressed archive with kernel shared lib, json with cgaph description and params binary.

Below command will generate the same


```bash
python3 -m tvm.driver.tvmc compile --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
--target="opencl, llvm" --target-llvm-mtriple aarch64-linux-gnu -o keras-resnet50.tar ./model_data/keras-resnet50/resnet50.h5
```

where
```
--cross-compiler : Indicates the cross compiler path for kernel library generation
--target="opencl, llvm" indicates target and host devices
--
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
python3 -m tvm.driver.tvmc run --device="cl" keras-resnet50.tar --rpc-key android --rpc-tracker 127.0.0.1:9120 --print-time
```

This inputs random inputs and validates the execution correctness of the compiled model.

```tvmc``` tool has various options to input custom data, profile, benchmark the execution.


### Deploy

The tar archive generated can be used with ```rtvm``` application in Android native to run the same using tvm_runtime.


# RPC Setup

for Android devices require cross compilation of tvm_rpc (also libtvm_runtime.so which is a dependency) for remote device.
RPC setup involved running tracker on host device and running tvm_rpc on target device.

### Tracker

below command runs the tracker on host over port ```9100```

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

Now we have the rpc setup with ```--rpc-tracker=27.0.0.1:9100``` and ```--rpc-key=android```.


# Target Specific Configuration

Below sections describe device/target specific settings to be used with tvmc tool

### Adreno GPU

Adreno GPU has a docker defined that helps to ease the development environment.

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
--rpc-key android --rpc-tracker 127.0.0.1:9120 --trials 1024 --tuning-records ./model_data/keras-resnet50/keras-resnet50-records.log --tuner xgb

# Tuning produces tuning log ./model_data/keras-resnet50/keras-resnet50.log


# Compilation
python3 -m tvm.driver.tvmc compile --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
--desired-layout NCHW --target="opencl, llvm" --target-opencl-device adreno --target-llvm-mtriple aarch64-linux-gnu \
./model_data/keras-resnet50/resnet50.h5 -o keras-resnet50.tar

# Compilation produces target artifacts keras-resnet50.tar

# Run on adreno device via RPC
# Assuming tracker is running on 127.0.0.1:9190 and target key is "android"
python3 -m tvm.driver.tvmc run --device="cl" keras-resnet50.tar --rpc-key android --rpc-tracker 127.0.0.1:9120 --print-time

```
