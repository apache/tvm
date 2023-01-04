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


# Performance Benchmark

## Results

See results on wiki page https://github.com/apache/tvm/wiki/Benchmark

## How to Reproduce

To obtain the best performance, we always do auto-tuning for the specific devices and get
the parameters for used kernels. To enable easy reproduction of our results, we release
pre-tuned parameters for popular networks on some common devices.
TVM will download related tuning cache files during compilation.

If you don't have the following listed devices, you can still run these scripts.
You can pick the one that is most similar to your device as argument.
In general, the performance should also be good.

It is recommended that you run tuning by yourself if you have your customized network or devices.
Please follow the tutorial for
[NVIDIA GPU](https://tvm.apache.org/docs/tutorials/autotvm/tune_conv2d_cuda.html),
[ARM CPU](https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_arm.html),
[Mobile GPU](https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_mobile_gpu.html) and
[Adreno GPU](https://www.qualcomm.com/products/features/adreno).

### NVIDIA GPU

Build TVM with LLVM and CUDA enabled. [Help](https://tvm.apache.org/docs/install/from_source.html)

```bash
python3 gpu_imagenet_bench.py --model 1080ti
python3 gpu_imagenet_bench.py --model titanx

# For NVIDIA Jetson TX2, you can run the following command directly on the board,
# or use cross compilation and RPC like what we do for ARM CPU.
python3 gpu_imagenet_bench.py --model tx2
```

### ARM CPU & Mali GPU
For embedded devices, we use RPC infrastructure in TVM to make the management easy.
You need to use it for reproducing benchmark results.

**Note**: We use llvm-4.0 in our tuning environment. Mismatch of the LLVM version during tuning and deployment can influence the performance, so you have to use a same version for reproduction.

0. Build TVM with LLVM enabled. [Help](https://tvm.apache.org/docs/install/from_source.html)

1. Start an RPC Tracker on the host machine
```bash
python3 -m tvm.exec.rpc_tracker
```

2. Register devices to the tracker
* For Linux device
  * Build tvm runtime on your device [Help](https://tvm.apache.org/docs/tutorials/frontend/deploy_model_on_rasp.html#build-tvm-runtime-on-device)
  * Register your device to tracker by
  ```bash
  python3 -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=[DEVICE_KEY]
  ```
  replace `[HOST_IP]` with the IP address of the host machine, `[DEVICE_KEY]` with the name of device.

  E.g. Here is an example command for RK3399,
  `python3 -m tvm.exec.rpc_server --tracker=10.77.1.123:9190 --key=rk3399`, where 10.77.1.123 is the IP address of the tracker.

* For Android device
   * Build and install tvm RPC apk on your device [Help](https://github.com/apache/tvm/tree/main/apps/android_rpc).
     Make sure you can pass the android rpc test. Then you have alreadly known how to register.

3. Verify the device registration
  We can query all registered devices by
  ```bash
  python3 -m tvm.exec.query_rpc_tracker
  ```
  You should be able to find your devices in `Queue Status`. Make sure the registration is correct before going ahead.

  For our test environment, one sample output can be
  ```bash
  Queue Status
  ----------------------------------
  key          total  free  pending
  ----------------------------------
  mate10pro    1      1     0
  p20pro       2      2     0
  pixel2       2      2     0
  rk3399       2      2     0
  rasp3b       8      8     0
  ```

4. Run benchmark
  ```bash
  # ARM CPU
  python3 arm_cpu_imagenet_bench.py --model rasp3b --rpc-key rasp3b
  python3 arm_cpu_imagenet_bench.py --model rk3399 --rpc-key rk3399
  python3 arm_cpu_imagenet_bench.py --model pixel2 --rpc-key pixel2
  python3 arm_cpu_imagenet_bench.py --model p20pro --rpc-key p20pro
  python3 arm_cpu_imagenet_bench.py --model mate10pro --rpc-key mate10pro
  ```

  ```bash
  # Mali GPU
  # NOTE: To make the test environment more stable, we close GUI and lock the frequency
  sudo /etc/init.d/lightdm stop
  sudo -i
  echo performance > /sys/class/misc/mali0/device/devfreq/ff9a0000.gpu/governor
  python3 mobile_gpu_imagenet_bench.py --model rk3399 --rpc-key rk3399
  python3 mobile_gpu_imagenet_bench.py --model rk3399 --rpc-key rk3399 --dtype float16
  ```

### AMD GPU

Build TVM with LLVM and ROCm enabled. [Help](https://tvm.apache.org/docs/install/from_source.html)
```bash
python3 gpu_imagenet_bench.py --model gfx900 --target rocm
```

### Adreno GPU

Adreno benchmarks are automated over the docker - [ci_adreno](https://github.com/apache/tvm/blob/main/docker/Dockerfile.ci_adreno).
Adreno docker share the Android devices from host. It is adviced to have host adb version same as docker, which is ```1.0.41```

Below command runs all (OpenCL native, CLML SDK) the benchmarks over given Android device.
```bash
export ANDROID_SERIAL=<ADB ID>
./tests/scripts/ci.py adreno -b
```
Below command runs all OpenCL native benchmarks over given Android device.
```bash
export ANDROID_SERIAL=<ADB ID>
./tests/scripts/ci.py adreno -n
```
CLML SDK benchmarks require CLML SDK path to be exported and the SDK version should match with target device's SDK version.

Below command runs all CLML SDK benchmarks over given Android device.
```bash
export ADRENO_OPENCL=<CLML SDK PATH>
export ANDROID_SERIAL=<ADB ID>
./tests/scripts/ci.py adreno -c
```

Note: Tuning cache is implicite through tophub repo for all the benchmarks and is tuned over Snapdragon Gen 1.
