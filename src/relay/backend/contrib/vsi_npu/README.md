# Versilicon NPU solution on TVM

In this implementation, we enabled offload AI-workloads to versilicon's neural network processor.

# Terms
NBG(network binary graph)

    NBG is the executeble format for the NPU, we can compile it from host server and deployment it to a target.

TIM-VX: (**T**ensor **I**nterface **M**odule)[https://github.com/VeriSilicon/TIM-VX]

# Implementation details
We have four parts in this implemetation.
1. register vsi-npu supported operator 
    python/tvm/relay/op/contrib/vsi_npu.py defined supported operator and specific patterns we can support in the NPU.
2. implemented nbg codegen in compilation
    src/relay/backend/contrib/vsi_npu/
3. implemented runtime to execute nbg
    src/runtime/contrib/vsi_npu/
4. test scripts
    test/python/contrib/test_vsi_npu/
5. CMake build script
    cmake/modules/contrib/VsiNpu.cmake

# Build from source 

## Build TIM-VX from source

## Build tvm as compiler
This step can be executed with a x86 host or arm based target. If you do cross build for your target,
just add toolchain configuration for cmake.

```sh
    mkdir host_compiler_build
    cd host_compiler_build
    cp ../cmake/config.cmake ./
    # NOTE: config llvm by set USE_LLVM to the llvm-config
    # add set(USE_VSI_NPU ON) to config.cmake, you can do it with cmake command option too
    # To speed up build, we can disable other backend in this configuration file
    cmake -DCMAKE_BUILD_TYPE=Debug -DTIM_VX_INSTALL_DIR=<full_path_to_tim_vx_install> ..
    make tvm -j12
```

## Build tvm as runtime 
Usually, NBG runtime will be deployed to embedded device. We need to prepare cross-compile-toolchain for cmake firstly.

```bash
   mkdir target_runtime_build
   cd target_runtime_build
   cp ../cmake/config.cmake ./
    # add set(USE_VSI_NPU ON) to config.cmake, you can do it with cmake command option too
   cmake -DCMAKE_BUILD_TYPE=Debug -DTIM_VX_INSTALL_DIR=<full_path_to_tim_vx_target_build_install_dir> \
         -DCMAKE_TOOLCHAIN_FILE=<path_to_cross_compile_toolchain.make> ..
   make runtime -j12
```

# Run the test

## Option: prepare test models
{todo: model and download link, tensorflow hosted models}

## start runtime on the target as a service
In this step, we need install some python package required by TVM python packages.

We need copy or map the while TVM source code(python part and target_runtime_build) to the device. 
```bash
   # make sure NPU driver installed and can work without error (check dmesg after you insmod galcore)
   # 0.Append tvm/python 
   export PYTHONPATH=<path/to/tvm/ptyon>:$PYTHONPATH
   # 1.Setup libraries
   export LD_LIBRARY_PATH=<path/to/versilicon/driver/sdk>:<path/to/tim-vx/target/install>:<path/to/tvm/target_runtime_build/>:$LD_LIBRARY_PATH
   # 2. start service on given TCP port
   python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

## execute test from host

```bash
    # 0. Set correct NPU target name for your device, you can learned this from your soc vendor
    export VSIMULATOR_CONFIG=PID_0x99
    # 1. Set up testcase, please refer model list from tests/python/contrib/test_vsi_npu/test_vsi_tflite_model_all.py
    export TFLITE_MODEL="<full/path/to/mobilenet_v1_1.0_224_quant.tflite>"
    # 2. Setup corss compile toolchain configuration 
    export PATH=<cross-compiler-path>:$PATH
    export CROSS_CC=<cross-compiler-binary-name>
    export ROOTFS=<rootfs-for-cross-compile>
    # 3. Remote service configuration
    export RPC_HOST=<target device ip address>
    export RPC_PORT=<TCP port exposed by the service>
    # debug purpose
    export MOD_PATH="<any/folder/can/write>"
    export MOD_NAME="NBG.so" # could be any name, for debug purpose
    # 4. Add TVM to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=<path/to/host_compiler_build/>
    # 5. Execute test
    python3 tests/python/contrib/test_vsi_npu/test_vsi_tflite_model_all.py
```
