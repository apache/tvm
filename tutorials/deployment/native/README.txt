/*!
 * Native Deploy utility on TVM runtime.
 *
 */

Native deploy utility is to ease the native verification of Model on target with numpy input and output.
Follow the below steps to deploy the compiled model on target and run natively.

Assumption:
Assume compilation is success and we have lib, params & json files.
Assume we have module input dump in .npy format.

Build:
-----
Step 1: Compile libtvm_runtime.so for target.

Update make/config.mk as required (enabling GPU/OpenCL/LLVM ...etc.)
Below is a sample command to compile runtime for OpenCL target on Android Native.

OPENCL_PATH=<OpenCL Path> CXX=<Path to Android NDK>/android-toolchain-arm64/bin/aarch64-linux-android-g++ make lib/libtvm_runtime.so

Step 2: libcnpy depepdency
deploy_native use this library from https://github.com/srkreddy1238/cnpy.git
Please refer README.md under this project for cross compilation details.

Step 3: Compile deploy_native for target.
Below sample command can be used to compile deploy_native for the target.

CNPY_PATH=<path to cnpy compiled> CXX=<Path to Android NDK>/android-toolchain-arm64/bin/aarch64-linux-android-g++ make


Deploy:
------

Step 1: Copy below files to target
Model.so     : NNVM compiled model lib
Model.json   : NNVM compiles graph
Model.params : NNVM compiles params
input_0.npy  : Graph input (input_0 is the input node name).

libtvm_runtime.so : Cross compiled tvm runtime.
libcnpy.so        : Numpy input/output utility lib

deploy_native : deploy utility.

Step 2:
libtvm_runtime.so, libcnpy.so should be copied under sysem lib folder for linking.

Run:
---
Sample command with
    model.so, model.json, model.params.
    input_0 being input node.
    output.npy output dump file
    (1, 125, 11, 11) being output shape.

./deploy_native ./model input_0 output.npy 1 125 11 11


This command will run the model on target and dumps the output into output.npy.


