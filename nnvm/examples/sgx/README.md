# TVM in Intel SGX Example

This application demonstrates running a ResNet18 using NNVM inside of an
[Intel SGX](https://software.intel.com/en-us/blogs/2013/09/26/protecting-application-secrets-with-intel-sgx) trusted computing environment.

## Prerequisites

1. A GNU/Linux environment
2. NNVM, TVM compiled with LLVM, and their corresponding Python modules
3. The [Linux SGX SDK](https://github.com/intel/linux-sgx) [link to pre-built libraries](https://01.org/intel-software-guard-extensions/downloads)
4. `pip install --user mxnet pillow`

## Running the example

`SGX_SDK=/path/to/sgxsdk bash run_example.sh`

If everything goes well, you should see a lot of build messages and below them
the text `It's a tabby!`.

## High-level overview

First of all, it helps to think of an SGX enclave as a library that can be called
to perform trusted computation.
In this library, one can use other libraries like TVM.

Building this example performs the following steps:

1. Downloads a pre-trained MXNet ResNet and a
   [test image](https://github.com/BVLC/caffe/blob/master/examples/images/cat.jpg)
2. Converts the ResNet to an NNVM graph + library
3. Links the graph JSON definition, params, and runtime library into into an SGX
   enclave along with some code that performs inference.
4. Compiles and runs an executable that loads the enclave and requests that it perform
   inference on the image.
   which invokes the TVM module.

For more information on building, please refer to the `Makefile`.  
For more information on the TVM module, please refer to `../howto_deploy`.  
For more in formation on SGX enclaves, please refer to the [SGX Enclave Demo](https://github.com/intel/linux-sgx/tree/master/SampleCode/SampleEnclave/)
