# TVM in Intel SGX Example

This application demonstrates the use of a simple TVM model in the [Intel SGX](https://software.intel.com/en-us/blogs/2013/09/26/protecting-application-secrets-with-intel-sgx) trusted computing environment.

## Prerequisites

1. A GNU/Linux environment
2. TVM compiled with LLVM and the `tvm` Python module
3. The [Linux SGX SDK](https://github.com/intel/linux-sgx) [link to pre-built libraries](https://01.org/intel-software-guard-extensions/downloads)

## Running the example

`SGX_SDK=/path/to/sgxsdk bash run_example.sh`

If everything goes well, you should see a lot of build messages and below them
the text `It works!`.

## High-level overview

First of all, it helps to think of an SGX enclave as a library that can be called
to perform trusted computation.
In this library, one can use other libraries like TVM.

Building this example performs the following steps:

1. Creates a simple TVM module that computes `x + 1` and save it as a system library.
2. Builds a minimal TVM runtime pack that can load the module.
3. Links the TVM module into an SGX enclave along with some code that runs the module.
4. Compiles and runs an executable that loads the enclave and calls a function
   which invokes the TVM module.

For more information on building, please refer to the `Makefile`.  
For more information on the TVM module, please refer to `../howto_deploy`.  
For more in formation on SGX enclaves, please refer to the [SGX Enclave Demo](https://github.com/intel/linux-sgx/tree/master/SampleCode/SampleEnclave/)
