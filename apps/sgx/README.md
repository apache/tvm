# TVM in Intel SGX Example

This application demonstrates the use of a simple TVM model in the [Intel SGX](https://software.intel.com/en-us/blogs/2013/09/26/protecting-application-secrets-with-intel-sgx) trusted computing environment.

## Prerequisites

1. The TVM premade Docker image

or

1. A GNU/Linux environment
2. TVM compiled with LLVM and SGX; and the `tvm` Python module
3. The [Linux SGX SDK](https://github.com/intel/linux-sgx) [link to pre-built libraries](https://01.org/intel-software-guard-extensions/downloads)
4. [Rust](https://rustup.sh)
5. The [rust-sgx-sdk](https://github.com/baidu/rust-sgx-sdk)
6. [xargo](https://github.com/japaric/xargo)

Check out the `/tvm/install/ubuntu_install_sgx.sh` for the commands to get these dependencies.

## Running the example

If using Docker, start by running

```
git clone https://github.com/dmlc/tvm.git
docker run --rm -it -v $(pwd)/tvm:/mnt tvmai/ci-cpu /bin/bash
```
then, in the container
```
cd /mnt
mkdir build && cd build
cmake .. -DUSE_LLVM=ON -DUSE_SGX=/opt/sgxsdk -DRUST_SGX_SDK=/opt/rust-sgx-sdk
make -j4
cd ../apps/sgx
```

`bash run_example.sh`

If everything goes well, you should see a lot of build messages and below them
the text `It works!`.

## High-level overview

First of all, it helps to think of an SGX enclave as a library that can be called
to perform trusted computation.
In this library, one can use other libraries like TVM.

Building this example performs the following steps:

1. Creates a simple TVM module that computes `x + 1` and save it as a system library.
2. Builds a TVM runtime that links the module and allows running it using the TVM Python runtime.
3. Packages the bundle into an SGX enclave
4. Runs the enclave using the usual TVM Python `module` API

For more information on building, please refer to the `Makefile`.  
For more information on the TVM module, please refer to `../howto_deploy`.  
For more in formation on SGX enclaves, please refer to the [SGX Enclave Demo](https://github.com/intel/linux-sgx/tree/master/SampleCode/SampleEnclave/)
