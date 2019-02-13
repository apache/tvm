cd /usr
git clone https://github.com/dmlc/tvm --recursive
cd /usr/tvm
echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_SGX /opt/sgxsdk\) >> config.cmake
echo set\(RUST_SGX_SDK /opt/rust-sgx-sdk\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j10
