cd /usr
git clone https://github.com/dmlc/tvm --recursive
cd /usr/tvm
echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j10
