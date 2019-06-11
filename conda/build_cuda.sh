conda build --output-folder /workspace/conda/pkg --variants "{cuda: True, cuda_version: $CUDA_VERSION}" /workspace/conda/tvm-libs
