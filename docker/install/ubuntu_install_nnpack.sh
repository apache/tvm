apt-get update && apt-get install -y --no-install-recommends --force-yes git cmake


git clone https://github.com/Maratyszcza/NNPACK NNPACK
cd NNPACK
# TODO: specific tag?
git checkout 1e005b0c2
cd -

mkdir -p NNPACK/build
cd NNPACK/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=. -DNNPACK_INFERENCE_ONLY=OFF -DNNPACK_CONVOLUTION_ONLY=OFF -DNNPACK_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make -j4 && make install
cd -
