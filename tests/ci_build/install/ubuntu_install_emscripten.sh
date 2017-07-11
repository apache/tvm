alias make="make -j4"

# Get latest cmake
wget https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.tar.gz
tar xf cmake-3.8.2-Linux-x86_64.tar.gz
export PATH=/cmake-3.8.2-Linux-x86_64/bin/:${PATH}

wget https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz
tar xf emsdk-portable.tar.gz
cd emsdk-portable
./emsdk update
./emsdk install latest
./emsdk activate latest
# Clone and pull latest sdk
./emsdk install clang-incoming-64bit
./emsdk activate clang-incoming-64bit
cd ..
