### NNPACK for Multi-Core CPU Support in TVM
[NNPACK](https://github.com/Maratyszcza/NNPACK) is an acceleration package
for neural network computations, which can run on x86-64, ARMv7, or ARM64 architecture CPUs.
Using NNPACK, higher-level libraries like _MXNet_ can speed up
the execution on multi-core CPU computers, including laptops and mobile devices.

***Note***: AS TVM already has natively tuned schedules, NNPACK is here mainly for reference and comparison purpose. 
For regular use prefer native tuned TVM implementation.

_TVM_ supports NNPACK for forward propagation (inference only) in convolution, max-pooling, and fully-connected layers.
In this document, we give a high level overview of how to use NNPACK with _TVM_.

### Conditions
The underlying implementation of NNPACK utilizes several acceleration methods,
including [fft](https://arxiv.org/abs/1312.5851) and [winograd](https://arxiv.org/abs/1509.09308).
These algorithms work better on some special `batch size`, `kernel size`, and `stride` settings than on other,
so depending on the context, not all convolution, max-pooling, or fully-connected layers can be powered by NNPACK.
When favorable conditions for running NNPACKS are not met,

NNPACK only supports Linux and OS X systems. Windows is not supported at present.
The following table explains under which conditions NNPACK will work.

| operation      | conditions |
|:---------      |:---------- |
|convolution     |2d convolution `and` no-bias=False `and` dilate=(1,1) `and` num_group=1 `and` batch-size = 1 or batch-size > 1 && stride = (1,1);|
|pooling         | max-pooling `and` kernel=(2,2) `and` stride=(2,2) `and` pooling_convention=full    |
|fully-connected| without any restrictions |

### Build/Install LLVM
LLVM is required for CPU codegen that needs LLVM.
Since LLVM takes long time to build from source, you can download pre-built version of LLVM from [LLVM Download Page](http://releases.llvm.org/download.html).
For llvm 4.0 you can do the following step : 

```bash
# Add llvm repository in apt source list
echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main" >> /etc/apt/sources.list

# Update apt source list
apt-get update
# Install clang and full llvm
apt-get install -y \
    clang-4.0 \
    clang-4.0-doc \
    libclang-common-4.0-dev \
    libclang-4.0-dev \
    libclang1-4.0 \
    libclang1-4.0-dbg \
    libllvm-4.0-ocaml-dev \
    libllvm4.0 \
    libllvm4.0-dbg \
    lldb-4.0 \
    llvm-4.0 \
    llvm-4.0-dev \
    llvm-4.0-doc \
    llvm-4.0-examples \
    llvm-4.0-runtime \
    clang-format-4.0 \
    python-clang-4.0 \
    libfuzzer-4.0-dev
```

### Build/Install NNPACK

If the trained model meets some conditions of using NNPACK,
you can build TVM with NNPACK support.
Follow these simple steps:  
* Build NNPACK shared library with the following commands. _TVM_ will link NNPACK dynamically.

Note: The following NNPACK installation instructions have been tested on Ubuntu 16.04.

#### Build [Ninja](https://ninja-build.org/)

NNPACK need a recent version of Ninja. So we need to install ninja from source.
```bash
git clone git://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
```

Set the environment variable PATH to tell bash where to find the ninja executable. For example, assume we cloned ninja on the home directory ~. then we can added the following line in ~/.bashrc. 
```bash
export PATH="${PATH}:~/ninja"
```

#### Build [NNPACK](https://github.com/Maratyszcza/NNPACK)

The new CMAKE version of NNPACK download [Peach](https://github.com/Maratyszcza/PeachPy) and other dependencies alone

```bash
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
# Add PIC option in CFLAG and CXXFLAG to build NNPACK shared library
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt
mkdir build
cd build
# Generate ninja build rule and add shared library in configuration
cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install

# Add NNPACK lib folder in your ldconfig
echo "/usr/local/lib" > /etc/ld.so.conf.d/nnpack.conf
sudo ldconfig
```

### Build TVM with NNPACK support

```bash
git clone --recursive https://github.com/dmlc/tvm
```

* Set `USE_NNPACK = 1` in config.mk.
* Set `NNPACK_PATH` to the $(YOUR_NNPACK_INSTALL_PATH)
* Set `LLVM_CONFIG = llvm-config-4.0` depending of llvm version installed

after configuration use `make` to build TVM

```bash
make
make install
```

#### Python Package Installation

The python package for [tvm](https://github.com/dmlc/tvm) depends of [topi](https://github.com/dmlc/tvm/tree/master/topi).
The tvm python package is located at `tvm/python` and topi python package is located in `tvm/topi/python` folder.
There are several ways to install the package, in all these cases the TVM library and TOPI must be present in the python env:

1. Set the environment variable PYTHONPATH to tell python where to find the libraries. For example, assume we cloned tvm on the home directory ~. then we can added the following line in ~/.bashrc. It is recommended for developers who may change the codes. The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call setup again)

```bash
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:${PYTHONPATH}
```

2. Install tvm and topi python bindings by setup.py:

```bash
# install tvm package for the current user
cd topi/python
python setup.py install --user; 
cd ../../python
python setup.py install --user; 
```
