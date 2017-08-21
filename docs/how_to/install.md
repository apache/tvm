Installation Guide
==================
This page gives instructions on how to build and install the tvm package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libtvm.so` for linux/osx and `libtvm.dll` for windows).
2. Setup for the language packages (e.g. Python Package).

To get started, clone tvm repo from github. It is important to clone the submodules along, with ```--recursive``` option.
```bash
git clone --recursive https://github.com/dmlc/tvm
```
For windows users who use github tools, you can open the git shell, and type the following command.
```bash
git submodule init
git submodule update
```

## Contents
- [Build the Shared Library](#build-the-shared-library)
- [Python Package Installation](#python-package-installation)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is `libtvm.so`
- On Windows the target library is `libtvm.dll`

The minimal building requirement is
- A recent c++ compiler supporting C++ 11 (g++-4.8 or higher)

You can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go to the specific language installation section.

### Building on Windows

TVM support build via MSVC using cmake. The minimum required VS version is **Visual Studio Community 2015 Update 3**. In order to generate the VS solution file using cmake,
make sure you have a recent version of cmake added to your path and then from the tvm directory:

```bash
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ..
```
This will generate the VS project using the MSVC 14 64 bit generator. Open the .sln file in the build directory and build with Visual Studio.

### Customized Building

Install prerequisites first: 

```bash
sudo apt-get update
sudo apt-get install -y python gcc libtinfo-dev
```

The configuration of tvm can be modified by ```config.mk```
- First copy ```make/config.mk``` to the project root, on which
  any local modification will be ignored by git, then modify the according flags.
- TVM optionally depends on LLVM. LLVM is required for CPU codegen that needs LLVM.
  - LLVM 4.0 or higher is needed for build with LLVM
  - By default CUDA and OpenCL code generator do not require llvm.

## Python Package Installation

The python package is located at python
There are several ways to install the package:

1. Set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `tvm` on the home directory
   `~`. then we can added the following line in `~/.bashrc`.
    It is ***recommended for developers*** who may change the codes.
    The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ```setup``` again)

    ```bash
    export PYTHONPATH=/path/to/tvm/python:${PYTHONPATH}
    ```

2. Install tvm python bindings by `setup.py`:

    ```bash
    # install tvm package for the current user
    # NOTE: if you installed python via homebrew, --user is not needed during installaiton
    #       it will be automatically installed to your user directory.
    #       providing --user flag may trigger error during installation in such case.
    cd python; python setup.py install --user
    # or install tvm package system wide
    cd python; sudo python setup.py install
    ```
