Installation Guide
==================
This page gives instructions on how to build and install the nnvm compiler package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libnnvm_compiler.so` for linux/osx and `libnnvm_compiler.dll` for windows).
2. Setup for the language packages (e.g. Python Package).

To get started, clone nnvm repo from github. It is important to clone the submodules along, with ```--recursive``` option.
```bash
git clone --recursive https://github.com/dmlc/nnvm
```
For windows users who use github tools, you can open the git shell, and type the following command.
```bash
git submodule init
git submodule update --recursive
```

NNVM compiler depend on TVM and TOPI, so make sure you install them by following [TVM document](http://docs.tvmlang.org/)

## Contents
- [Build the Shared Library](#build-the-shared-library)
- [Python Package Installation](#python-package-installation)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is `libnnvm_compiler.so`
- On Windows the target library is `libnnvm_compiler.dll`

The minimal building requirement is
- A recent c++ compiler supporting C++ 11 (g++-4.8 or higher)

You can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go to the specific language installation section.

## Python Package Installation

The python package is located at python
There are several ways to install the package:

1. Set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `nnvm` on the home directory
   `~`. then we can added the following line in `~/.bashrc`.
    It is ***recommended for developers*** who may change the codes.
    The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ```setup``` again)

    ```bash
    export PYTHONPATH=/path/to/nnvm/python:${PYTHONPATH}
    ```

2. Install nnvm python bindings by `setup.py`:

    ```bash
    # install nnvm package for the current user
    # NOTE: if you installed python via homebrew, --user is not needed during installaiton
    #       it will be automatically installed to your user directory.
    #       providing --user flag may trigger error during installation in such case.
    cd python; python setup.py install --user; cd ..
    ```
