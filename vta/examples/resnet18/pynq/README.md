# Resnet-18 Example on Pynq-based VTA Design

In order to run this example you'll need to have:
* VTA installed
* LLVM 4.0 or newer installed
* TVM installed
* NNVM installed
* MxNet installed
* A Pynq-based RPC server running
* Python packages installed

Required setup time from scratch: ~15 mins.

## VTA installation

Clone the VTA repository in the directory of your choosing:
```bash
git clone git@github.com:uwsaml/vta.git --recursive
```

Update your `~/.bashrc` file to include the VTA python libraries in your `PYTHONPATH` (don't forget to source the newly modified `.bashrc` file!):
```bash
export PYTHONPATH=<vta root>/python:${PYTHONPATH}
```

## LLVM installation

We provide the set of commands to install LLVM 6.0 (stable branch) on Ubuntu Xenial. Note that the [LLVM installation process](apt.llvm.org) can be adapted to different LLVM branches, and operating systems/distros.

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main”
sudo apt-get update
apt-get install clang-6.0 lldb-6.0 lld-6.0
```

To ensure that LLVM 6.0 was properly installed, check that the following command gives the path to your `llvm-config` binary.

```bash
which llvm-config-6.0
```

## TVM installation

Clone the TVM repository in the directory of your choosing:
```bash
git clone git@github.com:dmlc/tvm.git --recursive
```

TVM is rapidly changing, and to ensure stability, we keep track of working TVM checkpoints.
As of now, the TVM checkpoint `168f099155106d1188dbc54ac00acc02900a3c6f` is known to work with VTA.
```bash
cd <tvm root>
git checkout 168f099155106d1188dbc54ac00acc02900a3c6f
```

Before building TVM, copy the `make/config.mk` file into the root TVM directory:
```bash
cd <tvm root>
cp make/config.mk .
```

In the 'config.mk' file sure that:
* `LLVM_CONFIG` points to the `llvm-config` executable which path was derived in the TVM installation instructions above (e.g. `LLVM_CONFIG = /usr/bin/llvm-config-6.0`)
* `USE_RPC` should be set to 1

Launch the compilation, this takes about 5-10 minutes on two threads.
```bash
cd <tvm root>
make -j2
```

Finally update your `~/.bashrc` file to include the TVM python libraries in your `PYTHONPATH` (don't forget to source the newly modified `.bashrc` file!):
```bash
export PYTHONPATH=<tvm root>/python:<tvm root>/topi/python:${PYTHONPATH}
```

## NNVM installation

Clone the NNVM repository from `tqchen` in the directory of your choosing:
```bash
git clone git@github.com:tqchen/nnvm.git --recursive
```

To run this example, we rely on a special branch of NNVM `qt`:
```bash
cd <nnvm root>
git checkout qt
```

Launch the compilation, this takes about a minute on two threads.
```bash
cd <nnvm root>
make -j2
```

Finally update your `~/.bashrc` file to include the NNVM python libraries in your `PYTHONPATH` (don't forget to source the newly modified `.bashrc` file!):
```bash
export PYTHONPATH=<nnvm root>/python:${PYTHONPATH}
```

## MxNet Installation

Follow the [MxNet Installation Instructions](https://mxnet.incubator.apache.org)

## Pynq RPC Server Setup
                                                       
Follow the [Pynq RPC Server Guide](https://github.com/uwsaml/vta/tree/master/apps/pynq_rpc/README.md)

## Python packages

You'll need the following packages to be installed for the example to run properly. You can use `pip` to install those packages:
* `decorator` (for TVM)
* `enum34` (for NNVM)
* `Pillow`
* `wget`

## Running the example

Configure your environment with the following:
```bash
export VTA_PYNQ_RPC_HOST=192.168.2.99
export VTA_PYNQ_RPC_PORT=9091
```

Simply run the following python script:
```bash
python imagenet_predict.py
```

This will run imagenet classification using the ResNet18 architecture on a VTA design that performs 8-bit integer inference, to perform classification on a cat image `cat.jpg`.

The script reports runtime measured on the Pynq board (in seconds), and the top-1 result category:
```
('x', (1, 3, 224, 224))
Build complete...
('TVM prediction top-1:', 281, 'tabby, tabby cat')
t-cost=0.41906
```
