# Resnet-18 Example on Pynq-based VTA Design

In order to run this example you'll need to have:
* VTA installed
* TVM installed
* NNVM installed
* A Pynq-based RPC server running

## VTA installation

Clone the VTA repository in the directory of your choosing:
```bash
git clone git@github.com:uwsaml/vta.git --recursive
```

Update your `~/.bashrc` file to include the VTA python libraries in your `PYTHONPATH` (don't forget to source the newly modified `.bashrc` file!):
```bash
export PYTHONPATH=<vta root>/python:${PYTHONPATH}
```

## TVM installation

Clone the TVM repository in the directory of your choosing:
```bash
git clone git@github.com:dmlc/tvm.git --recursive
```

TVM is rapidly changing, and to ensure stability, we keep track of working TVM checkpoints.
As of now, the TVM checkpoint `e4c2af9abdcb3c7aabafba8084414d7739c17c4c` is known to work with VTA.
```bash
git checkout e4c2af9abdcb3c7aabafba8084414d7739c17c4c
```

Before building TVM, copy the `make/config.mk` file into the root TVM directory:
```bash
cd <tvm root>
cp make/config.mk .
```

In the 'config.mk' file sure that:
* `LLVM_CONFIG` points to the llvm-config executable (e.g. `LLVM_CONFIG = /usr/bin/llvm-config-4.0`). You'll need to have llvm4.0 installed or later.
* `USE_RPC` should be set to 1

Launch the compilation, this takes about 5 minutes.
```bash
cd <tvm root>
make -j4
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

To run this example, we rely on a special branch of NNVM: `qt`:
```bash
cd <nnvm root>
git checkout qt
```

Launch the compilation, this takes less a minute.
```bash
cd <nnvm root>
make -j4
```

Finally update your `~/.bashrc` file to include the NNVM python libraries in your `PYTHONPATH` (don't forget to source the newly modified `.bashrc` file!):
```bash
export PYTHONPATH=<nnvm root>/python:${PYTHONPATH}
```

## Pynq RPC Server Setup

Follow the [Pynq RPC Server Guide](https://github.com/saml/vta/tree/master/apps/pynq_rpc/README.md)

## Running the example

Simply run the following python script:
```bash
python imagenet_predict.py
```

This will run imagenet classification using the ResNet18 architecture on a VTA design that performs 8-bit integer inference, to perform classification on a cat image `cat.jpg`.

The script reports runtime measured on the Pynq board, and the top-1 result category:
```
('x', (1, 3, 224, 224))
Build complete...
('TVM prediction top-1:', 281, 'tabby, tabby cat')
t-cost=0.41906
```