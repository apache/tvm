"""
Run Kernel Remotely by RPC
==========================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang/>`_

This tutorial introduces how to use RPC feature in TVM.
With RPC feature, you can compile program on your local machine
then run it on remote device. It is useful when the resource of
remote device is limited, like Raspberry Pi and mobile platforms,
so you do not wish to put the compilation procedure on the device
in order to save time and space.
In this tutorial, I will take Raspberry Pi as our target platform
for example.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To set up a TVM RPC server on the board, we have prepared a script
# so you only need to run this command after following the
# installation guide to install TVM on your device:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
#
# or run the python code directly:
from tvm.contrib import rpc
server = rpc.Server(host='0.0.0.0', port=9090)

######################################################################
# .. note::
#
#   Because we do not need TVM do compilation on the device, we only
#   need to compile the runtime lib for saving time and space:
#
#   .. code-block:: bash
#
#     make runtime -j`nproc`
#
#   Also make sure that you have set :code:`USE_RPC=1` in your
#   :code:`config.mk`.
#

######################################################################
# Declare and Compile Kernel on Local Machine
# -------------------------------------------
# Here we will declare a simple kernel with TVM on the local machine:
#
n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
s = tvm.create_schedule(B.op)

######################################################################
# Then we cross compile the kernel:
#

f = tvm.build(s, [A, B], target='llvm', name='myadd')
# save it at local
f.save('mylib.o')

######################################################################
# .. note::
#
#   the argument :code:`target` in :code:`build` should be replaced by
#   :code:`'llvm'` with the target triple of your device, which might be
#   different for different device. For example, it is
#   :code:`'llvm -target=arm-linux-gnueabihf'` on my
#   Raspberry Pi. Usually, you can query that by execute :code:`gcc -v` on
#   your device. Here we use :code:`'llvm'` directly to make the tutorial runable.
#
#   More details about cross compilation can be found at
#   `here <https://clang.llvm.org/docs/CrossCompilation.html>`_.

######################################################################
# Run Kernel Remotely by RPC
# --------------------------
# Here we will show you how to run the kernel on the remote device:

# replace host with the ip address of your device
host = '0.0.0.0'
port = 9090
# connect the remote device
remote = rpc.connect(host, port)
ctx = remote.cpu(0)
# upload the lib to the remote device
remote.upload('mylib.o')
# now f is a remote module
f = remote.load_module('mylib.o')
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
# the function will run on the remote device
f(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

######################################################################
# You can also use time_evaluator to run the function multiple times on
# the device for the purpose of evaluating the performance of the kernel.
time_f = f.time_evaluator(f.entry_name, ctx, number=10)
cost = time_f(a, b)
print('%g secs/op' % cost)

# terminate the server after experiment
server.terminate()

######################################################################
# Summary
# -------
# This tutorial provides a walk through of RPC feature.
#
# - Set up RPC server on the remote device.
# - Declare and compile kernel on the local machine.
# - Upload and run the kernel remotely by RPC API.
