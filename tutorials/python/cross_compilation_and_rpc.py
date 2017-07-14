"""
Cross Compilation and RPC
=========================
**Author**: `Ziheng Jiang <https://github.com/ZihengJiang/>`_

This tutorial introduces cross compilation and remote device
execution with RPC in TVM.

With cross compilation and RPC, you can compile program on your
local machine then run it on remote device. It is useful when the
resource of remote device is limited, like Raspberry Pi and mobile
platforms, so you do not wish to put the compilation procedure on
the device in order to save time and space.
In this tutorial, I will take Raspberry Pi as our target platform
for example.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import rpc, util

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
# In the following code block, we simply start an RPC server on the
# same machine, for demonstration. This line can be omitted if we
# started an remote server.
#
server = rpc.Server(host='0.0.0.0', port=9090)

######################################################################
# .. note::
#
#   Usually device has limited resources and we only need to build
#   runtime. The idea is we will use TVM compiler on the local server
#   to compile and upload the compiled program to the device and run
#   the device function remotely.
#
#   .. code-block:: bash
#
#     make runtime
#
#   Also make sure that you have set :code:`USE_RPC=1` in your
#   :code:`config.mk`.
#

######################################################################
# Declare and Cross Compile Kernel on Local Machine
# -------------------------------------------------
# Here we will declare a simple kernel with TVM on the local machine:
#
n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
s = tvm.create_schedule(B.op)

######################################################################
# Then we cross compile the kernel:
#

# the target here should be 'llvm -target=armv7l-none-linux-gnueabihf',
# and we use 'llvm' here to make example run locally, see the detailed
# note in the following block
f = tvm.build(s, [A, B], target='llvm', name='myadd')
# save the lib at local temp folder
temp = util.tempdir()
path = temp.relpath('mylib.o')
f.save(path)

######################################################################
# .. note::
#
#   the argument :code:`target` in :code:`build` should be replaced
#   :code:`'llvm'` with the target triple of your device, which might be
#   different for different device. For example, it is
#   :code:`'llvm -target=armv7l-none-linux-gnueabihf'` for my Raspberry
#   Pi. Here we use :code:`'llvm'` directly to make the tutorial runable.
#
#   Usually, you can query the target by execute :code:`gcc -v` on your
#   device, although it may be still a loose configuration.
#
#   Besides :code:`-target`, you can also set other compilation options
#   like:
#
#   * -mtriple=<target triple>
#       Specify the target triple, same as '-target'.
#   * -mcpu=<cpuname>
#       Specify a specific chip in the current architecture to generate code for. By default this is inferred from the target triple and autodetected to the current architecture.
#   * -mattr=a1,+a2,-a3,...
#       Override or control specific attributes of the target, such as whether SIMD operations are enabled or not. The default set of attributes is set by the current CPU.
#       To get the list of available attributes, you can do:
#
#       .. code-block:: bash
#
#         llc -mtriple=<your device target triple> -mattr=help
#
#   These options are consistent with `llc <http://llvm.org/docs/CommandGuide/llc.html>`_.
#   So for my board, to get the best performance, the complete compilation
#   option would be:
#
#   .. code-block:: bash
#
#     llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon
#
#   It is recommended to set target triple and feature set to contain specific
#   feature available, so we can take full advantage of the features of the
#   board.
#   You can find more details about cross compilation attributes from
#   `LLVM guide of cross compilation <https://clang.llvm.org/docs/CrossCompilation.html>`_.

######################################################################
# Run Kernel Remotely by RPC
# --------------------------
# Here we will show you how to run the kernel on the remote device:

# replace host with the ip address of your device
host = '0.0.0.0'
port = 9090
# connect the remote device
remote = rpc.connect(host, port)

######################################################################
# Here we upload the lib to the remote device, then invoke a device local
# compiler for shared lib and load it into device memory. now `f` is a
# remote module object.
remote.upload(path)
f = remote.load_module('mylib.o')

# create array on the remote device
ctx = remote.cpu(0)
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
# the function will run on the remote device
f(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

######################################################################
# When you want to evaluate the performance of the kernel on the remote
# device, it is important to avoid overhead of remote function call.
# :code:`time_evaluator` will returns a remote function that runs the
# function over number times, measures the cost per run on the remote
# device and returns the measured cost.
#
time_f = f.time_evaluator(f.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)

# terminate the server after experiment
server.terminate()

######################################################################
# Summary
# -------
# This tutorial provides a walk through of cross compilation and RPC
# features in TVM.
#
# - Set up RPC server on the remote device.
# - Set up target device configuration to cross compile kernel on the
#   local machine.
# - Upload and run the kernel remotely by RPC API.
