import numpy as np
import tvm
import tvm_graph as tg
from tvm.contrib import util, rpc

LOCAL = True
tmp = util.tempdir()
sym_fname    = tmp.relpath('net.json')
lib_fname    = tmp.relpath('net.o')
param_fname = tmp.relpath('net.param')

x = tg.Variable('x')
y = tg.Variable('y')
sym = tg.exp(y + x)

shape = (10, 128)
dtype = tvm.float32
na = tvm.nd.array(np.ones(shape).astype(dtype))
nb = tvm.nd.array(np.ones(shape).astype(dtype))
tg.save_params(param_fname, {'x': na, 'y': nb})

target = "llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"
target = "llvm"
shapes = {'x': shape, 'y': shape}
tg.compile_graph(sym_fname, lib_fname, param_fname, sym, target, shapes)

host = 'rasp0'
host = '0.0.0.0'
port = 9090
if LOCAL:
    server = rpc.Server(host, port)

remote = rpc.connect(host, port)
ctx = remote.cpu(0)

rm = remote.load_executor(sym_fname, lib_fname, param_fname, ctx)
run, get_output = rm['run'], rm['get_output']

nc = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
run()
get_output(0, nc)

if LOCAL:
    server.terminate()
