import tvm_graph as tg
import numpy as np
import tvm
from tvm.contrib import util, rpc

tmp = util.tempdir()
sym_fname    = tmp.relpath('net.json')
lib_fname    = tmp.relpath('net.o')
params_fname = tmp.relpath('net.params')

x = tg.Variable('x')
y = tg.Variable('y')
sym = tg.exp(y + x)

shape = (10, 128)
dtype = tvm.float32
na = tvm.nd.array(np.ones(shape).astype(dtype))
nb = tvm.nd.array(np.ones(shape).astype(dtype))
tg.save_params(params_fname, {'x': na, 'y': nb})

target = "llvm"
shapes = {'x': shape, 'y': shape}
tg.compile_graph(sym_fname, lib_fname, sym, target, shapes)

host = '0.0.0.0'
port = 9090
server = rpc.Server(host, port)

remote = rpc.connect(host, port)
ctx = remote.cpu(0)

remote.upload(sym_fname)
remote.upload(lib_fname)
remote.upload(params_fname)

rm = remote.load_executor(sym_fname, lib_fname, ctx)
load_params, run, get_output = rm['load_params'], rm['run'], rm['get_output']
load_params(params_fname)

nc = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
run()
get_output(0, nc)

server.terminate()
